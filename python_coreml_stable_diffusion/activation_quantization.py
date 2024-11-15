#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import logging
import operator

import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel('INFO')

import argparse
import gc
import json
import os
import pickle
from copy import deepcopy

import coremltools as ct
import numpy as np
from coremltools.optimize.torch.quantization import (
    LinearQuantizer, LinearQuantizerConfig, ModuleLinearQuantizerConfig)
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from python_coreml_stable_diffusion import attention
from python_coreml_stable_diffusion import unet
from python_coreml_stable_diffusion.layer_norm import LayerNormANE
from python_coreml_stable_diffusion.torch2coreml import compute_psnr
from python_coreml_stable_diffusion.unet import Einsum

attention.SPLIT_SOFTMAX = True

CALIBRATION_DATA = [
    "image of a transparent tall glass with ice, fruits and mint, photograph, commercial, food, warm background, beautiful image, detailed",
    "picture of dimly lit living room, minimalist furniture, vaulted ceiling, huge room, floor to ceiling window with an ocean view, nighttime, 3D render, high quality, detailed",
    "modern office building, 8 stories tall, glass and steel, 3D render style, wide angle view, very detailed, sharp photographic image, in an office park, bright sunny day, clear blue skies, trees and landscaping",
    "cute small cat sitting in a movie theater eating popcorn, watching a movie, cozy indoor lighting, detailed, digital painting, character design",
    "a highly detailed matte painting of a man on a hill watching a rocket launch in the distance by studio ghibli, volumetric lighting, octane render, 4K resolution, hyperrealism, highly detailed, insanely detailed, cinematic lighting, depth of field",
    "an undersea world with several of fish, rocks, detailed, realistic, photograph, amazing, beautiful, high resolution",
    "large ocean wave hitting a beach at sunset, photograph, detailed",
    "pocket watch on a table, close up. macro, sharp, high gloss, brass, gears, sharp, detailed",
    "pocket watch in the style of pablo picasso, painting",
    "majestic royal tall ship on a calm sea, realistic painting, cloudy blue sky, in the style of edward hopper",
    "german castle on a mountain, blue sky, realistic, photograph, dramatic, wide angle view",
    "artificial intelligence, AI, concept art, blue line sketch",
    "a humanoid robot, concept art, 3D render, high quality, detailed",
    "donut with sprinkles and a cup of coffee on a wood table, detailed, photograph",
    "orchard at sunset, beautiful, photograph, great composition, detailed, realistic, HDR",
    "image of a map of a country, tattered, old, styled, illustration, for a video game style",
    "blue and green woven fibers, nano fiber material, detailed, concept art, micro photography",
]

RANDOM_TEST_DATA = [
    "a black and brown dog standing outside a door.",
    "a person on a motorcycle makes a turn on the track.",
    "inflatable boats sit on the arizona river, and on the bank",
    "a white cat sitting under a white umbrella",
    "black bear standing in a field of grass under a tree.",
    "a train that is parked on tracks and has graffiti writing on it, with a mountain range in the background.",
    "a cake inside of a pan sitting in an oven.",
    "a table with paper plates and flowers in a home",
]

def get_coreml_inputs(sample_inputs):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]

def convert_to_coreml(torchscript_module, sample_inputs):
    logger.info("Converting model to CoreML..")
    coreml_model = ct.convert(
        torchscript_module,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        inputs=get_coreml_inputs(sample_inputs),
        outputs=[ct.TensorType(name="noise_pred", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=True,
    )

    return coreml_model


def unet_data_loader(data_dir, device='cpu', calibration_nsamples=None):
    """
    Load calibration data from specified path.
    Limit number of samples to calibration_nsamples, if specified.
    """
    dataloader = []
    skip_load = False
    for file in sorted(os.listdir(data_dir)):
        if file.endswith('.pkl'):
            filepath = os.path.join(data_dir, file)
            with open(filepath, 'rb') as data:
                try:
                    while not skip_load:
                        unet_data = pickle.load(data)
                        for input in unet_data:
                            dataloader.append([x.to(torch.float).to(device) for x in input])

                            if calibration_nsamples:
                                if len(dataloader) >= calibration_nsamples:
                                    skip_load = True
                                    break
                except EOFError:
                    pass
        if skip_load: 
            break

    logger.info(f"Total calibration samples: {len(dataloader)}")
    return dataloader

def quantize_module_config(module_name):
    """
    Generate quantization config to apply W8A8 quantization for specified module.
    Rest of the model is kept in FP32 precision.
    """
    config = LinearQuantizerConfig(
        global_config=ModuleLinearQuantizerConfig(
            milestones=[0, 1000, 1000, 0],
            weight_dtype=torch.float32,
            activation_dtype=torch.float32,
        ),
        module_name_configs={
            module_name: ModuleLinearQuantizerConfig(
                quantization_scheme="symmetric",
                milestones=[0, 1000, 1000, 0],
            ),
        },
    )
    return config

def quantize_cumulative_config(skip_conv_layers, skip_einsum_layers):
    """
    Generate quantization config to apply W8A8 quantization.
    Skipped layers are kept in W8A32 precision.
    """
    logger.info(f"Skipping {len(skip_conv_layers)} conv layers and {len(skip_einsum_layers)} einsum layers")
    w8config = ModuleLinearQuantizerConfig(
            quantization_scheme="symmetric",
            milestones=[0, 1000, 1000, 0],
            activation_dtype=torch.float32)
    conv_modules_config = {name: w8config for name in skip_conv_layers}
    einsum_modules_config = {name: w8config for name in skip_einsum_layers}
    module_name_config = {}
    module_name_config.update(conv_modules_config)
    module_name_config.update(einsum_modules_config)

    config = LinearQuantizerConfig(
        global_config=ModuleLinearQuantizerConfig(
            quantization_scheme="symmetric",
            milestones=[0, 1000, 1000, 0],
        ),
        module_name_configs=module_name_config,
        module_type_configs={
            torch.cat: None,
            torch.nn.GroupNorm: None,
            torch.nn.SiLU: None,
            torch.nn.functional.gelu: None,
            operator.add: None,
        },
    )
    return config

def quantize(model, config, calibration_data):
    """
    Apply post training activation quantization to specified model, using calibration data
    """
    submodules = dict(model.named_modules(remove_duplicate=True))
    layer_norm_modules = [key for key, val in submodules.items() if isinstance(val, LayerNormANE)]
    non_traceable_module_names = layer_norm_modules + [
        "time_proj",
        "time_embedding",
    ]

    # Mark certain modules as non-traceable to make the UNet model fx traceable
    config.non_traceable_module_names = non_traceable_module_names
    config.preserved_attributes = ['config', 'device']

    sample_input = calibration_data[0]
    quantizer = LinearQuantizer(model, config)
    logger.info("Preparing model for quantization")
    prepared_model = quantizer.prepare(example_inputs=(sample_input,))
    prepared_model.eval()

    quantizer.step()

    logger.info("Calibrate")
    for idx, data in enumerate(calibration_data):
        logger.info(f"Calibration data sample: {idx}")
        prepared_model(*data)

    logger.info("Finalize model")
    quantized_model = quantizer.finalize()
    return quantized_model

def get_quantizable_modules(unet):
    quantizable_modules = []
    for name, module in unet.named_modules():
        if len(list(module.children())) > 0:
            continue
        if type(module) == torch.nn.modules.conv.Conv2d:
            quantizable_modules.append(('conv', name))
        if type(module) == Einsum:
            quantizable_modules.append(('einsum', name))

    return quantizable_modules

def recipe_overrides_for_inference_speedup(conv_layers, skipped_conv):
    """
    Quantize the slowest conv layers, even if in skipped set based on PSNR, for good inference speedup
    """
    for layer in conv_layers:
        if "up_blocks" in layer and "resnets" in layer and "conv1" in layer:
            if layer in skipped_conv:
                logger.info(f"removing {layer}")
                skipped_conv.remove(layer)
        if "upsamplers" in layer:
            if layer in skipped_conv:
                logger.info(f"removing {layer}")
                skipped_conv.remove(layer)

def recipe_overrides_for_quality(conv_layers, skipped_conv):
    """
    Do not quantize out projection layers to avoid quantizing outputs of preceding concat layers.
    Quantizing output of concat layers can lead to quality degradation, due to sharing of scales
    across concat inputs, which can have varied ranges. Since this is a constraint enforced during
    model conversion, it may not be captured in layer-wise PSNR analysis of PyTorch model.
    """
    out_proj_layers = [layer for layer in conv_layers if "to_out" in layer]
    for layer in out_proj_layers:
        if layer not in skipped_conv:
            logger.info(f"adding {layer}")
            skipped_conv.add(layer)

def register_input_log_hook(unet, inputs):
    """
    Register forward pre hook to save model inputs 
    """
    def hook(_, input):
        input_copy = deepcopy(input)
        input_copy = tuple(i.to('cpu') for i in input_copy)
        inputs.append(input_copy)

        # Return inputs unmodified
        return input

    return unet.register_forward_pre_hook(hook)

def generate_calibration_data(pipe, args, calibration_dir):
    # Register forward pre hook to record unet inputs
    unet_inputs = []
    handle = register_input_log_hook(pipe.unet, unet_inputs)

    # If directory doesn't exist, create it
    os.makedirs(calibration_dir, exist_ok=True)

    # Run calibration prompts through the pipeline and
    # serialize recorded UNet model inputs
    for prompt in CALIBRATION_DATA:
        gen = torch.manual_seed(args.seed)
        # run forward pass
        pipe(prompt=prompt, generator=gen)
        # save unet inputs
        filename = "_".join(prompt.split(" ")) + "_" + str(args.seed) + ".pkl"
        filepath = os.path.join(calibration_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(unet_inputs, f)
        # clear
        unet_inputs.clear()

    handle.remove()

def register_input_preprocessing_hook(pipe):
    """
    Register forward pre hook to convert UNet inputs from HuggingFace StableDiffusionPipeline
    to match expected model inputs in UNet2DConditionModel defined in unet.py
    """
    def hook(_, args, kwargs):
        sample = args[0]
        timestep = args[1]
        if len(timestep.shape) == 0:
            timestep = timestep[None]
        timestep = timestep.expand(sample.shape[0])
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        encoder_hidden_states = encoder_hidden_states.permute((0, 2, 1)).unsqueeze(2)
        modified_args = (sample, timestep, encoder_hidden_states)
        return (modified_args, {})

    return pipe.unet.register_forward_pre_hook(hook, with_kwargs=True)

def prepare_pipe(pipe, unet):
    """
    Create a new pipeline from `pipe` with `unet` as the noise predictor
    """
    new_pipe = deepcopy(pipe)
    unet.to(new_pipe.unet.device)
    new_pipe.unet = unet
    pre_hook_handle = register_input_preprocessing_hook(new_pipe)
    return new_pipe, pre_hook_handle

def run_pipe(pipe):
    gen = torch.manual_seed(args.seed)
    kwargs = dict(
        prompt=RANDOM_TEST_DATA,
        output_type="latent",
        generator=gen,
    )
    return np.array([latent.cpu().numpy() for latent in pipe(**kwargs).images])


def get_reference_pipeline(model_version):
    # Initialize pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        model_version,
        use_safetensors=True,
        use_auth_token=True,
    )
    DEFAULT_NUM_INFERENCE_STEPS = 50
    pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)

    # Initialize reference unet
    unet_cls = unet.UNet2DConditionModel
    reference_unet = unet_cls(**pipe.unet.config).eval()
    reference_unet.load_state_dict(pipe.unet.state_dict())

    # Initialize reference pipeline
    ref_pipe, _ = prepare_pipe(pipe, reference_unet)

    del pipe
    gc.collect()
    return ref_pipe

def main(args):
    # Initialize reference pipeline
    ref_pipe = get_reference_pipeline(args.model_version)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.debug(f"Placing pipe in {device}")
    ref_pipe.to(device)
    # Generate baseline outputs
    ref_out = run_pipe(ref_pipe)

    # Setup artifact file paths
    os.makedirs(args.o, exist_ok=True)
    recipe_json_path = os.path.join(args.o, f"{args.model_version.replace('/', '_')}_quantization_recipe.json")
    calibration_dir = os.path.join(args.o, f"calibration_data_{args.model_version.replace('/', '_')}")

    # Generate calibration data 
    if args.generate_calibration_data:
        generate_calibration_data(ref_pipe, args, calibration_dir)

    # Compute layer-wise PSNR
    if args.layerwise_sensitivity:
        logger.info("Compute Layer-wise PSNR")
        quantizable_modules = get_quantizable_modules(ref_pipe.unet)

        results = {
            'conv': {},
            'einsum': {},
            'model_version': args.model_version
        }
        dataloader = unet_data_loader(calibration_dir, device, args.calibration_nsamples)

        for module_type, module_name in tqdm(quantizable_modules):
            logger.info(f"Quantizing UNet Layer: {module_name}")
            config = quantize_module_config(module_name)
            quantized_unet = quantize(ref_pipe.unet, config, dataloader)

            # Generate outputs from quantized model
            q_pipe, _ = prepare_pipe(ref_pipe, quantized_unet)
            test_out = run_pipe(q_pipe)

            psnr = [float(f"{compute_psnr(r, t):.1f}") for r, t in zip(ref_out, test_out)]
            logger.info(f"PSNR: {psnr}")
            avg_psnr = sum(psnr) / len(psnr)
            logger.info(f"AVG PSNR: {avg_psnr}")
            results[module_type][module_name] = avg_psnr

            del quantized_unet
            del q_pipe
            gc.collect()

        with open(recipe_json_path, 'w') as f:
            json.dump(results, f, indent=2)

    if args.quantize_pytorch:
        logger.info("Quantizing UNet PyTorch model")
        dataloader = unet_data_loader(calibration_dir, device, args.calibration_nsamples)

        with open(recipe_json_path, "r") as f:
            results = json.load(f)

        logger.info(f"Conv PSNR threshold: {args.conv_psnr}, Attn PSNR threshold: {args.attn_psnr}")
        skipped_conv = set([layer for layer, psnr in results['conv'].items() if psnr < args.conv_psnr])
        skipped_einsum = set([layer for layer, psnr in results['einsum'].items() if psnr < args.attn_psnr])

        # Apply some overrides on PSNR based recipe for inference and quality improvements
        # Users can disable these selectively based on specific targets
        recipe_overrides_for_inference_speedup(results['conv'].keys(), skipped_conv)
        recipe_overrides_for_quality(results['conv'].keys(), skipped_conv)

        config = quantize_cumulative_config(skipped_conv, skipped_einsum)
        quantized_unet = quantize(ref_pipe.unet, config, dataloader)

        # Generate outputs from quantized model
        q_pipe, handle = prepare_pipe(ref_pipe, quantized_unet)
        test_out = run_pipe(q_pipe)

        psnr = [float(f"{compute_psnr(r, t):.1f}") for r, t in zip(ref_out, test_out)]
        logger.info(f"PSNR: {psnr}")
        avg_psnr = sum(psnr) / len(psnr)
        logger.info(f"AVG PSNR: {avg_psnr}")

        handle.remove()
        quantized_unet.to('cpu')
        sample_unet_input = {
            "sample": dataloader[0][0].to('cpu'),
            "timestep": dataloader[0][1].to('cpu'),
            "encoder_hidden_states": dataloader[0][2].to('cpu'),
        }

        logger.info("JIT tracing quantized model")
        traced_model = torch.jit.trace(quantized_unet, example_inputs=list(sample_unet_input.values()))

        logger.info("Converting to CoreML")
        coreml_sample_unet_input = {
            k: v.numpy().astype(np.float16)
            for k, v in sample_unet_input.items()
        }
        coreml_model = convert_to_coreml(traced_model, coreml_sample_unet_input)
        coreml_filename = f"Stable_Diffusion_version_{args.model_version.replace('/', '_')}_unet.mlpackage"
        coreml_model.save(os.path.join(args.o, coreml_filename))

        del q_pipe

    del ref_pipe
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        required=True,
        help="Output directory to save calibration data and quantization artifacts"
    )
    parser.add_argument(
        "--model-version",
        required=True,
        choices=("runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1-base"),
        help=
        ("The pre-trained model checkpoint and configuration to restore"
    ))
    parser.add_argument(
        "--generate-calibration-data",
        action="store_true",
        help="Generate calibration data for UNet model"
    )
    parser.add_argument(
        "--layerwise-sensitivity",
        action="store_true",
        help="Compute compression sensitivity per-layer, by quantizing one layer at a time"
    )
    parser.add_argument(
        "--quantize-pytorch",
        action="store_true",
        help="Generate activation quantized UNet model by quantizing layers above specified PSNR threshold"
    )
    parser.add_argument(
        "--calibration-nsamples",
        type=int,
        help="Number of samples to use for calibrating UNet model"
    )
    parser.add_argument("--seed",
                        "-s",
                        default=50,
                        type=int,
                        help="Random seed to be able to reproduce results"
    )
    parser.add_argument("--conv-psnr",
                        default=40.0,
                        type=float,
                        help="PSNR threshold for convolutional layers (default for stabilityai/stable-diffusion-2-1-base)"
    )
    parser.add_argument("--attn-psnr",
                        default=30.0,
                        type=float,
                        help="PSNR threshold for attention (Einsum) layers (default for stabilityai/stable-diffusion-2-1-base)"
    )
    args = parser.parse_args()

    main(args)
