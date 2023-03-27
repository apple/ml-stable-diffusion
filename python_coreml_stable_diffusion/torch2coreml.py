#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from python_coreml_stable_diffusion import unet

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import coremltools as ct
from diffusers import StableDiffusionPipeline
import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
from python_coreml_stable_diffusion import chunk_mlprogram
import requests
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

from types import MethodType


def _get_coreml_inputs(sample_inputs, args):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]


def compute_psnr(a, b):
    """ Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr


ABSOLUTE_MIN_PSNR = 35


def report_correctness(original_outputs, final_outputs, log_prefix):
    """ Report PSNR values across two compatible tensors
    """
    original_psnr = compute_psnr(original_outputs, original_outputs)
    final_psnr = compute_psnr(original_outputs, final_outputs)

    dB_change = final_psnr - original_psnr
    logger.info(
        f"{log_prefix}: PSNR changed by {dB_change:.1f} dB ({original_psnr:.1f} -> {final_psnr:.1f})"
    )

    if final_psnr < ABSOLUTE_MIN_PSNR:
        raise ValueError(f"{final_psnr:.1f} dB is too low!")
    else:
        logger.info(
            f"{final_psnr:.1f} dB > {ABSOLUTE_MIN_PSNR} dB (minimum allowed) parity check passed"
        )
    return final_psnr


def _get_out_path(args, submodule_name):
    fname = f"Stable_Diffusion_version_{args.model_version}_{submodule_name}.mlpackage"
    fname = fname.replace("/", "_")
    return os.path.join(args.o, fname)


# https://github.com/apple/coremltools/issues/1680
def _save_mlpackage(model, output_path):
    # First recreate MLModel object using its in memory spec, then save
    ct.models.MLModel(model._spec,
                      weights_dir=model._weights_dir,
                      is_temp_package=True).save(output_path)


def _convert_to_coreml(submodule_name, torchscript_module, sample_inputs,
                       output_names, args):
    out_path = _get_out_path(args, submodule_name)

    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        logger.info(f"Loading model from {out_path}")

        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(
            out_path, compute_units=ct.ComputeUnit[args.compute_unit])
        logger.info(
            f"Loading {out_path} took {time.time() - start:.1f} seconds")

        coreml_model.compute_unit = ct.ComputeUnit[args.compute_unit]
    else:
        logger.info(f"Converting {submodule_name} to CoreML..")
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            inputs=_get_coreml_inputs(sample_inputs, args),
            outputs=[ct.TensorType(name=name) for name in output_names],
            compute_units=ct.ComputeUnit[args.compute_unit],
            # skip_model_load=True,
        )

        del torchscript_module
        gc.collect()

        coreml_model.save(out_path)
        logger.info(f"Saved {submodule_name} model to {out_path}")

    return coreml_model, out_path


def quantize_weights_to_8bits(args):
    for model_name in [
            "text_encoder", "vae_decoder", "vae_encoder", "unet", "unet_chunk1",
            "unet_chunk2", "safety_checker"
    ]:
        out_path = _get_out_path(args, model_name)
        if os.path.exists(out_path):
            logger.info(f"Quantizing {model_name}")
            mlmodel = ct.models.MLModel(out_path,
                                        compute_units=ct.ComputeUnit.CPU_ONLY)
            mlmodel = ct.compression_utils.affine_quantize_weights(
                mlmodel, mode="linear")
            mlmodel.save(out_path)
            logger.info("Done")
        else:
            logger.info(
                f"Skipped quantizing {model_name} (Not found at {out_path})")


def _compile_coreml_model(source_model_path, output_dir, final_name):
    """ Compiles Core ML models using the coremlcompiler utility from Xcode toolchain
    """
    target_path = os.path.join(output_dir, f"{final_name}.mlmodelc")
    if os.path.exists(target_path):
        logger.warning(
            f"Found existing compiled model at {target_path}! Skipping..")
        return target_path

    logger.info(f"Compiling {source_model_path}")
    source_model_name = os.path.basename(
        os.path.splitext(source_model_path)[0])

    os.system(f"xcrun coremlcompiler compile {source_model_path} {output_dir}")
    compiled_output = os.path.join(output_dir, f"{source_model_name}.mlmodelc")
    shutil.move(compiled_output, target_path)

    return target_path


def bundle_resources_for_swift_cli(args):
    """
    - Compiles Core ML models from mlpackage into mlmodelc format
    - Download tokenizer resources for the text encoder
    """
    resources_dir = os.path.join(args.o, "Resources")
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir, exist_ok=True)
        logger.info(f"Created {resources_dir} for Swift CLI assets")

    # Compile model using coremlcompiler (Significantly reduces the load time for unet)
    for source_name, target_name in [("text_encoder", "TextEncoder"),
                                     ("vae_decoder", "VAEDecoder"),
                                     ("vae_encoder", "VAEEncoder"),
                                     ("unet", "Unet"),
                                     ("unet_chunk1", "UnetChunk1"),
                                     ("unet_chunk2", "UnetChunk2"),
                                     ("safety_checker", "SafetyChecker")]:
        source_path = _get_out_path(args, source_name)
        if os.path.exists(source_path):
            target_path = _compile_coreml_model(source_path, resources_dir,
                                                target_name)
            logger.info(f"Compiled {source_path} to {target_path}")
        else:
            logger.warning(
                f"{source_path} not found, skipping compilation to {target_name}.mlmodelc"
            )

    # Fetch and save vocabulary JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer vocab.json")
    with open(os.path.join(resources_dir, "vocab.json"), "wb") as f:
        f.write(requests.get(args.text_encoder_vocabulary_url).content)
    logger.info("Done")

    # Fetch and save merged pairs JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer merges.txt")
    with open(os.path.join(resources_dir, "merges.txt"), "wb") as f:
        f.write(requests.get(args.text_encoder_merges_url).content)
    logger.info("Done")

    return resources_dir


def convert_text_encoder(pipe, args):
    """ Converts the text encoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "text_encoder")
    if os.path.exists(out_path):
        logger.info(
            f"`text_encoder` already exists at {out_path}, skipping conversion."
        )
        return

    # Create sample inputs for tracing, conversion and correctness verification
    text_encoder_sequence_length = pipe.tokenizer.model_max_length
    text_encoder_hidden_size = pipe.text_encoder.config.hidden_size

    sample_text_encoder_inputs = {
        "input_ids":
        torch.randint(
            pipe.text_encoder.config.vocab_size,
            (1, text_encoder_sequence_length),
            # https://github.com/apple/coremltools/issues/1423
            dtype=torch.float32,
        )
    }
    sample_text_encoder_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_text_encoder_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_text_encoder_inputs_spec}")

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = torch.ones((bsz, seq_len, seq_len), dtype=dtype) * -1e4
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

    class TextEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.text_encoder = pipe.text_encoder
            setattr(
                self.text_encoder.text_model, "_build_causal_attention_mask",
                MethodType(_build_causal_attention_mask,
                           self.text_encoder.text_model))

        def forward(self, input_ids):
            return self.text_encoder(input_ids, return_dict=False)

    reference_text_encoder = TextEncoder().eval()

    logger.info("JIT tracing text_encoder..")
    reference_text_encoder = torch.jit.trace(
        reference_text_encoder,
        (sample_text_encoder_inputs["input_ids"].to(torch.int32), ),
    )
    logger.info("Done.")

    coreml_text_encoder, out_path = _convert_to_coreml(
        "text_encoder", reference_text_encoder, sample_text_encoder_inputs,
        ["last_hidden_state", "pooled_outputs"], args)

    # Set model metadata
    coreml_text_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_text_encoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_text_encoder.version = args.model_version
    coreml_text_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_text_encoder.input_description[
        "input_ids"] = "The token ids that represent the input text"

    # Set the output descriptions
    coreml_text_encoder.output_description[
        "last_hidden_state"] = "The token embeddings as encoded by the Transformer model"
    coreml_text_encoder.output_description[
        "pooled_outputs"] = "The version of the `last_hidden_state` output after pooling"

    _save_mlpackage(coreml_text_encoder, out_path)

    logger.info(f"Saved text_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = pipe.text_encoder(
            sample_text_encoder_inputs["input_ids"].to(torch.int32),
            return_dict=False,
        )[1].numpy()

        coreml_out = list(
            coreml_text_encoder.predict(
                {k: v.numpy()
                 for k, v in sample_text_encoder_inputs.items()}).values())[0]
        report_correctness(
            baseline_out, coreml_out,
            "text_encoder baseline PyTorch to reference CoreML")

    del reference_text_encoder, coreml_text_encoder, pipe.text_encoder
    gc.collect()


def modify_coremltools_torch_frontend_badbmm():
    """
    Modifies coremltools torch frontend for baddbmm to be robust to the `beta` argument being of non-float dtype:
    e.g. https://github.com/huggingface/diffusers/blob/v0.8.1/src/diffusers/models/attention.py#L315
    """
    from coremltools.converters.mil import register_torch_op
    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
    if "baddbmm" in _TORCH_OPS_REGISTRY:
        del _TORCH_OPS_REGISTRY["baddbmm"]

    @register_torch_op
    def baddbmm(context, node):
        """
        baddbmm(Tensor input, Tensor batch1, Tensor batch2, Scalar beta=1, Scalar alpha=1)
        output = beta * input + alpha * batch1 * batch2
        Notice that batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, then input must be broadcastable with a (b×n×p) tensor
        and out will be a (b×n×p) tensor.
        """
        assert len(node.outputs) == 1
        inputs = _get_inputs(context, node, expected=5)
        bias, batch1, batch2, beta, alpha = inputs

        if beta.val != 1.0:
            # Apply scaling factor beta to the bias.
            if beta.val.dtype == np.int32:
                beta = mb.cast(x=beta, dtype="fp32")
                logger.warning(
                    f"Casted the `beta`(value={beta.val}) argument of `baddbmm` op "
                    "from int32 to float32 dtype for conversion!")
            bias = mb.mul(x=beta, y=bias, name=bias.name + "_scaled")

            context.add(bias)

        if alpha.val != 1.0:
            # Apply scaling factor alpha to the input.
            batch1 = mb.mul(x=alpha, y=batch1, name=batch1.name + "_scaled")
            context.add(batch1)

        bmm_node = mb.matmul(x=batch1, y=batch2, name=node.name + "_bmm")
        context.add(bmm_node)

        baddbmm_node = mb.add(x=bias, y=bmm_node, name=node.name)
        context.add(baddbmm_node)


def convert_vae_decoder(pipe, args):
    """ Converts the VAE Decoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "vae_decoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vae_decoder` already exists at {out_path}, skipping conversion."
        )
        return

    if not hasattr(pipe, "unet"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_vae_decoder() before convert_unet()")

    z_shape = (
        1,  # B
        pipe.vae.latent_channels,  # C
        args.latent_h or pipe.unet.config.sample_size,  # H
        args.latent_w or pipe.unet.config.sample_size,  # w
    )

    sample_vae_decoder_inputs = {
        "z": torch.rand(*z_shape, dtype=torch.float16)
    }

    class VAEDecoder(nn.Module):
        """ Wrapper nn.Module wrapper for pipe.decode() method
        """

        def __init__(self):
            super().__init__()
            self.post_quant_conv = pipe.vae.post_quant_conv
            self.decoder = pipe.vae.decoder

        def forward(self, z):
            return self.decoder(self.post_quant_conv(z))

    baseline_decoder = VAEDecoder().eval()

    # No optimization needed for the VAE Decoder as it is a pure ConvNet
    traced_vae_decoder = torch.jit.trace(
        baseline_decoder, (sample_vae_decoder_inputs["z"].to(torch.float32), ))

    modify_coremltools_torch_frontend_badbmm()
    coreml_vae_decoder, out_path = _convert_to_coreml(
        "vae_decoder", traced_vae_decoder, sample_vae_decoder_inputs,
        ["image"], args)

    # Set model metadata
    coreml_vae_decoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_vae_decoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_vae_decoder.version = args.model_version
    coreml_vae_decoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_vae_decoder.input_description["z"] = \
        "The denoised latent embeddings from the unet model after the last step of reverse diffusion"

    # Set the output descriptions
    coreml_vae_decoder.output_description[
        "image"] = "Generated image normalized to range [-1, 1]"

    _save_mlpackage(coreml_vae_decoder, out_path)

    logger.info(f"Saved vae_decoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_decoder(
            z=sample_vae_decoder_inputs["z"].to(torch.float32)).numpy()
        coreml_out = list(
            coreml_vae_decoder.predict(
                {k: v.numpy()
                 for k, v in sample_vae_decoder_inputs.items()}).values())[0]
        report_correctness(baseline_out, coreml_out,
                           "vae_decoder baseline PyTorch to baseline CoreML")

    del traced_vae_decoder, pipe.vae.decoder, coreml_vae_decoder
    gc.collect()


def convert_vae_encoder(pipe, args):
    """ Converts the VAE Encoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "vae_encoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vae_encoder` already exists at {out_path}, skipping conversion."
        )
        return

    if not hasattr(pipe, "unet"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_vae_encoder() before convert_unet()")
    
    height = (args.latent_h or pipe.unet.config.sample_size) * 8
    width = (args.latent_w or pipe.unet.config.sample_size) * 8
    
    z_shape = (
        1,  # B
        3,  # C (RGB range from -1 to 1)
        height,  # H
        width,  # w
    )

    sample_vae_encoder_inputs = {
        "z": torch.rand(*z_shape, dtype=torch.float16)
    }

    class VAEEncoder(nn.Module):
        """ Wrapper nn.Module wrapper for pipe.encode() method
        """

        def __init__(self):
            super().__init__()
            self.quant_conv = pipe.vae.quant_conv
            self.encoder = pipe.vae.encoder

        def forward(self, z):
            return self.quant_conv(self.encoder(z))

    baseline_encoder = VAEEncoder().eval()

    # No optimization needed for the VAE Encoder as it is a pure ConvNet
    traced_vae_encoder = torch.jit.trace(
        baseline_encoder, (sample_vae_encoder_inputs["z"].to(torch.float32), ))

    modify_coremltools_torch_frontend_badbmm()
    coreml_vae_encoder, out_path = _convert_to_coreml(
        "vae_encoder", traced_vae_encoder, sample_vae_encoder_inputs,
        ["latent"], args)

    # Set model metadata
    coreml_vae_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_vae_encoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_vae_encoder.version = args.model_version
    coreml_vae_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_vae_encoder.input_description["z"] = \
        "The input image to base the initial latents on normalized to range [-1, 1]"

    # Set the output descriptions
    coreml_vae_encoder.output_description["latent"] = "The latent embeddings from the unet model from the input image."

    _save_mlpackage(coreml_vae_encoder, out_path)

    logger.info(f"Saved vae_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_encoder(
            z=sample_vae_encoder_inputs["z"].to(torch.float32)).numpy()
        coreml_out = list(
            coreml_vae_encoder.predict(
                {k: v.numpy()
                 for k, v in sample_vae_encoder_inputs.items()}).values())[0]
        report_correctness(baseline_out, coreml_out,
                           "vae_encoder baseline PyTorch to baseline CoreML")

    del traced_vae_encoder, pipe.vae.encoder, coreml_vae_encoder
    gc.collect()


def convert_unet(pipe, args):
    """ Converts the UNet component of Stable Diffusion
    """
    out_path = _get_out_path(args, "unet")

    # Check if Unet was previously exported and then chunked
    unet_chunks_exist = all(
        os.path.exists(
            out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage"))
        for idx in range(2))

    if args.chunk_unet and unet_chunks_exist:
        logger.info("`unet` chunks already exist, skipping conversion.")
        del pipe.unet
        gc.collect()
        return

    # If original Unet does not exist, export it from PyTorch+diffusers
    elif not os.path.exists(out_path):
        # Register the selected attention implementation globally
        unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet.AttentionImplementations[
            args.attention_implementation]
        logger.info(
            f"Attention implementation in effect: {unet.ATTENTION_IMPLEMENTATION_IN_EFFECT}"
        )

        # Prepare sample input shapes and values
        batch_size = 2  # for classifier-free guidance
        sample_shape = (
            batch_size,                    # B
            pipe.unet.config.in_channels,  # C
            args.latent_h or pipe.unet.config.sample_size,  # H
            args.latent_w or pipe.unet.config.sample_size,  # W
        )

        if not hasattr(pipe, "text_encoder"):
            raise RuntimeError(
                "convert_text_encoder() deletes pipe.text_encoder to save RAM. "
                "Please use convert_unet() before convert_text_encoder()")

        encoder_hidden_states_shape = (
            batch_size,
            pipe.text_encoder.config.hidden_size,
            1,
            pipe.text_encoder.config.max_position_embeddings,
        )

        # Create the scheduled timesteps for downstream use
        DEFAULT_NUM_INFERENCE_STEPS = 50
        pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)

        sample_unet_inputs = OrderedDict([
            ("sample", torch.rand(*sample_shape)),
            ("timestep",
             torch.tensor([pipe.scheduler.timesteps[0].item()] *
                          (batch_size)).to(torch.float32)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape))
        ])
        sample_unet_inputs_spec = {
            k: (v.shape, v.dtype)
            for k, v in sample_unet_inputs.items()
        }
        logger.info(f"Sample inputs spec: {sample_unet_inputs_spec}")

        # Initialize reference unet
        reference_unet = unet.UNet2DConditionModel(**pipe.unet.config).eval()
        load_state_dict_summary = reference_unet.load_state_dict(
            pipe.unet.state_dict())

        # Prepare inputs
        baseline_sample_unet_inputs = deepcopy(sample_unet_inputs)
        baseline_sample_unet_inputs[
            "encoder_hidden_states"] = baseline_sample_unet_inputs[
                "encoder_hidden_states"].squeeze(2).transpose(1, 2)

        # JIT trace
        logger.info("JIT tracing..")
        reference_unet = torch.jit.trace(reference_unet,
                                         list(sample_unet_inputs.values()))
        logger.info("Done.")

        if args.check_output_correctness:
            baseline_out = pipe.unet(**baseline_sample_unet_inputs,
                                     return_dict=False)[0].numpy()
            reference_out = reference_unet(**sample_unet_inputs)[0].numpy()
            report_correctness(baseline_out, reference_out,
                               "unet baseline to reference PyTorch")

        del pipe.unet
        gc.collect()

        coreml_sample_unet_inputs = {
            k: v.numpy().astype(np.float16)
            for k, v in sample_unet_inputs.items()
        }

        coreml_unet, out_path = _convert_to_coreml("unet", reference_unet,
                                                   coreml_sample_unet_inputs,
                                                   ["noise_pred"], args)
        del reference_unet
        gc.collect()

        # Set model metadata
        coreml_unet.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
        coreml_unet.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
        coreml_unet.version = args.model_version
        coreml_unet.short_description = \
            "Stable Diffusion generates images conditioned on text or other images as input through the diffusion process. " \
            "Please refer to https://arxiv.org/abs/2112.10752 for details."

        # Set the input descriptions
        coreml_unet.input_description["sample"] = \
            "The low resolution latent feature maps being denoised through reverse diffusion"
        coreml_unet.input_description["timestep"] = \
            "A value emitted by the associated scheduler object to condition the model on a given noise schedule"
        coreml_unet.input_description["encoder_hidden_states"] = \
            "Output embeddings from the associated text_encoder model to condition to generated image on text. " \
            "A maximum of 77 tokens (~40 words) are allowed. Longer text is truncated. " \
            "Shorter text does not reduce computation."

        # Set the output descriptions
        coreml_unet.output_description["noise_pred"] = \
            "Same shape and dtype as the `sample` input. " \
            "The predicted noise to facilitate the reverse diffusion (denoising) process"

        _save_mlpackage(coreml_unet, out_path)
        logger.info(f"Saved unet into {out_path}")

        # Parity check PyTorch vs CoreML
        if args.check_output_correctness:
            coreml_out = list(
                coreml_unet.predict(coreml_sample_unet_inputs).values())[0]
            report_correctness(baseline_out, coreml_out,
                               "unet baseline PyTorch to reference CoreML")

        del coreml_unet
        gc.collect()
    else:
        del pipe.unet
        gc.collect()
        logger.info(
            f"`unet` already exists at {out_path}, skipping conversion.")

    if args.chunk_unet and not unet_chunks_exist:
        logger.info("Chunking unet in two approximately equal MLModels")
        args.mlpackage_path = out_path
        args.remove_original = False
        chunk_mlprogram.main(args)


def convert_safety_checker(pipe, args):
    """ Converts the Safety Checker component of Stable Diffusion
    """
    if pipe.safety_checker is None:
        logger.warning(
            f"diffusers pipeline for {args.model_version} does not have a `safety_checker` module! " \
            "`--convert-safety-checker` will be ignored."
        )
        return

    out_path = _get_out_path(args, "safety_checker")
    if os.path.exists(out_path):
        logger.info(
            f"`safety_checker` already exists at {out_path}, skipping conversion."
        )
        return

    sample_image = np.random.randn(
        1,  # B
        args.latent_h or pipe.vae.config.sample_size,  # H
        args.latent_w or pipe.vae.config.sample_size,  # w
        3  # C
    ).astype(np.float32)

    # Note that pipe.feature_extractor is not an ML model. It simply
    # preprocesses data for the pipe.safety_checker module.
    safety_checker_input = pipe.feature_extractor(
        pipe.numpy_to_pil(sample_image),
        return_tensors="pt",
    ).pixel_values.to(torch.float32)

    sample_safety_checker_inputs = OrderedDict([
        ("clip_input", safety_checker_input),
        ("images", torch.from_numpy(sample_image)),
        ("adjustment", torch.tensor([0]).to(torch.float32)),
    ])

    sample_safety_checker_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_safety_checker_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_safety_checker_inputs_spec}")

    # Patch safety_checker's forward pass to be vectorized and avoid conditional blocks
    # (similar to pipe.safety_checker.forward_onnx)
    from diffusers.pipelines.stable_diffusion import safety_checker

    def forward_coreml(self, clip_input, images, adjustment):
        """ Forward pass implementation for safety_checker
        """

        def cosine_distance(image_embeds, text_embeds):
            return F.normalize(image_embeds) @ F.normalize(
                text_embeds).transpose(0, 1)

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds,
                                           self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = special_scores.gt(0).float().sum(dim=1).gt(0).float()
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(
            -1, cos_dist.shape[1])

        concept_scores = (cos_dist -
                          self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = concept_scores.gt(0).float().sum(dim=1).gt(0)[:,
                                                                          None,
                                                                          None,
                                                                          None]

        has_nsfw_concepts_inds, _ = torch.broadcast_tensors(
            has_nsfw_concepts, images)
        images[has_nsfw_concepts_inds] = 0.0  # black image

        return images, has_nsfw_concepts.float(), concept_scores

    baseline_safety_checker = deepcopy(pipe.safety_checker.eval())
    setattr(baseline_safety_checker, "forward",
            MethodType(forward_coreml, baseline_safety_checker))

    # In order to parity check the actual signal, we need to override the forward pass to return `concept_scores` which is the
    # output before thresholding
    # Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L100
    def forward_extended_return(self, clip_input, images, adjustment):

        def cosine_distance(image_embeds, text_embeds):
            normalized_image_embeds = F.normalize(image_embeds)
            normalized_text_embeds = F.normalize(text_embeds)
            return torch.mm(normalized_image_embeds,
                            normalized_text_embeds.t())

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds,
                                           self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(
            -1, cos_dist.shape[1])

        concept_scores = (cos_dist -
                          self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        images[has_nsfw_concepts] = 0.0

        return images, has_nsfw_concepts, concept_scores

    setattr(pipe.safety_checker, "forward",
            MethodType(forward_extended_return, pipe.safety_checker))

    # Trace the safety_checker model
    logger.info("JIT tracing..")
    traced_safety_checker = torch.jit.trace(
        baseline_safety_checker, list(sample_safety_checker_inputs.values()))
    logger.info("Done.")
    del baseline_safety_checker
    gc.collect()

    # Cast all inputs to float16
    coreml_sample_safety_checker_inputs = {
        k: v.numpy().astype(np.float16)
        for k, v in sample_safety_checker_inputs.items()
    }

    # Convert safety_checker model to Core ML
    coreml_safety_checker, out_path = _convert_to_coreml(
        "safety_checker", traced_safety_checker,
        coreml_sample_safety_checker_inputs,
        ["filtered_images", "has_nsfw_concepts", "concept_scores"], args)

    # Set model metadata
    coreml_safety_checker.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_safety_checker.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_safety_checker.version = args.model_version
    coreml_safety_checker.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_safety_checker.input_description["clip_input"] = \
        "The normalized image input tensor resized to (224x224) in channels-first (BCHW) format"
    coreml_safety_checker.input_description["images"] = \
        f"Output of the vae_decoder ({pipe.vae.config.sample_size}x{pipe.vae.config.sample_size}) in channels-last (BHWC) format"
    coreml_safety_checker.input_description["adjustment"] = \
        "Bias added to the concept scores to trade off increased recall for reduce precision in the safety checker classifier"

    # Set the output descriptions
    coreml_safety_checker.output_description["filtered_images"] = \
        f"Identical to the input `images`. If safety checker detected any sensitive content, " \
        "the corresponding image is replaced with a blank image (zeros)"
    coreml_safety_checker.output_description["has_nsfw_concepts"] = \
        "Indicates whether the safety checker model found any sensitive content in the given image"
    coreml_safety_checker.output_description["concept_scores"] = \
        "Concept scores are the scores before thresholding at zero yields the `has_nsfw_concepts` output. " \
        "These scores can be used to tune the `adjustment` input"

    _save_mlpackage(coreml_safety_checker, out_path)

    if args.check_output_correctness:
        baseline_out = pipe.safety_checker(
            **sample_safety_checker_inputs)[2].numpy()
        coreml_out = coreml_safety_checker.predict(
            coreml_sample_safety_checker_inputs)["concept_scores"]
        report_correctness(
            baseline_out, coreml_out,
            "safety_checker baseline PyTorch to reference CoreML")

    del traced_safety_checker, coreml_safety_checker, pipe.safety_checker
    gc.collect()


def main(args):
    os.makedirs(args.o, exist_ok=True)

    # Instantiate diffusers pipe as reference
    logger.info(
        f"Initializing StableDiffusionPipeline with {args.model_version}..")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_version,
                                                   use_auth_token=True)
    logger.info("Done.")

    # Convert models
    if args.convert_vae_decoder:
        logger.info("Converting vae_decoder")
        convert_vae_decoder(pipe, args)
        logger.info("Converted vae_decoder")

    if args.convert_vae_encoder:
        logger.info("Converting vae_encoder")
        convert_vae_encoder(pipe, args)
        logger.info("Converted vae_encoder")

    if args.convert_unet:
        logger.info("Converting unet")
        convert_unet(pipe, args)
        logger.info("Converted unet")

    if args.convert_text_encoder:
        logger.info("Converting text_encoder")
        convert_text_encoder(pipe, args)
        logger.info("Converted text_encoder")

    if args.convert_safety_checker:
        logger.info("Converting safety_checker")
        convert_safety_checker(pipe, args)
        logger.info("Converted safety_checker")

    if args.bundle_resources_for_swift_cli:
        logger.info("Bundling resources for the Swift CLI")
        bundle_resources_for_swift_cli(args)
        logger.info("Bundled resources for the Swift CLI")

    if args.quantize_weights_to_8bits:
        # Note: Not recommended, significantly degrades generated image quality
        logger.info("Quantizing weights to 8-bit precision")
        quantize_weights_to_8bits(args)
        logger.info("Quantized weights to 8-bit precision")


def parser_spec():
    parser = argparse.ArgumentParser()

    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument("--convert-text-encoder", action="store_true")
    parser.add_argument("--convert-vae-decoder", action="store_true")
    parser.add_argument("--convert-vae-encoder", action="store_true")
    parser.add_argument("--convert-unet", action="store_true")
    parser.add_argument("--convert-safety-checker", action="store_true")
    parser.add_argument(
        "--model-version",
        default="CompVis/stable-diffusion-v1-4",
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=stable-diffusion"
         ))
    parser.add_argument("--compute-unit",
                        choices=tuple(cu
                                      for cu in ct.ComputeUnit._member_names_),
                        default="ALL")

    parser.add_argument(
        "--latent-h",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of rows) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--latent-w",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of cols) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=tuple(ai
                      for ai in unet.AttentionImplementations._member_names_),
        default=unet.ATTENTION_IMPLEMENTATION_IN_EFFECT.name,
        help=
        "The enumerated implementations trade off between ANE and GPU performance",
    )
    parser.add_argument(
        "-o",
        default=os.getcwd(),
        help="The resulting mlpackages will be saved into this directory")
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        ("If specified, compares the outputs of original PyTorch and final CoreML models and reports PSNR in dB. ",
         "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
         ))
    parser.add_argument(
        "--chunk-unet",
        action="store_true",
        help=
        ("If specified, generates two mlpackages out of the unet model which approximately equal weights sizes. "
         "This is required for ANE deployment on iOS and iPadOS. Not required for macOS."
         ))
    parser.add_argument(
        "--quantize-weights-to-8bits",
        action="store_true",
        help=
        ("If specified, quantize 16-bits weights to 8-bits weights in-place for all models. "
         "Not recommended as the generated image quality degraded significantly after 8-bit weight quantization"
         ))

    # Swift CLI Resource Bundling
    parser.add_argument(
        "--bundle-resources-for-swift-cli",
        action="store_true",
        help=
        ("If specified, creates a resources directory compatible with the sample Swift CLI. "
         "It compiles all four models and adds them to a StableDiffusionResources directory "
         "along with a `vocab.json` and `merges.txt` for the text tokenizer"))
    parser.add_argument(
        "--text-encoder-vocabulary-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
        help="The URL to the vocabulary file use by the text tokenizer")
    parser.add_argument(
        "--text-encoder-merges-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
        help="The URL to the merged pairs used in by the text tokenizer.")

    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()

    main(args)
