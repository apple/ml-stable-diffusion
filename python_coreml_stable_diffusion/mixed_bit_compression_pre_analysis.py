from collections import OrderedDict
from copy import deepcopy
from functools import partial
import argparse
import gc
import json

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel('INFO')

import numpy as np
import os
from PIL import Image
from python_coreml_stable_diffusion.torch2coreml import compute_psnr, get_pipeline
import time

import torch
import torch.nn as nn
import requests
torch.set_grad_enabled(False)

from tqdm import tqdm

# Bit-widths the Neural Engine is capable of accelerating
NBITS = [1, 2, 4, 6, 8]

# Minimum number of elements in a weight tensor to be considered for palettization
# (saves pre-analysis time)
PALETTIZE_MIN_SIZE = 1e5

# Signal integrity is computed based on these 4 random prompts
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

TEST_RESOLUTION = 768

RANDOM_TEST_IMAGE_DATA = [
    Image.open(
        requests.get(path, stream=True).raw).convert("RGB").resize(
            (TEST_RESOLUTION, TEST_RESOLUTION), Image.LANCZOS
    ) for path in [
        "http://farm1.staticflickr.com/106/298138827_19bb723252_z.jpg",
        "http://farm4.staticflickr.com/3772/9666116202_648cd752d6_z.jpg",
        "http://farm3.staticflickr.com/2238/2472574092_f5534bb2f7_z.jpg",
        "http://farm1.staticflickr.com/220/475442674_47d81fdc2c_z.jpg",
        "http://farm8.staticflickr.com/7231/7359341784_4c5358197f_z.jpg",
        "http://farm8.staticflickr.com/7283/8737653089_d0c77b8597_z.jpg",
        "http://farm3.staticflickr.com/2454/3989339438_2f32b76ebb_z.jpg",
        "http://farm1.staticflickr.com/34/123005230_13051344b1_z.jpg",
]]


# Copied from https://github.com/apple/coremltools/blob/7.0b1/coremltools/optimize/coreml/_quantization_passes.py#L602
from coremltools.converters.mil.mil import types
def fake_linear_quantize(val, axis=-1, mode='LINEAR', dtype=types.int8):
    from coremltools.optimize.coreml._quantization_passes import AffineQuantParams
    from coremltools.converters.mil.mil.types.type_mapping import nptype_from_builtin

    val_dtype = val.dtype
    def _ensure_numerical_range_and_cast(val, low, high, np_dtype):
        '''
        For some cases, the computed quantized data might exceed the data range.
        For instance, after rounding and addition, we might get `128` for the int8 quantization.
        This utility function ensures the val in the data range before doing the cast.
        '''
        val = np.minimum(val, high)
        val = np.maximum(val, low)
        return val.astype(np_dtype)

    mode_dtype_to_range = {
        (types.int8, "LINEAR"): (-128, 127),
        (types.int8, "LINEAR_SYMMETRIC"): (-127, 127),
        (types.uint8, "LINEAR"): (0, 255),
        (types.uint8, "LINEAR_SYMMETRIC"): (0, 254),
    }

    if not isinstance(val, (np.ndarray, np.generic)):
        raise ValueError("Only numpy arrays are supported")

    params = AffineQuantParams()
    axes = tuple([i for i in range(len(val.shape)) if i != axis])
    val_min = np.amin(val, axis=axes, keepdims=True)
    val_max = np.amax(val, axis=axes, keepdims=True)

    if mode == "LINEAR_SYMMETRIC":
        # For the linear_symmetric mode, the range is symmetrical to 0
        max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
        val_min = -max_abs
        val_max = max_abs
    else:
        assert mode == "LINEAR"
        # For the linear mode, we need to make sure the data range contains `0`
        val_min = np.minimum(0.0, val_min)
        val_max = np.maximum(0.0, val_max)

    q_val_min, q_val_max = mode_dtype_to_range[(dtype, mode)]

    # Set the zero point to symmetric mode
    np_dtype = nptype_from_builtin(dtype)
    if mode == "LINEAR_SYMMETRIC":
        if dtype == types.int8:
            params.zero_point = (0 * np.ones(val_min.shape)).astype(np.int8)
        else:
            assert dtype == types.uint8
            params.zero_point = (127 * np.ones(val_min.shape)).astype(np.uint8)
    else:
        assert mode == "LINEAR"
        params.zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min)
        params.zero_point = np.round(params.zero_point)
        params.zero_point = _ensure_numerical_range_and_cast(params.zero_point, q_val_min, q_val_max, np_dtype)

    # compute the params
    params.scale = (val_max - val_min) / (q_val_max - q_val_min)
    params.scale = params.scale.astype(val.dtype).squeeze()

    params.quantized_data = np.round(
        val * (q_val_max - q_val_min) / (val_max - val_min)
    )
    params.quantized_data = (params.quantized_data + params.zero_point)
    params.quantized_data = _ensure_numerical_range_and_cast(params.quantized_data, q_val_min, q_val_max, np_dtype)

    params.zero_point = params.zero_point.squeeze()
    params.axis = axis

    return (params.quantized_data.astype(val_dtype) - params.zero_point.astype(val_dtype)) * params.scale


# Copied from https://github.com/apple/coremltools/blob/7.0b1/coremltools/optimize/coreml/_quantization_passes.py#L423
def fake_palettize(module, nbits, in_ngroups=1, out_ngroups=1):
    """ Simulate weight palettization
    """
    from coremltools.models.neural_network.quantization_utils import _get_kmeans_lookup_table_and_weight

    def compress_kmeans(val, nbits):
        lut, indices = _get_kmeans_lookup_table_and_weight(nbits, val)
        lut = lut.astype(val.dtype)
        indices = indices.astype(np.uint8)
        return lut, indices

    dtype = module.weight.data.dtype
    device = module.weight.data.device
    val = module.weight.data.cpu().numpy().astype(np.float16)
    
    if out_ngroups == 1 and in_ngroups == 1:
        lut, indices = compress_kmeans(val=val, nbits=nbits)
        module.weight.data = torch.from_numpy(lut[indices]).reshape(val.shape).to(dtype)

    elif out_ngroups > 1 and in_ngroups == 1:
        assert val.shape[0] % out_ngroups == 0
        rvals = [
            compress_kmeans(val=chunked_val, nbits=nbits)
            for chunked_val in np.split(val, out_ngroups, axis=0)
        ]
        shape = list(val.shape)
        shape[0] = shape[0] // out_ngroups
        module.weight.data = torch.cat([
            torch.from_numpy(lut[indices]).reshape(shape)
            for lut,indices in rvals
        ], dim=0).to(dtype).to(device)
 
    elif in_ngroups > 1 and out_ngroups == 1:
        assert val.shape[1] % in_ngroups == 0
        rvals = [
            compress_kmeans(val=chunked_val, nbits=nbits)
            for chunked_val in np.split(val, in_ngroups, axis=1)
        ]
        shape = list(val.shape)
        shape[1] = shape[1] // in_ngroups
        module.weight.data = torch.cat([
            torch.from_numpy(lut[indices]).reshape(shape)
            for lut,indices in rvals
        ], dim=1).to(dtype).to(device)
    else:
        raise ValueError(f"in_ngroups={in_ngroups} & out_ngroups={out_ngroups} is illegal!!!")
    
    return torch.from_numpy(val).to(dtype)


def restore_weight(module, value):
    device = module.weight.data.device
    module.weight.data = value.to(device)


def get_palettizable_modules(unet, min_size=PALETTIZE_MIN_SIZE):
    ret = [
        (name, getattr(module, 'weight').data.numel()) for name, module in unet.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
        if hasattr(module, 'weight') and getattr(module, 'weight').data.numel() > min_size
    ]
    candidates, sizes = [[a for a,b in ret], [b for a,b in ret]]
    logger.info(f"{len(candidates)} candidate tensors with {sum(sizes)/1e6} M total params")
    return candidates, sizes


def fake_int8_quantize(module):
    i = 0
    for name, submodule in tqdm(module.named_modules()):
        if hasattr(submodule, 'weight'):
            i+=1
            submodule.weight.data = torch.from_numpy(
                fake_linear_quantize(submodule.weight.data.numpy()))
    logger.info(f"{i} modules fake int8 quantized")
    return module


def fake_nbits_palette(module, nbits):
    i = 0
    for name, submodule in tqdm(module.named_modules()):
        if hasattr(submodule, 'weight'):
            i+=1
            fake_palettize(submodule, nbits=nbits)
    logger.info(f"{i} modules fake {nbits}-bits palettized")
    return module


def fake_palette_from_recipe(module, recipe):
    tot_bits = 0
    tot_numel = 0
    for name, submodule in tqdm(module.named_modules()):
        if hasattr(submodule, 'weight'):
            tot_numel += submodule.weight.numel()
            if name in recipe:
                nbits = recipe[name]
                assert nbits in NBITS + [16]
                tot_bits += submodule.weight.numel() * nbits
                if nbits == 16:
                    continue
                fake_palettize(submodule, nbits=nbits)
            else:
                tot_bits += submodule.weight.numel() * 16

    logger.info(f"Palettized to {tot_bits/tot_numel:.2f}-bits mixed palette ({tot_bits/8e6} MB) ")

# Globally synced RNG state
rng = torch.Generator()
rng_state = rng.get_state()

def run_pipe(pipe):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.debug(f"Placing pipe in {device}")

    global rng, rng_state
    rng.set_state(rng_state)
    kwargs = dict(
        prompt=RANDOM_TEST_DATA,
        negative_prompt=[""] * len(RANDOM_TEST_DATA),
        num_inference_steps=1,
        height=TEST_RESOLUTION,
        width=TEST_RESOLUTION,
        output_type="latent",
        generator=rng
    )
    if "Img2Img" in pipe.__class__.__name__:
        kwargs["image"] = RANDOM_TEST_IMAGE_DATA
        kwargs.pop("height")
        kwargs.pop("width")

        # Run a single denoising step
        kwargs["num_inference_steps"] = 4
        kwargs["strength"] = 0.25

    return np.array([latent.cpu().numpy() for latent in pipe.to(device)(**kwargs).images])


def benchmark_signal_integrity(pipe,
                               candidates,
                               nbits,
                               cumulative,
                               in_ngroups=1,
                               out_ngroups=1,
                               ref_out=None,
                               ):
    results = {}
    results['metadata'] = {
        'nbits': nbits,
        'out_ngroups': out_ngroups,
        'in_ngroups': in_ngroups,
        'cumulative': cumulative,
    }

    # If reference outputs are not provided, treat current pipe as reference
    if ref_out is None:
        ref_out = run_pipe(pipe)

    for candidate in tqdm(candidates):
        palettized = False
        for name, module in pipe.unet.named_modules():
            if name == candidate:
                orig_weight = fake_palettize(
                    module,
                    nbits,
                    out_ngroups=out_ngroups,
                    in_ngroups=in_ngroups,
                )
                palettized = True
                break
        if not palettized:
            raise KeyError(name)

        test_out = run_pipe(pipe)

        if not cumulative:
            restore_weight(module, orig_weight)

        results[candidate] = [
            float(f"{compute_psnr(r,t):.1f}")
            for r,t in zip(ref_out, test_out)
        ]
        logger.info(f"{nbits}-bit: {candidate} = {results[candidate]}")

    return results
            

def descending_psnr_order(results):
    if 'metadata' in results:
        results.pop('metadata')

    return OrderedDict(sorted(results.items(), key=lambda items: -sum(items[1])))


def simulate_quant_fn(ref_pipe, quantization_to_simulate):
    simulated_pipe = deepcopy(ref_pipe.to('cpu'))
    quantization_to_simulate(simulated_pipe.unet)
    simulated_out = run_pipe(simulated_pipe)
    del simulated_pipe
    gc.collect()

    ref_out = run_pipe(ref_pipe)
    simulated_psnr = sum([
        float(f"{compute_psnr(r, t):.1f}")
        for r, t in zip(ref_out, simulated_out)
    ]) / len(ref_out)

    return simulated_out, simulated_psnr


def build_recipe(results, sizes, psnr_threshold, default_nbits):
    stats = {'nbits': 0}
    recipe = {}

    for key in results[str(NBITS[0])]:
        if key == 'metadata':
            continue

        achieved_nbits = default_nbits
        for nbits in NBITS:
            avg_psnr = sum(results[str(nbits)][key])/len(RANDOM_TEST_DATA)
            if avg_psnr > psnr_threshold:
                achieved_nbits = nbits
                break
        recipe[key] = achieved_nbits
        stats['nbits'] += achieved_nbits * sizes[key]

    stats['size_mb'] = stats['nbits'] / (8*1e6)
    tot_size = sum(list(sizes.values()))
    stats['nbits'] /= tot_size

    return recipe, stats


def plot(results, args):
    import matplotlib.pyplot as plt
    max_model_size = sum(results['cumulative'][str(NBITS[0])]['metadata']['sizes'])
    f, ax = plt.subplots(1, 1, figsize=(7, 5))

    def compute_x_axis(sizes, nbits, default_nbits):
        max_compression_percent = (default_nbits - nbits) / default_nbits
        progress = np.cumsum(sizes)
        normalized_progress = progress / progress.max()

        return normalized_progress * max_compression_percent * 100

    # Linear 8-bit baseline and the intercept points for mixed-bit recipes
    linear8bit_baseline = results['baselines']['linear_8bit']

    # Mark the linear 8-bit baseline
    ax.plot(
        8 / args.default_nbits * 100,
        linear8bit_baseline,
        'bx',
        markersize=8,
        label="8-bit (linear quant)")

    # Plot the iso-dB line that matches the 8-bit baseline
    ax.plot([0,100], [linear8bit_baseline]*2, '--b')

    # Plot non-mixed-bit palettization curves
    for idx, nbits in enumerate(NBITS):
        size_keys = compute_x_axis(results['cumulative'][str(nbits)]['metadata']['sizes'], nbits, args.default_nbits)
        psnr = [
            sum(v) / len(RANDOM_TEST_DATA) # avg psnr
            for k,v in results['cumulative'][str(nbits)].items() if k != 'metadata'
        ]
        ax.plot(
            size_keys,
            psnr,
            label=f"{nbits}-bit")


    # Plot mixed-bit results
    mixed_palettes = [
        (float(spec.rsplit('_')[1]), psnr)
        for spec,psnr in results['baselines'].items()
        if 'recipe' in spec
    ]
    mixedbit_sizes = [100. * (1. - a[0] / args.default_nbits) for a in mixed_palettes]
    mixedbit_psnrs = [a[1] for a in mixed_palettes]
    ax.plot(
        mixedbit_sizes,
        mixedbit_psnrs,
        label="mixed-bit",
    )

    ax.set_xlabel("Model Size Reduction (%)")
    ax.set_ylabel("Signal Integrity (PSNR in dB)")
    ax.set_title(args.model_version)
    ax.legend()

    f.savefig(os.path.join(args.o, f"{args.model_version.replace('/','_')}_psnr_vs_size.png"))

def main(args):

    # Initialize pipe
    pipe = get_pipeline(args)

    # Preserve a pristine copy for reference outputs
    ref_pipe = deepcopy(pipe)
    if args.default_nbits != 16:
        logger.info(f"Palettizing unet to default {args.default_nbits}-bit")
        fake_nbits_palette(pipe.unet, args.default_nbits)
        logger.info("Done.")

    # Cache reference outputs
    ref_out = run_pipe(pipe)

    # Bookkeeping
    os.makedirs(args.o, exist_ok=True)

    results = {
        'single_layer': {},
        'cumulative': {},
        'model_version': args.model_version,
    }
    json_name = f"{args.model_version.replace('/','-')}_palettization_recipe.json"
    candidates, sizes = get_palettizable_modules(pipe.unet)

    sizes_table = dict(zip(candidates, sizes))

    if os.path.isfile(os.path.join(args.o, json_name)):
        with open(os.path.join(args.o, json_name), "r") as f:
            results = json.load(f)

    # Analyze uniform-precision palettization impact on signal integrity
    for nbits in NBITS:
        if str(nbits) not in results['single_layer']:
            # Measure the impact of palettization of each layer independently
            results['single_layer'][str(nbits)] = benchmark_signal_integrity(
                pipe,
                candidates,
                nbits,
                cumulative=False,
                ref_out=ref_out,
            )
            with open(os.path.join(args.o, json_name), 'w') as f:
                json.dump(results, f, indent=2)

        # Measure the cumulative impact of palettization based on ascending individual impact computed earlier
        sorted_candidates = descending_psnr_order(results['single_layer'][str(nbits)])

        if str(nbits) not in results['cumulative']:
            results['cumulative'][str(nbits)] = benchmark_signal_integrity(
                deepcopy(pipe),
                sorted_candidates,
                nbits,
                cumulative=True,
                ref_out=ref_out,
            )
            results['cumulative'][str(nbits)]['metadata'].update({
                'candidates': list(sorted_candidates.keys()),
                'sizes': [sizes_table[candidate] for candidate in sorted_candidates],
            })

            with open(os.path.join(args.o, json_name), 'w') as f:
                json.dump(results, f, indent=2)

    # Generate uniform-quantization baselines
    results['baselines'] = {
        "original": simulate_quant_fn(ref_pipe, lambda x: x)[1],
        "linear_8bit": simulate_quant_fn(ref_pipe, fake_int8_quantize)[1],
    }
    with open(os.path.join(args.o, json_name), 'w') as f:
        json.dump(results, f, indent=2)


    # Generate mixed-bit recipes via decreasing PSNR thresholds
    results['recipes'] = {}
    recipe_psnr_thresholds = np.linspace(
        results['baselines']['original'] - 1,
        results['baselines']["linear_8bit"] + 5,
        args.num_recipes,
    )

    for recipe_no, psnr_threshold in enumerate(recipe_psnr_thresholds):
        logger.info(f"Building recipe #{recipe_no}")
        recipe, stats = build_recipe(
            results['cumulative'],
            sizes_table,
            psnr_threshold,
            args.default_nbits,
        )
        achieved_psnr = simulate_quant_fn(ref_pipe, lambda x: partial(fake_palette_from_recipe, recipe=recipe)(x))[1]
        logger.info(
            f"Recipe #{recipe_no}: {stats['nbits']:.2f}-bits @ per-layer {psnr_threshold} dB, "
            f"end-to-end {achieved_psnr} dB & "
            f"{stats['size_mb']:.2f} MB"
        )

        # Save achieved PSNR and compressed size
        recipe_key = f"recipe_{stats['nbits']:.2f}_bit_mixedpalette"
        results['baselines'][recipe_key] = float(f"{achieved_psnr:.1f}")
        results['recipes'][recipe_key] = recipe

        with open(os.path.join(args.o, json_name), 'w') as f:
            json.dump(results, f, indent=2)

    # Plot model size vs signal integrity
    plot(results, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        required=True,
        help="Output directory to save the palettization artifacts (recipe json, PSNR plots etc.)"
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=stable-diffusion"
     ))
    parser.add_argument(
        "--default-nbits",
        help="Default number of bits to use for palettization",
        choices=tuple(NBITS + [16]),
        default=16,
        type=int,
    )
    parser.add_argument(
        "--num-recipes",
        help="Maximum number of recipes to generate (with decreasing model size and signal integrity)",
        default=7,
        type=int,
    )
    parser.add_argument(
        "--custom-vae-version",
        type=str,
        default=None,
        help=
        ("Custom VAE checkpoint to override the pipeline's built-in VAE. "
         "If specified, the specified VAE will be converted instead of the one associated to the `--model-version` checkpoint. "
         "No precision override is applied when using a custom VAE."
         ))
    # needed since this calls `torch2coreml` and that would throw an error
    parser.add_argument(
        "--sd3-version",
        action="store_true",
        help=("If specified, the pre-trained model will be treated as an SD3 model."))

    args = parser.parse_args()
    main(args)
