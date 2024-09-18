#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import coremltools as ct

import logging
import json

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np

import os
import time
import subprocess
import sys


def _macos_version():
    """
    Returns macOS version as a tuple of integers. On non-Macs, returns an empty tuple.
    """
    if sys.platform == "darwin":
        try:
            ver_str = subprocess.run(["sw_vers", "-productVersion"], stdout=subprocess.PIPE).stdout.decode('utf-8').strip('\n')
            return tuple([int(v) for v in ver_str.split(".")])
        except:
            raise Exception("Unable to determine the macOS version")
    return ()


class CoreMLModel:
    """ Wrapper for running CoreML models using coremltools
    """

    def __init__(self, model_path, compute_unit, sources='packages', optimization_hints=None):

        logger.info(f"Loading {model_path}")

        start = time.time()
        if sources == 'packages':
            assert os.path.exists(model_path) and model_path.endswith(".mlpackage")

            self.model = ct.models.MLModel(
                model_path,
                compute_units=ct.ComputeUnit[compute_unit],
                optimization_hints=optimization_hints,
            )
            DTYPE_MAP = {
                65552: np.float16,
                65568: np.float32,
                131104: np.int32,
            }
            self.expected_inputs = {
                input_tensor.name: {
                    "shape": tuple(input_tensor.type.multiArrayType.shape),
                    "dtype": DTYPE_MAP[input_tensor.type.multiArrayType.dataType],
                }
                for input_tensor in self.model._spec.description.input
            }
        elif sources == 'compiled':
            assert os.path.exists(model_path) and model_path.endswith(".mlmodelc")

            self.model = ct.models.CompiledMLModel(
                model_path,
                compute_units=ct.ComputeUnit[compute_unit],
                optimization_hints=optimization_hints,
            )

            # Grab expected inputs from metadata.json
            with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
                config = json.load(f)[0]

            self.expected_inputs = {
                input_tensor['name']: {
                    "shape": tuple(eval(input_tensor['shape'])),
                    "dtype": np.dtype(input_tensor['dataType'].lower()),
                }
                for input_tensor in config['inputSchema']
            }
        else:
            raise ValueError(f'Expected `packages` or `compiled` for sources, received {sources}')

        load_time = time.time() - start
        logger.info(f"Done. Took {load_time:.1f} seconds.")

        if load_time > LOAD_TIME_INFO_MSG_TRIGGER:
            logger.info(
                "Loading a CoreML model through coremltools triggers compilation every time. "
                "The Swift package we provide uses precompiled Core ML models (.mlmodelc) to avoid compile-on-load."
            )

    def _verify_inputs(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.expected_inputs:
                if not isinstance(v, np.ndarray):
                    raise TypeError(
                        f"Expected numpy.ndarray, got {v} for input: {k}")

                expected_dtype = self.expected_inputs[k]["dtype"]
                if not v.dtype == expected_dtype:
                    raise TypeError(
                        f"Expected dtype {expected_dtype}, got {v.dtype} for input: {k}"
                    )

                expected_shape = self.expected_inputs[k]["shape"]
                if not v.shape == expected_shape:
                    raise TypeError(
                        f"Expected shape {expected_shape}, got {v.shape} for input: {k}"
                    )
            else:
                raise ValueError(f"Received unexpected input kwarg: {k}")

    def __call__(self, **kwargs):
        self._verify_inputs(**kwargs)
        return self.model.predict(kwargs)


LOAD_TIME_INFO_MSG_TRIGGER = 10  # seconds


def get_resource_type(resources_dir: str) -> str:
    """
        Detect resource type based on filepath extensions.
        returns:
            `packages`: for .mlpackage resources
            'compiled`: for .mlmodelc resources
    """
    directories = [f for f in os.listdir(resources_dir) if os.path.isdir(os.path.join(resources_dir, f))]

    # consider directories ending with extension
    extensions = set([os.path.splitext(e)[1] for e in directories if os.path.splitext(e)[1]])

    # if one extension present we may be able to infer sources type
    if len(set(extensions)) == 1:
        extension = extensions.pop()
    else:
        raise ValueError(f'Multiple file extensions found at {resources_dir}.'
                         f'Cannot infer resource type from contents.')

    if extension == '.mlpackage':
        sources = 'packages'
    elif extension == '.mlmodelc':
        sources = 'compiled'
    else:
        raise ValueError(f'Did not find .mlpackage or .mlmodelc at {resources_dir}')

    return sources


def _load_mlpackage(submodule_name,
                    mlpackages_dir,
                    model_version,
                    compute_unit,
                    sources=None):
    """
        Load Core ML (mlpackage) models from disk (As exported by torch2coreml.py)

    """

    # if sources not provided, attempt to infer `packages` or `compiled` from the
    # resources directory
    if sources is None:
        sources = get_resource_type(mlpackages_dir)

    if sources == 'packages':
        logger.info(f"Loading {submodule_name} mlpackage")
        fname = f"Stable_Diffusion_version_{model_version}_{submodule_name}.mlpackage".replace(
            "/", "_")
        mlpackage_path = os.path.join(mlpackages_dir, fname)

        if not os.path.exists(mlpackage_path):
            raise FileNotFoundError(
                f"{submodule_name} CoreML model doesn't exist at {mlpackage_path}")

    elif sources == 'compiled':
        logger.info(f"Loading {submodule_name} mlmodelc")

        # FixMe: Submodule names and compiled resources names differ. Can change if names match in the future.
        submodule_names = ["text_encoder", "text_encoder_2", "unet", "vae_decoder", "vae_encoder", "safety_checker"]
        compiled_names = ['TextEncoder', 'TextEncoder2', 'Unet', 'VAEDecoder', 'VAEEncoder', 'SafetyChecker']
        name_map = dict(zip(submodule_names, compiled_names))

        cname = name_map[submodule_name] + '.mlmodelc'
        mlpackage_path = os.path.join(mlpackages_dir, cname)

        if not os.path.exists(mlpackage_path):
            raise FileNotFoundError(
                f"{submodule_name} CoreML model doesn't exist at {mlpackage_path}")

    # On macOS 15+, set fast prediction optimization hint for the unet.
    optimization_hints = None
    if submodule_name == "unet" and _macos_version() >= (15, 0):
        optimization_hints = {"specializationStrategy": ct.SpecializationStrategy.FastPrediction}

    return CoreMLModel(mlpackage_path,
                       compute_unit,
                       sources=sources,
                       optimization_hints=optimization_hints)


def _load_mlpackage_controlnet(mlpackages_dir, model_version, compute_unit):
    """ Load Core ML (mlpackage) models from disk (As exported by torch2coreml.py)
    """
    model_name = model_version.replace("/", "_")

    logger.info(f"Loading controlnet_{model_name} mlpackage")

    fname = f"ControlNet_{model_name}.mlpackage"

    mlpackage_path = os.path.join(mlpackages_dir, fname)

    if not os.path.exists(mlpackage_path):
        raise FileNotFoundError(
            f"controlnet_{model_name} CoreML model doesn't exist at {mlpackage_path}")

    return CoreMLModel(mlpackage_path, compute_unit)


def get_available_compute_units():
    return tuple(cu for cu in ct.ComputeUnit._member_names_)
