#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import coremltools as ct

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np

import os
import time


class CoreMLModel:
    """ Wrapper for running CoreML models using coremltools
    """

    def __init__(self, model_path, compute_unit):
        assert os.path.exists(model_path) and model_path.endswith(".mlpackage")

        logger.info(f"Loading {model_path}")

        start = time.time()
        self.model = ct.models.MLModel(
            model_path, compute_units=ct.ComputeUnit[compute_unit])
        load_time = time.time() - start
        logger.info(f"Done. Took {load_time:.1f} seconds.")

        if load_time > LOAD_TIME_INFO_MSG_TRIGGER:
            logger.info(
                "Loading a CoreML model through coremltools triggers compilation every time. "
                "The Swift package we provide uses precompiled Core ML models (.mlmodelc) to avoid compile-on-load."
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
                raise ValueError("Received unexpected input kwarg: {k}")

    def __call__(self, **kwargs):
        self._verify_inputs(**kwargs)
        return self.model.predict(kwargs)


LOAD_TIME_INFO_MSG_TRIGGER = 10  # seconds


def _load_mlpackage(submodule_name, mlpackages_dir, model_version,
                    compute_unit):
    """ Load Core ML (mlpackage) models from disk (As exported by torch2coreml.py)
    """
    logger.info(f"Loading {submodule_name} mlpackage")

    fname = f"Stable_Diffusion_version_{model_version}_{submodule_name}.mlpackage".replace(
        "/", "_")
    mlpackage_path = os.path.join(mlpackages_dir, fname)

    if not os.path.exists(mlpackage_path):
        raise FileNotFoundError(
            f"{submodule_name} CoreML model doesn't exist at {mlpackage_path}")

    return CoreMLModel(mlpackage_path, compute_unit)

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
