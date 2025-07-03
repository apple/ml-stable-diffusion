#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
from collections import OrderedDict

import coremltools as ct
from coremltools.converters.mil import Block, Program, Var
from coremltools.converters.mil.frontend.milproto.load import load as _milproto_to_pymil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Placeholder
from coremltools.converters.mil.mil import types as types
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import random_gen_input_feature_type

import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
from python_coreml_stable_diffusion import torch2coreml
import shutil
import time


def _verify_output_correctness_of_chunks(full_model,
                                         first_chunk_model=None,
                                         second_chunk_model=None,
                                         pipeline_model=None,):
    """ Verifies the end-to-end output correctness of full (original) model versus chunked models
    """
    # Generate inputs for first chunk and full model
    input_dict = {}
    for input_desc in full_model._spec.description.input:
        input_dict[input_desc.name] = random_gen_input_feature_type(input_desc)

    # Generate outputs for full model
    outputs_from_full_model = full_model.predict(input_dict)

    if pipeline_model is not None:
        outputs_from_pipeline_model = pipeline_model.predict(input_dict)
        final_outputs = outputs_from_pipeline_model

    elif first_chunk_model is not None and second_chunk_model is not None:
        # Generate outputs for first chunk
        outputs_from_first_chunk_model = first_chunk_model.predict(input_dict)

        # Prepare inputs for second chunk model from first chunk's outputs and regular inputs
        second_chunk_input_dict = {}
        for input_desc in second_chunk_model._spec.description.input:
            if input_desc.name in outputs_from_first_chunk_model:
                second_chunk_input_dict[
                    input_desc.name] = outputs_from_first_chunk_model[
                        input_desc.name]
            else:
                second_chunk_input_dict[input_desc.name] = input_dict[
                    input_desc.name]

        # Generate output for second chunk model
        outputs_from_second_chunk_model = second_chunk_model.predict(
            second_chunk_input_dict)
        final_outputs = outputs_from_second_chunk_model
    else:
        raise ValueError

    # Verify correctness across all outputs from second chunk and full model
    for out_name in outputs_from_full_model.keys():
        torch2coreml.report_correctness(
            original_outputs=outputs_from_full_model[out_name],
            final_outputs=final_outputs[out_name],
            log_prefix=f"{out_name}")


def _load_prog_from_mlmodel(model):
    """ Load MIL Program from an MLModel
    """
    model_spec = model.get_spec()
    start_ = time.time()
    logger.info(
        "Loading MLModel object into a MIL Program object (including the weights).."
    )
    prog = _milproto_to_pymil(
        model_spec=model_spec,
        specification_version=model_spec.specificationVersion,
        file_weights_dir=model.weights_dir,
    )
    logger.info(f"Program loaded in {time.time() - start_:.1f} seconds")

    return prog


def _get_op_idx_split_location(prog: Program):
    """ Find the op that approximately bisects the graph as measure by weights size on each side
    """
    main_block = prog.functions["main"]
    main_block.operations = list(main_block.operations)
    total_size_in_mb = 0

    for op in main_block.operations:
        if op.op_type == "const" and isinstance(op.val.val, np.ndarray):
            size_in_mb = op.val.val.size * op.val.val.itemsize / (1024 * 1024)
            total_size_in_mb += size_in_mb
    half_size = total_size_in_mb / 2

    # Find the first non const op (single child), where the total cumulative size exceeds
    # the half size for the first time
    cumulative_size_in_mb = 0
    for op in main_block.operations:
        if op.op_type == "const" and isinstance(op.val.val, np.ndarray):
            size_in_mb = op.val.val.size * op.val.val.itemsize / (1024 * 1024)
            cumulative_size_in_mb += size_in_mb

        # Note: The condition "not op.op_type.startswith("const")" is to make sure that the
        # incision op is neither of type "const" nor "constexpr_*" ops that
        # are used to store compressed weights
        if (cumulative_size_in_mb > half_size and not op.op_type.startswith("const")
                and len(op.outputs) == 1
                and len(op.outputs[0].child_ops) == 1):
            op_idx = main_block.operations.index(op)
            return op_idx, cumulative_size_in_mb, total_size_in_mb


def _get_first_chunk_outputs(block, op_idx):
    # Get the list of all vars that go across from first program (all ops from 0 to op_idx (inclusive))
    # to the second program (all ops from op_idx+1 till the end). These all vars need to be made the output
    # of the first program and the input of the second program
    boundary_vars = set()
    block.operations = list(block.operations)
    for i in range(op_idx + 1):
        op = block.operations[i]
        if not op.op_type.startswith("const"):
            for var in op.outputs:
                if var.val is None:  # only consider non const vars
                    for child_op in var.child_ops:
                        child_op_idx = block.operations.index(child_op)
                        if child_op_idx > op_idx:
                            boundary_vars.add(var)
    return list(boundary_vars)


@block_context_manager
def _add_fp32_casts(block, boundary_vars):
    new_boundary_vars = []
    for var in boundary_vars:
        if var.dtype != types.fp16:
            new_boundary_vars.append(var)
        else:
            fp32_var = mb.cast(x=var, dtype="fp32", name=var.name)
            new_boundary_vars.append(fp32_var)
    return new_boundary_vars


def _make_first_chunk_prog(prog, op_idx):
    """ Build first chunk by declaring early outputs and removing unused subgraph
    """
    block = prog.functions["main"]
    boundary_vars = _get_first_chunk_outputs(block, op_idx)

    # Due to possible numerical issues, cast any fp16 var to fp32
    new_boundary_vars = _add_fp32_casts(block, boundary_vars)

    block.outputs.clear()
    block.set_outputs(new_boundary_vars)
    PASS_REGISTRY["common::dead_code_elimination"](prog)
    return prog


def _make_second_chunk_prog(prog, op_idx):
    """ Build second chunk by rebuilding a pristine MIL Program from MLModel
    """
    block = prog.functions["main"]
    block.opset_version = ct.target.iOS16

    # First chunk outputs are second chunk inputs (e.g. skip connections)
    boundary_vars = _get_first_chunk_outputs(block, op_idx)

    # This op will not be included in this program. Its output var will be made into an input
    block.operations = list(block.operations)
    boundary_op = block.operations[op_idx]

    # Add all boundary ops as inputs
    with block:
        for var in boundary_vars:
            new_placeholder = Placeholder(
                sym_shape=var.shape,
                dtype=var.dtype if var.dtype != types.fp16 else types.fp32,
                name=var.name,
            )

            block._input_dict[
                new_placeholder.outputs[0].name] = new_placeholder.outputs[0]

            block.function_inputs = tuple(block._input_dict.values())
            new_var = None
            if var.dtype == types.fp16:
                new_var = mb.cast(x=new_placeholder.outputs[0],
                                  dtype="fp16",
                                  before_op=var.op)
            else:
                new_var = new_placeholder.outputs[0]

            block.replace_uses_of_var_after_op(
                anchor_op=boundary_op,
                old_var=var,
                new_var=new_var,
                # This is needed if the program contains "constexpr_*" ops. In normal cases, there are stricter
                # rules for removing them, and their presence may prevent replacing this var.
                # However in this case, since we want to remove all the ops in chunk 1, we can safely
                # set this to True.
                force_replace=True,
            )

    PASS_REGISTRY["common::dead_code_elimination"](prog)

    # Remove any unused inputs
    new_input_dict = OrderedDict()
    for k, v in block._input_dict.items():
        if len(v.child_ops) > 0:
            new_input_dict[k] = v
    block._input_dict = new_input_dict
    block.function_inputs = tuple(block._input_dict.values())

    return prog

def main(args):
    ct_version = ct.__version__

    # Starting from coremltools==8.0b2, there is this `bisect_model` API that
    # we can directly call into.
    from coremltools.models.utils import bisect_model
    logger.info(f"Start chunking model {args.mlpackage_path} into two pieces.")
    ct.models.utils.bisect_model(
        model=args.mlpackage_path,
        output_dir=args.o,
        merge_chunks_to_pipeline=args.merge_chunks_in_pipeline_model,
        check_output_correctness=args.check_output_correctness,
    )
    logger.info(f"Model chunking is done.")

    # Remove original (non-chunked) model if requested
    if args.remove_original:
        logger.info(
            "Removing original (non-chunked) model at {args.mlpackage_path}")
        shutil.rmtree(args.mlpackage_path)
        logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlpackage-path",
        required=True,
        help=
        "Path to the mlpackage file to be split into two mlpackages of approximately same file size.",
    )
    parser.add_argument(
        "-o",
        required=True,
        help=
        "Path to output directory where the two model chunks should be saved.",
    )
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help=
        "If specified, removes the original (non-chunked) model to avoid duplicating storage."
    )
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        ("If specified, compares the outputs of original Core ML model with that of pipelined CoreML model chunks and reports PSNR in dB. ",
         "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
         ))
    parser.add_argument(
        "--merge-chunks-in-pipeline-model",
        action="store_true",
        help=
        ("If specified, model chunks are managed inside a single pipeline model for easier asset maintenance"
         ))

    args = parser.parse_args()
    main(args)
