#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import contextlib
import coremltools as ct
from diffusers import StableDiffusionPipeline
import json
import logging
import numpy as np
import os
import unittest
from PIL import Image
from statistics import median
import tempfile
import time

import torch

torch.set_grad_enabled(False)

from python_coreml_stable_diffusion import torch2coreml, pipeline, coreml_model

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Testing configuration
TEST_SEED = 93
TEST_PROMPT = "a high quality photo of an astronaut riding a horse in space"
TEST_COMPUTE_UNIT = ["CPU_AND_GPU", "ALL", "CPU_AND_NE"]
TEST_PSNR_THRESHOLD = 35  # dB
TEST_ABSOLUTE_MAX_LATENCY = 90  # seconds
TEST_WARMUP_INFERENCE_STEPS = 3
TEST_TEXT_TO_IMAGE_SPEED_REPEATS = 3
TEST_MINIMUM_PROMPT_TO_IMAGE_CLIP_COSINE_SIMILARITY = 0.3  # in range [0.,1.]


class TestStableDiffusionForTextToImage(unittest.TestCase):
    """ Test Stable Diffusion text-to-image pipeline for:

    - PyTorch to CoreML conversion via coremltools
    - Speed of CoreML runtime across several compute units
    - Integration with `diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.py`
    - Efficacy of the safety_checker
    - Affinity of the generated image with the original prompt via CLIP score
    - The bridge between Python and Swift CLI
    - The signal parity of Swift CLI generated image with that of Python CLI
    """
    cli_args = None

    @classmethod
    def setUpClass(cls):
        cls.pytorch_pipe = StableDiffusionPipeline.from_pretrained(
            cls.cli_args.model_version,
            use_auth_token=True,
        )

        # To be initialized after test_torch_to_coreml_conversion is run
        cls.coreml_pipe = None
        cls.active_compute_unit = None

    @classmethod
    def tearDownClass(cls):
        cls.pytorch_pipe = None
        cls.coreml_pipe = None
        cls.active_compute_unit = None

    def test_torch_to_coreml_conversion(self):
        """ Tests:
        - PyTorch to CoreML conversion via coremltools
        """
        with self.subTest(model="vae_decoder"):
            logger.info("Converting vae_decoder")
            torch2coreml.convert_vae_decoder(self.pytorch_pipe, self.cli_args)
            logger.info("Successfully converted vae_decoder")

        with self.subTest(model="unet"):
            logger.info("Converting unet")
            torch2coreml.convert_unet(self.pytorch_pipe, self.cli_args)
            logger.info("Successfully converted unet")

        with self.subTest(model="text_encoder"):
            logger.info("Converting text_encoder")
            torch2coreml.convert_text_encoder(self.pytorch_pipe, self.cli_args)
            logger.info("Successfully converted text_encoder")

        with self.subTest(model="safety_checker"):
            logger.info("Converting safety_checker")
            torch2coreml.convert_safety_checker(self.pytorch_pipe,
                                                self.cli_args)
            logger.info("Successfully converted safety_checker")

    def test_end_to_end_image_generation_speed(self):
        """ Tests:
        - Speed of CoreML runtime across several compute units
        - Integration with `diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.py`
        """
        latency = {
            compute_unit:
            self._coreml_text_to_image_with_compute_unit(compute_unit)
            for compute_unit in TEST_COMPUTE_UNIT
        }
        latency["num_repeats_for_median"] = TEST_TEXT_TO_IMAGE_SPEED_REPEATS

        json_path = os.path.join(self.cli_args.o, "benchmark.json")
        logger.info(f"Saving inference benchmark results to {json_path}")
        with open(json_path, "w") as f:
            json.dump(latency, f)

        for compute_unit in TEST_COMPUTE_UNIT:
            with self.subTest(compute_unit=compute_unit):
                self.assertGreater(TEST_ABSOLUTE_MAX_LATENCY,
                                   latency[compute_unit])

    def test_image_to_prompt_clip_score(self):
        """ Tests:
        Affinity of the generated image with the original prompt via CLIP score
        """
        logger.warning(
            "This test will download the CLIP ViT-B/16 model (approximately 600 MB) from Hugging Face"
        )

        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16")

        for compute_unit in TEST_COMPUTE_UNIT:
            with self.subTest(compute_unit=compute_unit):
                image_path = pipeline.get_image_path(self.cli_args,
                                                     prompt=TEST_PROMPT,
                                                     compute_unit=compute_unit)
                image = Image.open(image_path)

                # Preprocess images and text for inference with CLIP
                inputs = processor(text=[TEST_PROMPT],
                                   images=image,
                                   return_tensors="pt",
                                   padding=True)
                outputs = model(**inputs)

                # Compute cosine similarity between image and text embeddings
                image_text_cosine_similarity = outputs.image_embeds @ outputs.text_embeds.T
                logger.info(
                    f"Image ({image_path}) to text ({TEST_PROMPT}) CLIP score: {image_text_cosine_similarity[0].item():.2f}"
                )

                # Ensure that the minimum cosine similarity threshold is achieved
                self.assertGreater(
                    image_text_cosine_similarity,
                    TEST_MINIMUM_PROMPT_TO_IMAGE_CLIP_COSINE_SIMILARITY,
                )

    def test_safety_checker_efficacy(self):
        """ Tests:
        - Efficacy of the safety_checker
        """
        self._init_coreml_pipe(compute_unit=self.active_compute_unit)

        safety_checker_test_prompt = "NSFW"
        image = self.coreml_pipe(safety_checker_test_prompt)

        # Image must have been erased by the safety checker
        self.assertEqual(np.array(image["images"][0]).sum(), 0.)
        self.assertTrue(image["nsfw_content_detected"].any())

    def test_swift_cli_image_generation(self):
        """ Tests:
        - The bridge between Python and Swift CLI
        - The signal parity of Swift CLI generated image with that of Python CLI
        """
        # coremltools to Core ML compute unit mapping
        compute_unit_map = {
            "ALL": "all",
            "CPU_AND_GPU": "cpuAndGPU",
            "CPU_AND_NE": "cpuAndNeuralEngine"
        }

        # Prepare resources for Swift CLI
        resources_dir = torch2coreml.bundle_resources_for_swift_cli(
            self.cli_args)
        logger.info("Bundled resources for Swift CLI")

        # Execute image generation with Swift CLI
        # Note: First time takes ~5 minutes due to project building and so on
        cmd = " ".join([
            f"swift run StableDiffusionSample \"{TEST_PROMPT}\"",
            f"--resource-path {resources_dir}",
            f"--seed {TEST_SEED}",
            f"--output-path {self.cli_args.o}",
            f"--compute-units {compute_unit_map[TEST_COMPUTE_UNIT[-1]]}"
        ])
        logger.info(f"Executing `{cmd}`")
        os.system(cmd)
        logger.info(f"Image generation with Swift CLI is complete")

        # Load Swift CLI generated image
        swift_cli_image = Image.open(
            os.path.join(
                self.cli_args.o, "_".join(TEST_PROMPT.rsplit(" ")) + "." +
                str(TEST_SEED) + ".final.png"))

        # Load Python CLI (pipeline.py) generated image
        python_cli_image = Image.open(pipeline.get_image_path(self.cli_args,
                                                              prompt=TEST_PROMPT,
                                                              compute_unit=TEST_COMPUTE_UNIT[-1]))

        # Compute signal parity
        swift2torch_psnr = torch2coreml.report_correctness(
            np.array(swift_cli_image.convert("RGB")),
            np.array(python_cli_image.convert("RGB")),
            "Swift CLI and Python CLI generated images")
        self.assertGreater(swift2torch_psnr, torch2coreml.ABSOLUTE_MIN_PSNR)

    def _init_coreml_pipe(self, compute_unit):
        """ Initializes CoreML pipe for the requested compute_unit
        """
        assert compute_unit in ct.ComputeUnit._member_names_, f"Not a valid coremltools.ComputeUnit: {compute_unit}"

        if self.active_compute_unit == compute_unit:
            logger.info(
                "self.coreml_pipe matches requested compute_unit, skipping reinitialization"
            )
            assert \
                isinstance(self.coreml_pipe, pipeline.CoreMLStableDiffusionPipeline), \
                type(self.coreml_pipe)
        else:
            self.active_compute_unit = compute_unit
            self.coreml_pipe = pipeline.get_coreml_pipe(
                pytorch_pipe=self.pytorch_pipe,
                mlpackages_dir=self.cli_args.o,
                model_version=self.cli_args.model_version,
                compute_unit=self.active_compute_unit,)


    def _coreml_text_to_image_with_compute_unit(self, compute_unit):
        """ Benchmark end-to-end text-to-image generation with the requested compute_unit
        """
        self._init_coreml_pipe(compute_unit)

        # Warm up (not necessary in all settings but improves consistency for benchmarking)
        logger.info(
            f"Warmup image generation with {TEST_WARMUP_INFERENCE_STEPS} inference steps"
        )
        image = self.coreml_pipe(
            TEST_PROMPT, num_inference_steps=TEST_WARMUP_INFERENCE_STEPS)

        # Test end-to-end speed
        logger.info(
            f"Run full image generation {TEST_TEXT_TO_IMAGE_SPEED_REPEATS} times and report median"
        )

        def test_coreml_text_to_image_speed():
            """ Execute Core ML based image generation
            """
            _reset_seed()
            image = self.coreml_pipe(TEST_PROMPT)["images"][0]
            out_path = pipeline.get_image_path(self.cli_args,
                                        prompt=TEST_PROMPT,
                                        compute_unit=compute_unit)
            logger.info(f"Saving generated image to {out_path}")
            image.save(out_path)

        def collect_timings(callable, n):
            """ Collect user latency for callable
            """
            user_latencies = []
            for _ in range(n):
                s = time.time()
                callable()
                user_latencies.append(float(f"{time.time() - s:.2f}"))
            return user_latencies

        coreml_latencies = collect_timings(
            callable=test_coreml_text_to_image_speed,
            n=TEST_TEXT_TO_IMAGE_SPEED_REPEATS)
        coreml_median_latency = median(coreml_latencies)

        logger.info(
            f"End-to-end latencies with coremltools.ComputeUnit.{compute_unit}: median={coreml_median_latency:.2f}"
        )

        return coreml_median_latency


def _reset_seed():
    """ Reset RNG state in order to reproduce the results across multiple runs
    """
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)


def _get_test_artifacts_dir(args):
    if cli_args.persistent_test_artifacts_dir is not None:
        os.makedirs(cli_args.persistent_test_artifacts_dir, exist_ok=True)
        return contextlib.nullcontext(
            enter_result=cli_args.persistent_test_artifacts_dir)
    else:
        return tempfile.TemporaryDirectory(
            prefix="python_coreml_stable_diffusion_tests")


def _extend_parser(parser):
    parser.add_argument(
        "--persistent-test-artifacts-dir",
        type=str,
        default=None,
        help=
        ("If specified, test artifacts such as Core ML models and generated images are saved in this directory. ",
         "Otherwise, all artifacts are erased after the test program terminates."
         ))
    parser.add_argument(
        "--fast",
        action="store_true",
        help=
        "If specified, runs fewer repeats for `test_end_to_end_image_generation_speed`"
    )
    parser.add_argument(
        "--test-image-to-prompt-clip-score-opt-in",
        action="store_true",
        help=
        ("If specified, enables `test_image_to_prompt_clip_score` to verify the relevance of the "
         "generated image content to the original text prompt. This test is an opt-in "
         "test because it involves an additional one time 600MB model download."
         ))
    parser.add_argument(
        "--test-swift-cli-opt-in",
        action="store_true",
        help=
        ("If specified, compiles all models and builds the Swift CLI to run image generation and compares "
         "results across Python and Swift runtime"))
    parser.add_argument(
        "--test-safety-checker-efficacy-opt-in",
        action="store_true",
        help=
        ("If specified, generates a potentially NSFW image to check whether the `safety_checker` "
         "accurately detects and removes the content"))
    return parser


if __name__ == "__main__":
    # Reproduce the CLI of the original pipeline
    parser = torch2coreml.parser_spec()
    parser = _extend_parser(parser)
    cli_args = parser.parse_args()

    cli_args.check_output_correctness = True
    cli_args.prompt = TEST_PROMPT
    cli_args.seed = TEST_SEED
    cli_args.compute_unit = TEST_COMPUTE_UNIT[0]
    cli_args.scheduler = None  # use default
    torch2coreml.ABSOLUTE_MIN_PSNR = TEST_PSNR_THRESHOLD

    if cli_args.fast:
        logger.info(
            "`--fast` detected: Image generation will be run once " \
            f"(instead of {TEST_TEXT_TO_IMAGE_SPEED_REPEATS } times) " \
            "with ComputeUnit.ALL (other compute units are skipped)" \
            " (median can not be reported)")
        TEST_TEXT_TO_IMAGE_SPEED_REPEATS = 1
        TEST_COMPUTE_UNIT = ["ALL"]

        logger.info("`--fast` detected: Skipping `--check-output-correctness` tests")
        cli_args.check_output_correctness = False
    elif cli_args.attention_implementation == "ORIGINAL":
        TEST_COMPUTE_UNIT = ["CPU_AND_GPU", "ALL"]
    elif cli_args.attention_implementation == "SPLIT_EINSUM":
        TEST_COMPUTE_UNIT = ["ALL", "CPU_AND_NE"]

    logger.info(f"Testing compute units: {TEST_COMPUTE_UNIT}")


    # Save CoreML model files and generated images into the artifacts dir
    with _get_test_artifacts_dir(cli_args) as test_artifacts_dir:
        cli_args.o = test_artifacts_dir
        logger.info(f"Test artifacts will be saved under {test_artifacts_dir}")

        TestStableDiffusionForTextToImage.cli_args = cli_args

        # Run the following tests in sequential order
        suite = unittest.TestSuite()
        suite.addTest(
            TestStableDiffusionForTextToImage(
                "test_torch_to_coreml_conversion"))
        suite.addTest(
            TestStableDiffusionForTextToImage(
                "test_end_to_end_image_generation_speed"))

        if cli_args.test_safety_checker_efficacy_opt_in:
            suite.addTest(
                TestStableDiffusionForTextToImage("test_safety_checker_efficacy"))

        if cli_args.test_image_to_prompt_clip_score_opt_in:
            suite.addTest(
                TestStableDiffusionForTextToImage(
                    "test_image_to_prompt_clip_score"))

        if cli_args.test_swift_cli_opt_in:
            suite.addTest(
                TestStableDiffusionForTextToImage(
                    "test_swift_cli_image_generation"))

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)
