#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse

from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin

import gc
import inspect

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os

from python_coreml_stable_diffusion.coreml_model import (
    CoreMLModel,
    _load_mlpackage,
    _load_mlpackage_controlnet,
    get_available_compute_units,
)

import time
import torch  # Only used for `torch.from_tensor` in `pipe.scheduler.step()`
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from typing import List, Optional, Union
from PIL import Image


class CoreMLStableDiffusionPipeline(DiffusionPipeline):
    """ Core ML version of
    `diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline`
    """

    def __init__(
        self,
        text_encoder: CoreMLModel,
        unet: CoreMLModel,
        vae_decoder: CoreMLModel,
        feature_extractor: CLIPFeatureExtractor,
        safety_checker: Optional[CoreMLModel],
        scheduler: Union[DDIMScheduler,
                         DPMSolverMultistepScheduler,
                         EulerAncestralDiscreteScheduler,
                         EulerDiscreteScheduler,
                         LMSDiscreteScheduler,
                         PNDMScheduler],
        tokenizer: CLIPTokenizer,
        controlnet: Optional[List[CoreMLModel]],
    ):
        super().__init__()

        # Register non-Core ML components of the pipeline similar to the original pipeline
        self.register_modules(
            tokenizer=tokenizer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

        if safety_checker is None:
            # Reproduce original warning:
            # https://github.com/huggingface/diffusers/blob/v0.9.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L119
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # Register Core ML components of the pipeline
        self.safety_checker = safety_checker
        self.text_encoder = text_encoder
        self.unet = unet
        self.unet.in_channels = self.unet.expected_inputs["sample"]["shape"][1]

        self.controlnet = controlnet

        self.vae_decoder = vae_decoder

        VAE_DECODER_UPSAMPLE_FACTOR = 8

        # In PyTorch, users can determine the tensor shapes dynamically by default
        # In CoreML, tensors have static shapes unless flexible shapes were used during export
        # See https://coremltools.readme.io/docs/flexible-inputs
        latent_h, latent_w = self.unet.expected_inputs["sample"]["shape"][2:]
        self.height = latent_h * VAE_DECODER_UPSAMPLE_FACTOR
        self.width = latent_w * VAE_DECODER_UPSAMPLE_FACTOR

        logger.info(
            f"Stable Diffusion configured to generate {self.height}x{self.width} images"
        )

    def _encode_prompt(self, prompt, num_images_per_prompt,
                       do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(
                text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}")
            text_input_ids = text_input_ids[:, :self.tokenizer.
                                            model_max_length]

        text_embeddings = self.text_encoder(
            input_ids=text_input_ids.astype(np.float32))["last_hidden_state"]

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    "`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    " {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.astype(
                    np.float32))["last_hidden_state"]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate(
                [uncond_embeddings, text_embeddings])

        text_embeddings = text_embeddings.transpose(0, 2, 1)[:, :, None, :]

        return text_embeddings

    def run_controlnet(self, 
                       sample, 
                       timestep, 
                       encoder_hidden_states, 
                       controlnet_cond, 
                       output_dtype=np.float16):
        if not self.controlnet:
            raise ValueError(
                "Conditions for controlnet are given but the pipeline has no controlnet modules")

        for i, (module, cond) in enumerate(zip(self.controlnet, controlnet_cond)):
            module_outputs = module(
                sample=sample.astype(np.float16), 
                timestep=timestep.astype(np.float16), 
                encoder_hidden_states=encoder_hidden_states.astype(np.float16), 
                controlnet_cond=cond.astype(np.float16),
            )
            if i == 0:
                outputs = module_outputs
            else:
                for key in outputs.keys():
                    outputs[key] += module_outputs[key]
        
        outputs = {k: v.astype(output_dtype) for k, v in outputs.items()}
        
        return outputs

    def run_safety_checker(self, image):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image),
                return_tensors="np",
            )

            safety_checker_outputs = self.safety_checker(
                clip_input=safety_checker_input.pixel_values.astype(
                    np.float16),
                images=image.astype(np.float16),
                adjustment=np.array([0.]).astype(
                    np.float16),  # defaults to 0 in original pipeline
            )

            # Unpack dict
            has_nsfw_concept = safety_checker_outputs["has_nsfw_concepts"]
            image = safety_checker_outputs["filtered_images"]
            concept_scores = safety_checker_outputs["concept_scores"]

            logger.info(
                f"Generated image has nsfw concept={has_nsfw_concept.any()}")
        else:
            has_nsfw_concept = None

        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(z=latents.astype(np.float16))["image"]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        return image

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        latents=None):
        latents_shape = (batch_size, num_channels_latents, self.height // 8,
                         self.width // 8)
        if latents is None:
            latents = np.random.randn(*latents_shape).astype(np.float16)
        elif latents.shape != latents_shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
            )

        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def prepare_control_cond(self, 
                             controlnet_cond, 
                             do_classifier_free_guidance, 
                             batch_size, 
                             num_images_per_prompt):
        processed_cond_list = []
        for cond in controlnet_cond:
            cond = np.stack([cond] * batch_size * num_images_per_prompt)
            if do_classifier_free_guidance:
                cond = np.concatenate([cond] * 2)
            processed_cond_list.append(cond)
        return processed_cond_list

    def check_inputs(self, prompt, height, width, callback_steps):
        if height != self.height or width != self.width:
            logger.warning(
                "`height` and `width` dimensions (of the output image tensor) are fixed when exporting the Core ML models " \
                "unless flexible shapes are used during export (https://coremltools.readme.io/docs/flexible-inputs). " \
                "This pipeline was provided with Core ML models that generate {self.height}x{self.width} images (user requested {height}x{width})"
            )

        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (callback_steps is not None and
                                        (not isinstance(callback_steps, int)
                                         or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}.")

    def prepare_extra_step_kwargs(self, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        return extra_step_kwargs

    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        latents=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        controlnet_cond=None,
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if batch_size > 1 or num_images_per_prompt > 1:
            raise NotImplementedError(
                "For batched generation of multiple images and/or multiple prompts, please refer to the Swift package."
            )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables and controlnet cond
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            latents,
        )

        if controlnet_cond:
            controlnet_cond = self.prepare_control_cond(
                controlnet_cond,
                do_classifier_free_guidance,
                batch_size,
                num_images_per_prompt,
            )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # 7. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # controlnet
            if controlnet_cond:
                additional_residuals = self.run_controlnet(
                    sample=latent_model_input,
                    timestep=np.array([t, t]),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_cond,
                )
            else:
                additional_residuals = {}

            # predict the noise residual
            noise_pred = self.unet(
                sample=latent_model_input.astype(np.float16),
                timestep=np.array([t, t], np.float16),
                encoder_hidden_states=text_embeddings.astype(np.float16),
                **additional_residuals,
            )["noise_pred"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(torch.from_numpy(noise_pred),
                                          t,
                                          torch.from_numpy(latents),
                                          **extra_step_kwargs,
            ).prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)


def get_available_schedulers():
    schedulers = {}
    for scheduler in [DDIMScheduler,
                      DPMSolverMultistepScheduler,
                      EulerAncestralDiscreteScheduler,
                      EulerDiscreteScheduler,
                      LMSDiscreteScheduler,
                      PNDMScheduler]:
        schedulers[scheduler().__class__.__name__.replace("Scheduler", "")] = scheduler
    return schedulers

SCHEDULER_MAP = get_available_schedulers()

def get_coreml_pipe(pytorch_pipe,
                    mlpackages_dir,
                    model_version,
                    compute_unit,
                    delete_original_pipe=True,
                    scheduler_override=None,
                    controlnet_models=None):
    """ Initializes and returns a `CoreMLStableDiffusionPipeline` from an original
    diffusers PyTorch pipeline
    """
    # Ensure `scheduler_override` object is of correct type if specified
    if scheduler_override is not None:
        assert isinstance(scheduler_override, SchedulerMixin)
        logger.warning(
            "Overriding scheduler in pipeline: "
            f"Default={pytorch_pipe.scheduler}, Override={scheduler_override}")

    # Gather configured tokenizer and scheduler attributes from the original pipe
    coreml_pipe_kwargs = {
        "tokenizer": pytorch_pipe.tokenizer,
        "scheduler": pytorch_pipe.scheduler if scheduler_override is None else scheduler_override,
        "feature_extractor": pytorch_pipe.feature_extractor,
    }

    model_names_to_load = ["text_encoder", "unet", "vae_decoder"]
    if getattr(pytorch_pipe, "safety_checker", None) is not None:
        model_names_to_load.append("safety_checker")
    else:
        logger.warning(
            f"Original diffusers pipeline for {model_version} does not have a safety_checker, "
            "Core ML pipeline will mirror this behavior.")
        coreml_pipe_kwargs["safety_checker"] = None

    if delete_original_pipe:
        del pytorch_pipe
        gc.collect()
        logger.info("Removed PyTorch pipe to reduce peak memory consumption")

    if controlnet_models:
        model_names_to_load.remove("unet")
        coreml_pipe_kwargs["unet"] = _load_mlpackage(
            "control-unet",
            mlpackages_dir,
            model_version,
            compute_unit,
        )
        coreml_pipe_kwargs["controlnet"] = [_load_mlpackage_controlnet(
            mlpackages_dir, 
            model_version, 
            compute_unit,
        ) for model_version in controlnet_models]
    else:
        coreml_pipe_kwargs["controlnet"] = None

    # Load Core ML models
    logger.info(f"Loading Core ML models in memory from {mlpackages_dir}")
    coreml_pipe_kwargs.update({
        model_name: _load_mlpackage(
            model_name,
            mlpackages_dir,
            model_version,
            compute_unit,
        )
        for model_name in model_names_to_load
    })
    logger.info("Done.")

    logger.info("Initializing Core ML pipe for image generation")
    coreml_pipe = CoreMLStableDiffusionPipeline(**coreml_pipe_kwargs)
    logger.info("Done.")

    return coreml_pipe


def get_image_path(args, **override_kwargs):
    """ mkdir output folder and encode metadata in the filename
    """
    out_folder = os.path.join(args.o, "_".join(args.prompt.replace("/", "_").rsplit(" ")))
    os.makedirs(out_folder, exist_ok=True)

    out_fname = f"randomSeed_{override_kwargs.get('seed', None) or args.seed}"
    out_fname += f"_computeUnit_{override_kwargs.get('compute_unit', None) or args.compute_unit}"
    out_fname += f"_modelVersion_{override_kwargs.get('model_version', None) or args.model_version.replace('/', '_')}"

    if args.scheduler is not None:
        out_fname += f"_customScheduler_{override_kwargs.get('scheduler', None) or args.scheduler}"
        out_fname += f"_numInferenceSteps{override_kwargs.get('num_inference_steps', None) or args.num_inference_steps}"

    return os.path.join(out_folder, out_fname + ".png")

def prepare_controlnet_cond(image_path, height, width):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((height, width), resample=Image.LANCZOS)
    image = np.array(image).transpose(2, 0, 1) / 255.0
    return image

def main(args):
    logger.info(f"Setting random seed to {args.seed}")
    np.random.seed(args.seed)

    logger.info("Initializing PyTorch pipe for reference configuration")
    from diffusers import StableDiffusionPipeline
    pytorch_pipe = StableDiffusionPipeline.from_pretrained(args.model_version,
                                                           use_auth_token=True)

    user_specified_scheduler = None
    if args.scheduler is not None:
        user_specified_scheduler = SCHEDULER_MAP[
            args.scheduler].from_config(pytorch_pipe.scheduler.config)

    coreml_pipe = get_coreml_pipe(pytorch_pipe=pytorch_pipe,
                                  mlpackages_dir=args.i,
                                  model_version=args.model_version,
                                  compute_unit=args.compute_unit,
                                  scheduler_override=user_specified_scheduler,
                                  controlnet_models=args.controlnet)

    if args.controlnet:
        controlnet_cond = []
        for i, _ in enumerate(args.controlnet):
            image_path = args.controlnet_inputs[i]
            image = prepare_controlnet_cond(image_path, coreml_pipe.height, coreml_pipe.width)
            controlnet_cond.append(image)
    else:
        controlnet_cond = None

    logger.info("Beginning image generation.")
    image = coreml_pipe(
        prompt=args.prompt,
        height=coreml_pipe.height,
        width=coreml_pipe.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_cond=controlnet_cond,
        negative_prompt=args.negative_prompt,
    )

    out_path = get_image_path(args)
    logger.info(f"Saving generated image to {out_path}")
    image["images"][0].save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        required=True,
        help="The text prompt to be used for text-to-image generation.")
    parser.add_argument(
        "-i",
        required=True,
        help=("Path to input directory with the .mlpackage files generated by "
              "python_coreml_stable_diffusion.torch2coreml"))
    parser.add_argument("-o", required=True)
    parser.add_argument("--seed",
                        "-s",
                        default=93,
                        type=int,
                        help="Random seed to be able to reproduce results")
    parser.add_argument(
        "--model-version",
        default="CompVis/stable-diffusion-v1-4",
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=stable-diffusion"
         ))
    parser.add_argument(
        "--compute-unit",
        choices=get_available_compute_units(),
        default="ALL",
        help=("The compute units to be used when executing Core ML models. "
              f"Options: {get_available_compute_units()}"))
    parser.add_argument(
        "--scheduler",
        choices=tuple(SCHEDULER_MAP.keys()),
        default=None,
        help=("The scheduler to use for running the reverse diffusion process. "
             "If not specified, the default scheduler from the diffusers pipeline is utilized"))
    parser.add_argument(
        "--num-inference-steps",
        default=50,
        type=int,
        help="The number of iterations the unet model will be executed throughout the reverse diffusion process")
    parser.add_argument(
        "--guidance-scale",
        default=7.5,
        type=float,
        help="Controls the influence of the text prompt on sampling process (0=random images)")
    parser.add_argument(
        "--controlnet",
        nargs="*", 
        type=str,
        help=("Enables ControlNet and use control-unet instead of unet for additional inputs. "
            "For Multi-Controlnet, provide the model names separated by spaces."))
    parser.add_argument(
        "--controlnet-inputs",
        nargs="*", 
        type=str,
        help=("Image paths for ControlNet inputs. "
            "Please enter images corresponding to each controlnet provided at --controlnet option in same order."))
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="The negative text prompt to be used for text-to-image generation.")

    args = parser.parse_args()
    main(args)
