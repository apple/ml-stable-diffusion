#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
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
from typing import List, Optional, Union, Tuple
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
            scheduler: Union[
                DDIMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                LMSDiscreteScheduler,
                PNDMScheduler
            ],
            tokenizer: CLIPTokenizer,
            controlnet: Optional[List[CoreMLModel]],
            xl: Optional[bool] = False,
            force_zeros_for_empty_prompt: Optional[bool] = True,
            feature_extractor: Optional[CLIPFeatureExtractor] = None,
            safety_checker: Optional[CoreMLModel] = None,
            text_encoder_2: Optional[CoreMLModel] = None,
            tokenizer_2: Optional[CLIPTokenizer] = None

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
        self.xl = xl
        self.force_zeros_for_empty_prompt = force_zeros_for_empty_prompt

        # Register Core ML components of the pipeline
        self.safety_checker = safety_checker
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
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

    def _encode_prompt(self,
                       prompt,
                       prompt_2: Optional[str] = None,
                       do_classifier_free_guidance: bool = True,
                       negative_prompt: Optional[str] = None,
                       negative_prompt_2: Optional[str] = None,
                       ):

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if self.xl is True:
            prompts = [prompt, prompt_2] if prompt_2 is not None else [prompt, prompt]

            # refiner uses only one tokenizer and text encoder (tokenizer_2 and text_encoder_2)
            tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]

            text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [
                self.text_encoder_2]
            hidden_state_key = 'hidden_embeds'
        else:
            prompts = [prompt]
            tokenizers = [self.tokenizer]
            text_encoders = [self.text_encoder]
            hidden_state_key = 'last_hidden_state'

        prompt_embeds_list = []
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )

            text_input_ids = text_inputs.input_ids

            # tokenize without max_length to catch any truncation
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="np").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            embeddings = text_encoder(input_ids=text_input_ids.astype(np.float32))

            prompt_embeds_list.append(embeddings[hidden_state_key])

            # We are only ALWAYS interested in the pooled output of the final text encoder
            if self.xl:
                pooled_prompt_embeds = embeddings['pooled_outputs']

        prompt_embeds = np.concatenate(prompt_embeds_list, axis=-1)

        if do_classifier_free_guidance and negative_prompt is None and self.force_zeros_for_empty_prompt:
            negative_prompt_embeds = np.zeros_like(prompt_embeds)

            if self.xl:
                negative_pooled_prompt_embeds = np.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance:

            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompts is not None and type(prompts) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):

                max_length = prompt_embeds.shape[1]

                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="np",
                )
                uncond_input_ids = uncond_input.input_ids

                negative_embeddings = text_encoder(
                    input_ids=uncond_input_ids.astype(np.float32)
                )

                negative_text_embeddings = negative_embeddings[hidden_state_key]

                negative_prompt_embeds_list.append(negative_text_embeddings)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                if self.xl:
                    negative_pooled_prompt_embeds = negative_embeddings['pooled_outputs']

            negative_prompt_embeds = np.concatenate(negative_prompt_embeds_list, axis=-1)

        if do_classifier_free_guidance:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate(
                [negative_prompt_embeds, prompt_embeds])

            if self.xl:
                pooled_prompt_embeds = np.concatenate(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds])

        prompt_embeddings = prompt_embeds.transpose(0, 2, 1)[:, :, None, :]

        if self.xl:
            return prompt_embeddings, pooled_prompt_embeds
        else:
            return prompt_embeddings, None

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

        if self.xl:
            scaling_factor =0.13025
        else:
            scaling_factor = 0.18215

        latents = 1 / scaling_factor * latents

        dtype = self.vae_decoder.expected_inputs['z']['dtype']
        image = self.vae_decoder(z=latents.astype(dtype))["image"]
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

        init_noise = self.scheduler.init_noise_sigma

        if isinstance(init_noise, torch.Tensor):
            init_noise = init_noise.numpy()

        latents = latents * init_noise

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

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = np.array(add_time_ids).astype(dtype)
        return add_time_ids

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
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)
        height = height or self.height
        width = width or self.width

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

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
        text_embeddings, pooled_prompt_embeds = self._encode_prompt(
            prompt=prompt,
            prompt_2=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=None
        )

        # 4. Prepare XL kwargs if needed
        unet_additional_kwargs = {}

        # we add pooled prompt embeds + time_ids to unet kwargs
        if self.xl:
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size,
                                                  text_embeddings.dtype)
            if do_classifier_free_guidance:

                # TODO: This checks if the time_ids input is looking for time_ids.shape == (12,) or (2, 6)
                # Remove once model input shapes are ubiquitous
                if len(self.unet.expected_inputs['time_ids']['shape']) > 1:
                    add_time_ids = [add_time_ids]

                add_time_ids = np.concatenate([add_time_ids, add_time_ids])

            unet_additional_kwargs.update({'text_embeds': add_text_embeds.astype(np.float16),
                                           'time_ids': add_time_ids.astype(np.float16)})

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables and controlnet cond
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

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            if isinstance(latent_model_input, torch.Tensor):
                latent_model_input = latent_model_input.numpy()

            # controlnet
            if controlnet_cond:
                control_net_additional_residuals = self.run_controlnet(
                    sample=latent_model_input,
                    timestep=np.array([t, t]),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_cond,
                )
            else:
                control_net_additional_residuals = {}

            # predict the noise residual
            unet_additional_kwargs.update(control_net_additional_residuals)

            noise_pred = self.unet(
                sample=latent_model_input.astype(np.float16),
                timestep=np.array([t, t], np.float16),
                encoder_hidden_states=text_embeddings.astype(np.float16),
                **unet_additional_kwargs,
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
                    controlnet_models=None,
                    force_zeros_for_empty_prompt=True,
                    sources=None):
    """
    Initializes and returns a `CoreMLStableDiffusionPipeline` from an original
    diffusers PyTorch pipeline
        sources: 'packages' or 'compiled' forces creation of model from specified sources. sources must be in mlpackages_dir
    """

    # Ensure `scheduler_override` object is of correct type if specified
    if scheduler_override is not None:
        assert isinstance(scheduler_override, SchedulerMixin)
        logger.warning(
            "Overriding scheduler in pipeline: "
            f"Default={pytorch_pipe.scheduler}, Override={scheduler_override}")

    # Gather configured tokenizer and scheduler attributes from the original pipe
    if 'xl' in model_version:
        coreml_pipe_kwargs = {
            "tokenizer": pytorch_pipe.tokenizer,
            'tokenizer_2': pytorch_pipe.tokenizer_2,
            "scheduler": pytorch_pipe.scheduler if scheduler_override is None else scheduler_override,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            'xl': True
        }

        model_packages_to_load = ["text_encoder", "text_encoder_2", "unet", "vae_decoder"]

    else:
        coreml_pipe_kwargs = {
            "tokenizer": pytorch_pipe.tokenizer,
            "scheduler": pytorch_pipe.scheduler if scheduler_override is None else scheduler_override,
            "feature_extractor": pytorch_pipe.feature_extractor,
        }
        model_packages_to_load = ["text_encoder", "unet", "vae_decoder"]

    if getattr(pytorch_pipe, "safety_checker", None) is not None:
        model_packages_to_load.append("safety_checker")
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
        model_packages_to_load.remove("unet")
        coreml_pipe_kwargs["unet"] = _load_mlpackage(
            submodule_name="control-unet",
            mlpackages_dir=mlpackages_dir,
            model_version=model_version,
            compute_unit=compute_unit,
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
            submodule_name=model_name,
            mlpackages_dir=mlpackages_dir,
            model_version=model_version,
            compute_unit=compute_unit,
            sources=sources,
        )
        for model_name in model_packages_to_load
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

    SDP = StableDiffusionXLPipeline if 'xl' in args.model_version else StableDiffusionPipeline

    pytorch_pipe = SDP.from_pretrained(
        args.model_version,
        use_auth_token=True,
    )

    # Get Scheduler
    user_specified_scheduler = None
    if args.scheduler is not None:
        user_specified_scheduler = SCHEDULER_MAP[
            args.scheduler].from_config(pytorch_pipe.scheduler.config)

    # Get Force Zeros Config if it exists
    force_zeros_for_empty_prompt: bool = False
    if 'force_zeros_for_empty_prompt' in pytorch_pipe.config:
        force_zeros_for_empty_prompt = pytorch_pipe.config['force_zeros_for_empty_prompt']

    coreml_pipe = get_coreml_pipe(
        pytorch_pipe=pytorch_pipe,
        mlpackages_dir=args.i,
        model_version=args.model_version,
        compute_unit=args.compute_unit,
        scheduler_override=user_specified_scheduler,
        controlnet_models=args.controlnet,
        force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
        sources=args.model_sources,
    )

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
    parser.add_argument('--model-sources',
                        default=None,
                        choices=['packages', 'compiled'],
                        help='Force build from `packages` or `compiled`')

    args = parser.parse_args()
    main(args)
