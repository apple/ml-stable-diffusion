// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Accelerate
import CoreGraphics
import CoreML
import Foundation
import NaturalLanguage



/// A pipeline used to generate image samples from text input using stable diffusion xl
///
/// This implementation matches:
/// [Hugging Face Diffusers Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py)
@available(iOS 16.2, macOS 13.1, *)
public class StableDiffusionPipelineXL: StableDiffusionPipeline {

    /// Keep track of whether the model requires an aesthetic score in it's added embedding
    /// This is only required for the SDXL refiner
    var requiresAestheticsScore: Bool = false

    /// Image generation using stable diffusion
    /// - Parameters:
    ///   - configuration: Image generation configuration
    ///   - progressHandler: Callback to perform after each step, stops on receiving false response
    /// - Returns: An array of `imageCount` optional images.
    ///            The images will be nil if safety checks were performed and found the result to be un-safe
    public override func generateImages(
        configuration config: Configuration,
        progressHandler: (Progress) -> Bool = { _ in true }
    ) throws -> [CGImage?] {
        // Keep track of whether the model requires an aesthetic score
        var requiresAestheticsScore = false

        // Try to encode the input prompt and negative prompt with textEncoder if it's not nil
        var promptEmbedding1Hidden: MLShapedArray<Float32>?
        var negativePromptEmbedding1Hidden: MLShapedArray<Float32>?
        if let textEncoder = textEncoder {
            (promptEmbedding1Hidden, _) = try textEncoder.encode(config.prompt)
            (negativePromptEmbedding1Hidden, _) = try textEncoder.encode(config.negativePrompt)

            // Unload resources to reduce memory if flag is set
            if reduceMemory {
                textEncoder.unloadResources()
            }
        } else {
            // text_encoder is nil, since this is an XL pipeline, that means this is a refiner model
            requiresAestheticsScore = true
        }

        // Try to encode the input prompt and negative prompt with textEncoder2 if it's not nil
        var promptEmbedding2Hidden: MLShapedArray<Float32>?
        var promptEmbedding2Pooled: MLShapedArray<Float32>?
        var negativePromptEmbedding2Hidden: MLShapedArray<Float32>?
        var negativePromptEmbedding2Pooled: MLShapedArray<Float32>?
        if let textEncoder2 = textEncoder2 {
            (promptEmbedding2Hidden, promptEmbedding2Pooled) = try textEncoder2.encode(config.prompt)
            (negativePromptEmbedding2Hidden, negativePromptEmbedding2Pooled) = try textEncoder2.encode(config.negativePrompt)

            // Unload resources to reduce memory if flag is set
            if reduceMemory {
                textEncoder2.unloadResources()
            }
        }

        var embeddingsToConcatenate = [MLShapedArray<Float32>]()

        if let promptEmbedding1Hidden = promptEmbedding1Hidden {
            embeddingsToConcatenate.append(promptEmbedding1Hidden)
        }

        if let promptEmbedding2Hidden = promptEmbedding2Hidden {
            embeddingsToConcatenate.append(promptEmbedding2Hidden)
        }

        var negativeEmbeddingsToConcatenate = [MLShapedArray<Float32>]()

        if let negativePromptEmbedding1Hidden = negativePromptEmbedding1Hidden {
            negativeEmbeddingsToConcatenate.append(negativePromptEmbedding1Hidden)
        }

        if let negativePromptEmbedding2Hidden = negativePromptEmbedding2Hidden {
            negativeEmbeddingsToConcatenate.append(negativePromptEmbedding2Hidden)
        }

        let concatNegativeEmbedding = MLShapedArray<Float32>(
            concatenating: negativeEmbeddingsToConcatenate,
            alongAxis: -1
        )

        let concatPromptEmbedding = MLShapedArray<Float32>(
            concatenating: embeddingsToConcatenate,
            alongAxis: -1
        )

        let encoderHiddenStates = MLShapedArray<Float32>(
            concatenating: [concatNegativeEmbedding, concatPromptEmbedding],
            alongAxis: 0
        )

        let hiddenStates = useMultilingualTextEncoder ? encoderHiddenStates : toHiddenStates(encoderHiddenStates)

        guard let negativeAddEmbed = negativePromptEmbedding2Pooled,
              let addEmbed = promptEmbedding2Pooled else { return [] }

        let addEmbeds = MLShapedArray<Float32>(
            concatenating: [negativeAddEmbed, addEmbed],
            alongAxis: 0
        )

        var timeIds = MLShapedArray<Float32>(scalars: [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0,
                                                       1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0], shape: [2, 6])
        if (requiresAestheticsScore) {
            timeIds = MLShapedArray<Float32>(scalars: [1024.0, 1024.0, 0.0, 0.0, 2.5,
                                                       1024.0, 1024.0, 0.0, 0.0, 6], shape: [2, 5])
        }


        /// Setup schedulers
        let scheduler: [Scheduler] = (0..<config.imageCount).map { _ in
            switch config.schedulerType {
            case .pndmScheduler: return PNDMScheduler(stepCount: config.stepCount)
            case .dpmSolverMultistepScheduler: return DPMSolverMultistepScheduler(stepCount: config.stepCount)
            }
        }

        // Generate random latent samples from specified seed
        var latents: [MLShapedArray<Float32>] = try generateLatentSamples(configuration: config, scheduler: scheduler[0])
        if reduceMemory {
            encoder?.unloadResources()
        }
        let timestepStrength: Float? = config.mode == .imageToImage ? config.strength : nil

        // Convert cgImage for ControlNet into MLShapedArray
        let controlNetConds = try config.controlNetInputs.map { cgImage in
            let shapedArray = try cgImage.plannerRGBShapedArray(minValue: 0.0, maxValue: 1.0)
            return MLShapedArray(
                concatenating: [shapedArray, shapedArray],
                alongAxis: 0
            )
        }

        // De-noising loop
        let timeSteps: [Int] = scheduler[0].calculateTimesteps(strength: timestepStrength)
        for (step,t) in timeSteps.enumerated() {

            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            let latentUnetInput = latents.map {
                MLShapedArray<Float32>(concatenating: [$0, $0], alongAxis: 0)
            }

            // Before Unet, execute controlNet and add the output into Unet inputs
            let additionalResiduals = try controlNet?.execute(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates,
                images: controlNetConds
            )

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try unet.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates,
                textEmbeds: addEmbeds,
                timeIds: timeIds,
                additionalResiduals: additionalResiduals
            )

            noise = performGuidance(noise, config.guidanceScale)

            // Retreive denoised latents from scheduler to pass into progress report
            var denoisedLatents: [MLShapedArray<Float32>] = []

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            for i in 0..<config.imageCount {
                latents[i] = scheduler[i].step(
                    output: noise[i],
                    timeStep: t,
                    sample: latents[i]
                )

                if let denoisedLatent = scheduler[i].modelOutputs.last {
                    denoisedLatents.append(denoisedLatent)
                }
            }

            let currentLatentSamples = config.useDenoisedIntermediates ? denoisedLatents : latents

            // Report progress
            let progress = Progress(
                pipeline: self,
                prompt: config.prompt,
                step: step,
                stepCount: timeSteps.count,
                currentLatentSamples: currentLatentSamples,
                configuration: config
            )
            if !progressHandler(progress) {
                // Stop if requested by handler
                return []
            }
        }

        if reduceMemory {
            controlNet?.unloadResources()
            unet.unloadResources()
        }

        // Decode the latent samples to images
        return try decodeToImages(latents, configuration: config)
    }
}
