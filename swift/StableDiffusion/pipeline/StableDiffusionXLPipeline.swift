// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Accelerate
import CoreGraphics
import CoreML
import Foundation
import NaturalLanguage


/// A pipeline used to generate image samples from text input using stable diffusion XL
///
/// This implementation matches:
/// [Hugging Face Diffusers XL Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py)
@available(iOS 17.0, macOS 14.0, *)
public struct StableDiffusionXLPipeline: StableDiffusionPipelineProtocol {
    
    public typealias Configuration = PipelineConfiguration
    public typealias Progress = PipelineProgress
    
    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoderXLModel?
    var textEncoder2: TextEncoderXLModel

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    var unet: Unet
    
    /// Model used to refine the image, if present
    var unetRefiner: Unet?

    /// Model used to generate final image from latent diffusion process
    var decoder: Decoder
    
    /// Model used to latent space for image2image, and soon, in-painting
    var encoder: Encoder?
    
    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    var reduceMemory: Bool = false

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - textEncoder2: Second text encoding model
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - reduceMemory: Option to enable reduced memory mode
    /// - Returns: Pipeline ready for image generation
    public init(
        textEncoder: TextEncoderXLModel?,
        textEncoder2: TextEncoderXLModel,
        unet: Unet,
        unetRefiner: Unet?,
        decoder: Decoder,
        encoder: Encoder?,
        reduceMemory: Bool = false
    ) {
        self.textEncoder = textEncoder
        self.textEncoder2 = textEncoder2
        self.unet = unet
        self.unetRefiner = unetRefiner
        self.decoder = decoder
        self.encoder = encoder
        self.reduceMemory = reduceMemory
    }

    /// Load required resources for this pipeline
    ///
    /// If reducedMemory is true this will instead call prewarmResources instead
    /// and let the pipeline lazily load resources as needed
    public func loadResources() throws {
        if reduceMemory {
            try prewarmResources()
        } else {
            try textEncoder2.loadResources()
            try unet.loadResources()
            try decoder.loadResources()

            do {
                try textEncoder?.loadResources()
            } catch {
                print("Error loading resources for textEncoder: \(error)")
            }

            // Only prewarm refiner unet on load so it's unloaded until needed
            do {
                try unetRefiner?.prewarmResources()
            } catch {
                print("Error loading resources for unetRefiner: \(error)")
            }

            do {
                try encoder?.loadResources()
            } catch {
                print("Error loading resources for vae encoder: \(error)")
            }
        }
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        textEncoder?.unloadResources()
        textEncoder2.unloadResources()
        unet.unloadResources()
        unetRefiner?.unloadResources()
        decoder.unloadResources()
        encoder?.unloadResources()
    }

    /// Prewarm resources one at a time
    public func prewarmResources() throws {
        try textEncoder2.prewarmResources()
        try unet.prewarmResources()
        try decoder.prewarmResources()

        do {
            try textEncoder?.prewarmResources()
        } catch {
            print("Error prewarming resources for textEncoder: \(error)")
        }

        do {
            try unetRefiner?.prewarmResources()
        } catch {
            print("Error prewarming resources for unetRefiner: \(error)")
        }

        do {
            try encoder?.prewarmResources()
        } catch {
            print("Error prewarming resources for vae encoder: \(error)")
        }
    }
    /// Image generation using stable diffusion
    /// - Parameters:
    ///   - configuration: Image generation configuration
    ///   - progressHandler: Callback to perform after each step, stops on receiving false response
    /// - Returns: An array of `imageCount` optional images.
    ///            The images will be nil if safety checks were performed and found the result to be un-safe
    public func generateImages(
        configuration config: Configuration,
        progressHandler: (Progress) -> Bool = { _ in true }
    ) throws -> [CGImage?] {

        // Determine input type of Unet
        // SDXL Refiner has a latentTimeIdShape of [2, 5]
        // SDXL Base has either [12] or [2, 6]
        let isRefiner = unet.latentTimeIdShape.last == 5

        // Setup geometry conditioning for base/refiner inputs
        var baseInput: ModelInputs?
        var refinerInput: ModelInputs?

        // Check if the first textEncoder is available, which is required for base models
        if textEncoder != nil {
            baseInput = try generateConditioning(using: config, forRefiner: isRefiner)
        }

        // Check if the refiner unet exists, or if the current unet is a refiner
        if unetRefiner != nil || isRefiner {
            refinerInput = try generateConditioning(using: config, forRefiner: true)
        }

        if reduceMemory {
            textEncoder?.unloadResources()
            textEncoder2.unloadResources()
        }

        /// Setup schedulers
        let scheduler: [Scheduler] = (0..<config.imageCount).map { _ in
            switch config.schedulerType {
            case .pndmScheduler: return PNDMScheduler(stepCount: config.stepCount)
            case .dpmSolverMultistepScheduler: return DPMSolverMultistepScheduler(stepCount: config.stepCount, timeStepSpacing: config.schedulerTimestepSpacing)
            }
        }

        // Generate random latent samples from specified seed
        var latents: [MLShapedArray<Float32>] = try generateLatentSamples(configuration: config, scheduler: scheduler[0])

        // Store denoised latents from scheduler to pass into decoder
        var denoisedLatents: [MLShapedArray<Float32>] = latents.map { MLShapedArray(converting: $0) }

        if reduceMemory {
            encoder?.unloadResources()
        }

        let timestepStrength: Float? = config.mode == .imageToImage ? config.strength : nil

        // Store current model
        var unetModel = unet
        var currentInput = baseInput ?? refinerInput

        var unetHiddenStates = currentInput?.hiddenStates
        var unetPooledStates = currentInput?.pooledStates
        var unetGeometryConditioning = currentInput?.geometryConditioning

        let timeSteps: [Int] = scheduler[0].calculateTimesteps(strength: timestepStrength)

        // Calculate which step to swap to refiner
        let refinerStartStep = Int(Float(timeSteps.count) * config.refinerStart)

        // De-noising loop
        for (step,t) in timeSteps.enumerated() {
            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            let latentUnetInput = latents.map {
                MLShapedArray<Float32>(concatenating: [$0, $0], alongAxis: 0)
            }

            // Switch to refiner if specified
            if let refiner = unetRefiner, step == refinerStartStep {
                unet.unloadResources()

                unetModel = refiner
                currentInput = refinerInput
                unetHiddenStates = currentInput?.hiddenStates
                unetPooledStates = currentInput?.pooledStates
                unetGeometryConditioning = currentInput?.geometryConditioning
            }

            guard let hiddenStates = unetHiddenStates,
                  let pooledStates = unetPooledStates,
                  let geometryConditioning = unetGeometryConditioning else {
                throw PipelineError.missingUnetInputs
            }

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try unetModel.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates,
                pooledStates: pooledStates,
                geometryConditioning: geometryConditioning
            )

            noise = performGuidance(noise, config.guidanceScale)

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            for i in 0..<config.imageCount {
                latents[i] = scheduler[i].step(
                    output: noise[i],
                    timeStep: t,
                    sample: latents[i]
                )
                
                denoisedLatents[i] = scheduler[i].modelOutputs.last ?? latents[i]
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

        // Unload resources
        if reduceMemory {
            unet.unloadResources()
        }
        unetRefiner?.unloadResources()


        // Decode the latent samples to images
        return try decodeToImages(denoisedLatents, configuration: config)
    }

    func encodePrompt(_ prompt: String, forRefiner: Bool = false) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>) {
        var embeds = MLShapedArray<Float32>()
        var pooled = MLShapedArray<Float32>()

        if forRefiner {
            let (embeds2, pooledValue) = try textEncoder2.encode(prompt)

            // Refiner only takes textEncoder2 embeddings
            // [1, 77, 1280]
            embeds = embeds2
            pooled = pooledValue
        } else {
            guard let encoder = textEncoder else {
                throw PipelineError.startingText2ImgWithoutTextEncoder
            }
            let (embeds1, _) = try encoder.encode(prompt)
            let (embeds2, pooledValue) = try textEncoder2.encode(prompt)

            // Base needs concatenated embeddings
            // [1, 77, 768], [1, 77, 1280] -> [1, 77, 2048]
            embeds = MLShapedArray<Float32>(
                concatenating: [embeds1, embeds2],
                alongAxis: 2
            )
            pooled = pooledValue
        }

        return (embeds, pooled)
    }

    func generateConditioning(using config: Configuration, forRefiner: Bool = false) throws -> ModelInputs {
        // Encode the input prompt and negative prompt
        let (promptEmbedding, pooled) = try encodePrompt(config.prompt, forRefiner: forRefiner)
        let (negativePromptEmbedding, negativePooled) = try encodePrompt(config.negativePrompt, forRefiner: forRefiner)

        // Convert to Unet hidden state representation
        // Concatenate the prompt and negative prompt embeddings
        let hiddenStates = toHiddenStates(MLShapedArray(concatenating: [negativePromptEmbedding, promptEmbedding], alongAxis: 0))
        let pooledStates = MLShapedArray(concatenating: [negativePooled, pooled], alongAxis: 0)

        // Inline helper functions for geometry creation
        func refinerGeometry() -> MLShapedArray<Float32> {
            let negativeGeometry = MLShapedArray<Float32>(
                scalars: [
                    config.originalSize, config.originalSize,
                    config.cropsCoordsTopLeft, config.cropsCoordsTopLeft,
                    config.negativeAestheticScore
                ],
                shape: [1, 5]
            )
            let positiveGeometry = MLShapedArray<Float32>(
                scalars: [
                    config.originalSize, config.originalSize,
                    config.cropsCoordsTopLeft, config.cropsCoordsTopLeft,
                    config.aestheticScore
                ],
                shape: [1, 5]
            )
            return MLShapedArray<Float32>(concatenating: [negativeGeometry, positiveGeometry], alongAxis: 0)
        }

        func baseGeometry() -> MLShapedArray<Float32> {
            let geometry = MLShapedArray<Float32>(
                scalars: [
                    config.originalSize, config.originalSize,
                    config.cropsCoordsTopLeft, config.cropsCoordsTopLeft,
                    config.targetSize, config.targetSize
                ],
                // TODO: This checks if the time_ids input is looking for [12] or [2, 6]
                // Remove once model input shapes are ubiquitous
                shape: unet.latentTimeIdShape.count > 1 ? [1, 6] : [6]
            )
            return MLShapedArray<Float32>(concatenating: [geometry, geometry], alongAxis: 0)
        }

        let geometry = forRefiner ? refinerGeometry() : baseGeometry()

        return ModelInputs(hiddenStates: hiddenStates, pooledStates: pooledStates, geometryConditioning: geometry)
    }

    func generateLatentSamples(configuration config: Configuration, scheduler: Scheduler) throws -> [MLShapedArray<Float32>] {
        var sampleShape = unet.latentSampleShape
        sampleShape[0] = 1
        
        let stdev = scheduler.initNoiseSigma
        var random = randomSource(from: config.rngType, seed: config.seed)
        let samples = (0..<config.imageCount).map { _ in
            MLShapedArray<Float32>(
                converting: random.normalShapedArray(sampleShape, mean: 0.0, stdev: Double(stdev)))
        }
        if let image = config.startingImage, config.mode == .imageToImage {
            guard let encoder else {
                throw PipelineError.startingImageProvidedWithoutEncoder
            }
            let latent = try encoder.encode(image, scaleFactor: config.encoderScaleFactor, random: &random)
            return scheduler.addNoise(originalSample: latent, noise: samples, strength: config.strength)
        }
        return samples
    }

    public func decodeToImages(_ latents: [MLShapedArray<Float32>], configuration config: Configuration) throws -> [CGImage?] {
        defer {
            if reduceMemory {
                decoder.unloadResources()
            }
        }
        
        return try decoder.decode(latents, scaleFactor: config.decoderScaleFactor)
    }

    struct ModelInputs {
        var hiddenStates: MLShapedArray<Float32>
        var pooledStates: MLShapedArray<Float32>
        var geometryConditioning: MLShapedArray<Float32>
    }
}
