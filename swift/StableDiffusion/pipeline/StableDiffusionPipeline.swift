// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Accelerate
import CoreGraphics
import CoreML
import Foundation
import NaturalLanguage

/// Schedulers compatible with StableDiffusionPipeline
public enum StableDiffusionScheduler {
    /// Scheduler that uses a pseudo-linear multi-step (PLMS) method
    case pndmScheduler
    /// Scheduler that uses a second order DPM-Solver++ algorithm
    case dpmSolverMultistepScheduler
}

/// RNG compatible with StableDiffusionPipeline
public enum StableDiffusionRNG {
    /// RNG that matches numpy implementation
    case numpyRNG
    /// RNG that matches PyTorch CPU implementation.
    case torchRNG
    /// RNG that matches PyTorch CUDA implementation.
    case nvidiaRNG
}

public enum PipelineError: String, Swift.Error {
    case missingUnetInputs
    case startingImageProvidedWithoutEncoder
    case startingText2ImgWithoutTextEncoder
    case unsupportedOSVersion
}

@available(iOS 16.2, macOS 13.1, *)
public protocol StableDiffusionPipelineProtocol: ResourceManaging {
    var canSafetyCheck: Bool { get }

    func generateImages(
        configuration config: PipelineConfiguration,
        progressHandler: (PipelineProgress) -> Bool
    ) throws -> [CGImage?]

    func decodeToImages(
        _ latents: [MLShapedArray<Float32>],
        configuration config: PipelineConfiguration
    ) throws -> [CGImage?]
}

@available(iOS 16.2, macOS 13.1, *)
public extension StableDiffusionPipelineProtocol {
    var canSafetyCheck: Bool { false }
}

/// A pipeline used to generate image samples from text input using stable diffusion
///
/// This implementation matches:
/// [Hugging Face Diffusers Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
@available(iOS 16.2, macOS 13.1, *)
public struct StableDiffusionPipeline: StableDiffusionPipelineProtocol {

    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoderModel

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    var unet: Unet

    /// Model used to generate final image from latent diffusion process
    var decoder: Decoder
    
    /// Model used to latent space for image2image, and soon, in-painting
    var encoder: Encoder?

    /// Optional model for checking safety of generated image
    var safetyChecker: SafetyChecker? = nil
    
    /// Optional model used before Unet to control generated images by additonal inputs
    var controlNet: ControlNet? = nil

    /// Reports whether this pipeline can perform safety checks
    public var canSafetyCheck: Bool {
        safetyChecker != nil
    }

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    var reduceMemory: Bool = false

    /// Option to use system multilingual NLContextualEmbedding as encoder
    var useMultilingualTextEncoder: Bool = false

    /// Optional natural language script to use for the text encoder.
    var script: Script? = nil

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - controlNet: Optional model to control generated images by additonal inputs
    ///   - safetyChecker: Optional model for checking safety of generated images
    ///   - reduceMemory: Option to enable reduced memory mode
    /// - Returns: Pipeline ready for image generation
    public init(
        textEncoder: TextEncoderModel,
        unet: Unet,
        decoder: Decoder,
        encoder: Encoder?,
        controlNet: ControlNet? = nil,
        safetyChecker: SafetyChecker? = nil,
        reduceMemory: Bool = false
    ) {
        self.textEncoder = textEncoder
        self.unet = unet
        self.decoder = decoder
        self.encoder = encoder
        self.controlNet = controlNet
        self.safetyChecker = safetyChecker
        self.reduceMemory = reduceMemory
    }

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - controlNet: Optional model to control generated images by additonal inputs
    ///   - safetyChecker: Optional model for checking safety of generated images
    ///   - reduceMemory: Option to enable reduced memory mode
    ///   - useMultilingualTextEncoder: Option to use system multilingual NLContextualEmbedding as encoder
    ///   - script: Optional natural language script to use for the text encoder.
    /// - Returns: Pipeline ready for image generation
    @available(iOS 17.0, macOS 14.0, *)
    public init(
        textEncoder: TextEncoderModel,
        unet: Unet,
        decoder: Decoder,
        encoder: Encoder?,
        controlNet: ControlNet? = nil,
        safetyChecker: SafetyChecker? = nil,
        reduceMemory: Bool = false,
        useMultilingualTextEncoder: Bool = false,
        script: Script? = nil
    ) {
        self.textEncoder = textEncoder
        self.unet = unet
        self.decoder = decoder
        self.encoder = encoder
        self.controlNet = controlNet
        self.safetyChecker = safetyChecker
        self.reduceMemory = reduceMemory
        self.useMultilingualTextEncoder = useMultilingualTextEncoder
        self.script = script
    }

    /// Load required resources for this pipeline
    ///
    /// If reducedMemory is true this will instead call prewarmResources instead
    /// and let the pipeline lazily load resources as needed
    public func loadResources() throws {
        if reduceMemory {
            try prewarmResources()
        } else {
            try unet.loadResources()
            try textEncoder.loadResources()
            try decoder.loadResources()
            try encoder?.loadResources()
            try controlNet?.loadResources()
            try safetyChecker?.loadResources()
        }
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        textEncoder.unloadResources()
        unet.unloadResources()
        decoder.unloadResources()
        encoder?.unloadResources()
        controlNet?.unloadResources()
        safetyChecker?.unloadResources()
    }

    // Prewarm resources one at a time
    public func prewarmResources() throws {
        try textEncoder.prewarmResources()
        try unet.prewarmResources()
        try decoder.prewarmResources()
        try encoder?.prewarmResources()
        try controlNet?.prewarmResources()
        try safetyChecker?.prewarmResources()
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

        // Encode the input prompt and negative prompt
        let promptEmbedding = try textEncoder.encode(config.prompt)
        let negativePromptEmbedding = try textEncoder.encode(config.negativePrompt)

        if reduceMemory {
            textEncoder.unloadResources()
        }

        // Convert to Unet hidden state representation
        // Concatenate the prompt and negative prompt embeddings
        let concatEmbedding = MLShapedArray<Float32>(
            concatenating: [negativePromptEmbedding, promptEmbedding],
            alongAxis: 0
        )

        let hiddenStates = useMultilingualTextEncoder ? concatEmbedding : toHiddenStates(concatEmbedding)

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
        
        // Convert cgImage for ControlNet into MLShapedArray
        let controlNetConds = try config.controlNetInputs.map { cgImage in
            let shapedArray = try cgImage.planarRGBShapedArray(minValue: 0.0, maxValue: 1.0)
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
                additionalResiduals: additionalResiduals
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

        if reduceMemory {
            controlNet?.unloadResources()
            unet.unloadResources()
        }

        // Decode the latent samples to images
        return try decodeToImages(denoisedLatents, configuration: config)
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
        let images = try decoder.decode(latents, scaleFactor: config.decoderScaleFactor)
        if reduceMemory {
            decoder.unloadResources()
        }

        // If safety is disabled return what was decoded
        if config.disableSafety {
            return images
        }

        // If there is no safety checker return what was decoded
        guard let safetyChecker = safetyChecker else {
            return images
        }

        // Otherwise change images which are not safe to nil
        let safeImages = try images.map { image in
            try safetyChecker.isSafe(image) ? image : nil
        }

        if reduceMemory {
            safetyChecker.unloadResources()
        }

        return safeImages
    }

}

/// Sampling progress details
@available(iOS 16.2, macOS 13.1, *)
public struct PipelineProgress {
    public let pipeline: StableDiffusionPipelineProtocol
    public let prompt: String
    public let step: Int
    public let stepCount: Int
    public let currentLatentSamples: [MLShapedArray<Float32>]
    public let configuration: PipelineConfiguration
    public var isSafetyEnabled: Bool {
        pipeline.canSafetyCheck && !configuration.disableSafety
    }
    public var currentImages: [CGImage?] {
        try! pipeline.decodeToImages(currentLatentSamples, configuration: configuration)
    }
}

@available(iOS 16.2, macOS 13.1, *)
public extension StableDiffusionPipeline {
    /// Sampling progress details
    typealias Progress = PipelineProgress
}

// Helper functions

@available(iOS 16.2, macOS 13.1, *)
extension StableDiffusionPipelineProtocol {
    internal func randomSource(from rng: StableDiffusionRNG, seed: UInt32) -> RandomSource {
        switch rng {
        case .numpyRNG:
            return NumPyRandomSource(seed: seed)
        case .torchRNG:
            return TorchRandomSource(seed: seed)
        case .nvidiaRNG:
            return NvRandomSource(seed: seed)
        }
    }

    func toHiddenStates(_ embedding: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        // Unoptimized manual transpose [0, 2, None, 1]
        // e.g. From [2, 77, 768] to [2, 768, 1, 77]
        let fromShape = embedding.shape
        let stateShape = [fromShape[0],fromShape[2], 1, fromShape[1]]
        var states = MLShapedArray<Float32>(repeating: 0.0, shape: stateShape)
        for i0 in 0..<fromShape[0] {
            for i1 in 0..<fromShape[1] {
                for i2 in 0..<fromShape[2] {
                    states[scalarAt:i0,i2,0,i1] = embedding[scalarAt:i0, i1, i2]
                }
            }
        }
        return states
    }

    func performGuidance(_ noise: [MLShapedArray<Float32>], _ guidanceScale: Float) -> [MLShapedArray<Float32>] {
        noise.map { performGuidance($0, guidanceScale) }
    }

    func performGuidance(_ noise: MLShapedArray<Float32>, _ guidanceScale: Float) -> MLShapedArray<Float32> {
        var shape = noise.shape
        shape[0] = 1
        return MLShapedArray<Float>(unsafeUninitializedShape: shape) { result, _ in
            noise.withUnsafeShapedBufferPointer { scalars, _, strides in
                for i in 0 ..< result.count {
                    // unconditioned + guidance*(text - unconditioned)
                    result.initializeElement(
                        at: i,
                        to: scalars[i] + guidanceScale * (scalars[strides[0] + i] - scalars[i])
                    )
                }
            }
        }
    }
}
