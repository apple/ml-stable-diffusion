// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.

import Accelerate
import CoreGraphics
import CoreImage
import CoreML
import Foundation

@available(iOS 17.0, macOS 14.0, *)
public struct StableDiffusion3Pipeline: StableDiffusionPipelineProtocol {
    public typealias Configuration = PipelineConfiguration
    public typealias Progress = PipelineProgress

    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoderXLModel
    var textEncoder2: TextEncoderXLModel
    var textEncoderT5: TextEncoderT5Model?

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    var mmdit: MultiModalDiffusionTransformer

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
    ///   - mmdit: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - reduceMemory: Option to enable reduced memory mode
    /// - Returns: Pipeline ready for image generation
    public init(
        textEncoder: TextEncoderXLModel,
        textEncoder2: TextEncoderXLModel,
        textEncoderT5: TextEncoderT5?,
        mmdit: MultiModalDiffusionTransformer,
        decoder: Decoder,
        encoder: Encoder?,
        reduceMemory: Bool = false
    ) {
        self.textEncoder = textEncoder
        self.textEncoder2 = textEncoder2
        self.textEncoderT5 = textEncoderT5
        self.mmdit = mmdit
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
            try textEncoder.loadResources()
            try textEncoder2.loadResources()
            try textEncoderT5?.loadResources()
            try mmdit.loadResources()
            try decoder.loadResources()

            do {
                try encoder?.loadResources()
            } catch {
                print("Error loading resources for vae encoder: \(error)")
            }
        }
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        textEncoder.unloadResources()
        textEncoder2.unloadResources()
        textEncoderT5?.unloadResources()
        mmdit.unloadResources()
        decoder.unloadResources()
        encoder?.unloadResources()
    }

    /// Prewarm resources one at a time
    public func prewarmResources() throws {
        try textEncoder.prewarmResources()
        try textEncoder2.prewarmResources()
        try textEncoderT5?.prewarmResources()
        try mmdit.prewarmResources()
        try decoder.prewarmResources()

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
        // Setup geometry conditioning for base/refiner inputs
        let sd3Input: ModelInputs = try generateConditioning(using: config)

        if reduceMemory {
            textEncoder.unloadResources()
            textEncoder2.unloadResources()
            textEncoderT5?.unloadResources()
        }

        // Setup schedulers
        let scheduler: [DiscreteFlowScheduler] = (0..<config.imageCount).map { _ in
            DiscreteFlowScheduler(stepCount: config.stepCount, timeStepShift: config.schedulerTimestepShift)
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
        let mmditModel = mmdit

        let mmditHiddenStates = sd3Input.hiddenStates
        let mmditPooledStates = sd3Input.pooledStates

        let timeSteps: [Float] = scheduler[0].calculateTimestepsFromSigmas(strength: timestepStrength)

        // De-noising loop
        for (step, t) in timeSteps.enumerated() {
            // Expand the latents for classifier-free guidance
            // and input to the MMDiT noise prediction model
            let latentUnetInput = latents.map {
                MLShapedArray<Float32>(concatenating: [$0, $0], alongAxis: 0)
            }

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try mmditModel.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                tokenLevelTextEmbeddings: mmditHiddenStates,
                pooledTextEmbeddings: mmditPooledStates
            )

            noise = performGuidance(noise, config.guidanceScale)

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            for i in 0..<config.imageCount {
                latents[i] = scheduler[i].step(
                    output: noise[i],
                    timeStep: scheduler[i].timeSteps[step], // TODO: allow float timesteps in scheduler step protocol
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
            mmdit.unloadResources()
        }

        // Decode the latent samples to images
        return try decodeToImages(denoisedLatents, configuration: config)
    }

    func encodePrompt(_ prompt: String) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>) {
        var embeds = MLShapedArray<Float32>()
        var pooled = MLShapedArray<Float32>()

        let (embeds1, pooledValue1) = try textEncoder.encode(prompt)
        let (embeds2, pooledValue2) = try textEncoder2.encode(prompt)
        var embedsT5 = try textEncoderT5?.encode(prompt).encoderHiddenStates ?? MLShapedArray<Float32>(repeating: 0, shape: [1, 4096, 1, 77])

        // Truncate T5
        embedsT5 = truncatedT5Embeds(embedsT5)

        let padding1 = MLShapedArray<Float32>(repeating: 0, shape: [1, 77, 2048])

        // Base needs concatenated embeddings
        // [1, 77, 768], [1, 77, 1280], [1, 77, 2048] -> [1, 77, 4096]
        embeds = MLShapedArray<Float32>(
            concatenating: [embeds1, embeds2, padding1],
            alongAxis: 2
        )

        // [1, 77, 4096] -> [1, 4096, 1 77]
        embeds = toHiddenStates(embeds)

        // [1, 4096, 1 77], [1, 4096, 1, 77] -> [1, 4096, 1, 154]
        embeds = MLShapedArray<Float32>(
            concatenating: [embeds, embedsT5],
            alongAxis: 3
        )

        // [1, 768], [1, 1280] -> [1, 2048]
        pooled = MLShapedArray<Float32>(
            concatenating: [pooledValue1, pooledValue2],
            alongAxis: 1
        )

        return (embeds, pooled)
    }

    func generateConditioning(using config: Configuration) throws -> ModelInputs {
        // Encode the input prompt and negative prompt
        let (promptEmbedding, pooled) = try encodePrompt(config.prompt)
        let (negativePromptEmbedding, negativePooled) = try encodePrompt(config.negativePrompt)

        // Convert to Unet hidden state representation
        // Concatenate the prompt and negative prompt embeddings
        let hiddenStates = MLShapedArray(concatenating: [promptEmbedding, negativePromptEmbedding], alongAxis: 0)
        let pooledScalars = MLShapedArray(concatenating: [pooled, negativePooled], alongAxis: 0)

        let pooledStates = MLShapedArray<Float32>(
            scalars: pooledScalars.scalars,
            shape: [2, 2048, 1, 1]
        )

        return ModelInputs(hiddenStates: hiddenStates, pooledStates: pooledStates)
    }

    func generateLatentSamples(configuration config: Configuration, scheduler: Scheduler) throws -> [MLShapedArray<Float32>] {
        var sampleShape = mmdit.latentImageEmbeddingsShape
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

    func performGuidance(_ noise: [MLShapedArray<Float32>], _ guidanceScale: Float) -> [MLShapedArray<Float32>] {
        noise.map { performGuidance($0, guidanceScale) }
    }

    func performGuidance(_ noise: MLShapedArray<Float32>, _ guidanceScale: Float) -> MLShapedArray<Float32> {
        var shape = noise.shape
        shape[0] = 1
        return MLShapedArray<Float>(unsafeUninitializedShape: shape) { result, _ in
            noise.withUnsafeShapedBufferPointer { scalars, _, strides in
                for i in 0..<result.count {
                    // unconditioned + guidance*(text - unconditioned)
                    let text = scalars[i]
                    let negText = scalars[strides[0] + i]
                    let guidance = negText + guidanceScale * (text - negText)
                    result.initializeElement(
                        at: i,
                        to: guidance
                    )
                }
            }
        }
    }

    public func decodeToImages(_ latents: [MLShapedArray<Float32>], configuration config: Configuration) throws -> [CGImage?] {
        defer {
            if reduceMemory {
                decoder.unloadResources()
            }
        }

        return try decoder.decode(latents, scaleFactor: config.decoderScaleFactor, shiftFactor: config.decoderShiftFactor)

        // TODO: use latent rgb factors with blur for preview images
        // This will require a method to decode with either the vae or the rgb factors depending on config
        // return try decodePreviewImage(latents, scaleFactor: config.decoderScaleFactor)
    }

    /// Shape 16 x 3
    let rgbFactors: [[Float]] = [
        [-0.0645,  0.0177,  0.1052], [ 0.0028,  0.0312,  0.0650],
        [ 0.1848,  0.0762,  0.0360], [ 0.0944,  0.0360,  0.0889],
        [ 0.0897,  0.0506, -0.0364], [-0.0020,  0.1203,  0.0284],
        [ 0.0855,  0.0118,  0.0283], [-0.0539,  0.0658,  0.1047],
        [-0.0057,  0.0116,  0.0700], [-0.0412,  0.0281, -0.0039],
        [ 0.1106,  0.1171,  0.1220], [-0.0248,  0.0682, -0.0481],
        [ 0.0815,  0.0846,  0.1207], [-0.0120, -0.0055, -0.0867],
        [-0.0749, -0.0634, -0.0456], [-0.1418, -0.1457, -0.1259]
    ]

    public func decodePreviewImage(
        _ latents: [MLShapedArray<Float32>],
        scaleFactor: Float32
    ) throws -> [CGImage] {
        let height = 64
        let width = 64
        let channels = 16
        let outputChannels = 3

        // Ensure there is a first element in latents and extract its scalars
        guard let latentScalars = latents.first?.scalars else {
            throw NSError(domain: "DecodeError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid latent array"])
        }

        // The latentScalars is a flat array, we need to reshape and multiply
        var reshapedLatent = [Float32](repeating: 0, count: height * width * channels)

        // We reorder the indices manually to switch from [channels, height, width] to [height, width, channels]
        for h in 0..<height {
            for w in 0..<width {
                for c in 0..<channels {
                    let oldIndex = c * height * width + h * width + w
                    let newIndex = h * width * channels + w * channels + c
                    reshapedLatent[newIndex] = latentScalars[oldIndex] // 1.5305 + 0.0609
                }
            }
        }

        // Prepare to hold the result of the multiplication
        var imageArray = [Float32](repeating: 0, count: height * width * outputChannels)

        // Perform matrix multiplication using Accelerate
        vDSP_mmul(reshapedLatent, 1,
                  rgbFactors.flatMap { $0 }, 1,
                  &imageArray, 1,
                  vDSP_Length(height * width),  // number of rows in output
                  vDSP_Length(outputChannels),  // number of columns in output
                  vDSP_Length(channels))        // common dimension

        // Convert imageArray into a CGImage
        let latentImage = imageArray.toCGImage(width: width, height: height)

        // Apply a Gaussian blur to the preview image to reduce pixeled look
        let ciImage = CIImage(cgImage: latentImage!)
        let blurFilter = CIFilter(name: "CIGaussianBlur")!
        blurFilter.setValue(ciImage, forKey: kCIInputImageKey)
        blurFilter.setValue(4.0, forKey: kCIInputRadiusKey)

        let context = CIContext()
        guard let outputImage = blurFilter.outputImage,
              let cgBlurredPreview = context.createCGImage(outputImage, from: ciImage.extent)
        else {
            throw PipelineError.errorCreatingPreview
        }

        return [cgBlurredPreview]
    }

    struct ModelInputs {
        var hiddenStates: MLShapedArray<Float32>
        var pooledStates: MLShapedArray<Float32>
    }

    /// Helper function to truncate the T5 embeddings
    func truncatedT5Embeds(_ embedding: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        // Unoptimized manual truncation
        // e.g. From [1, 4096, 1, 128] to [1, 4096, 1, 77]
        let fromShape = embedding.shape
        let stateShape = [fromShape[0], fromShape[1], fromShape[2], 77]
        var states = MLShapedArray<Float32>(repeating: 0.0, shape: stateShape)
        for i0 in 0..<fromShape[0] {
            for i1 in 0..<fromShape[1] {
                for i2 in 0..<fromShape[2] {
                    for i3 in 0..<stateShape[3] {
                        states[scalarAt: i0, i1, i2, i3] = embedding[scalarAt: i0, i1, i2, i3]
                    }
                }
            }
        }
        return states
    }
}

extension Array where Element == Float32 {
    func toCGImage(width: Int, height: Int) -> CGImage? {
        // Define color space and bitmap info
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue

        // Calculate bytes per pixel and bytes per row
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel

        // Allocate memory for the pixel data
        var data = [UInt8](repeating: 0, count: height * bytesPerRow)

        // Fill the data array with pixel data
        for h in 0..<height {
            for w in 0..<width {
                let pixelIndex = h * width + w
                let dataIndex = h * bytesPerRow + w * bytesPerPixel
                let pixelBase = pixelIndex * 3 // Base index for R, G, B values in the source array

                // Ensure your source array has enough data
                if (pixelBase + 3) < self.count {
                    let redValue = (self[pixelBase] + 1) / 2 * 255
                    let bluValue = (self[pixelBase + 1] + 1) / 2 * 255
                    let grnValue = (self[pixelBase + 2] + 1) / 2 * 255
                    data[dataIndex] = UInt8(clamp(value: redValue, lower: 0, upper: 255))     // Red
                    data[dataIndex + 1] = UInt8(clamp(value: bluValue, lower: 0, upper: 255)) // Green
                    data[dataIndex + 2] = UInt8(clamp(value: grnValue, lower: 0, upper: 255)) // Blue
                    data[dataIndex + 3] = 255 // Alpha
                }
            }
        }

        // Create the context
        guard let context = CGContext(data: &data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else {
            print("Failed to create CGContext.")
            return nil
        }

        // Create a CGImage from context
        guard let smallImage = context.makeImage() else {
            return nil
        }

        // Define the upscaled dimensions
        let scaledWidth = width * 8
        let scaledHeight = height * 8

        // Create a new context with scaled dimensions
        guard let largeContext = CGContext(data: nil, width: scaledWidth, height: scaledHeight, bitsPerComponent: 8, bytesPerRow: scaledWidth * 4, space: colorSpace, bitmapInfo: bitmapInfo) else {
            return nil
        }

        // Draw the small image into the large context
        largeContext.interpolationQuality = .high
        largeContext.draw(smallImage, in: CGRect(x: 0, y: 0, width: scaledWidth, height: scaledHeight))

        // Convert the upscaled context to a CGImage
        return largeContext.makeImage()
    }

    /// Helper function to clamp the values within the specified range
    private func clamp(value: Float32, lower: UInt8, upper: UInt8) -> UInt8 {
        return UInt8(Swift.max(Float32(lower), Swift.min(value, Float32(upper))))
    }
}
