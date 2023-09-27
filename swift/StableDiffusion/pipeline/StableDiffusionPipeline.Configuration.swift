// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreGraphics

/// Type of processing that will be performed to generate an image
public enum PipelineMode {
    case textToImage
    case imageToImage
    // case inPainting
}

/// Image generation configuration
public struct PipelineConfiguration: Hashable {
    
    /// Text prompt to guide sampling
    public var prompt: String
    /// Negative text prompt to guide sampling
    public var negativePrompt: String = ""
    /// Starting image for image2image or in-painting
    public var startingImage: CGImage? = nil
    /// Fraction of inference steps to be used in `.imageToImage` pipeline mode
    /// Must be between 0 and 1
    /// Higher values will result in greater transformation of the `startingImage`
    public var strength: Float = 1.0
    /// Fraction of inference steps to at which to start using the refiner unet if present in `textToImage` mode
    /// Must be between 0 and 1
    /// Higher values will result in fewer refiner steps
    public var refinerStart: Float = 0.8
    /// Number of images to generate
    public var imageCount: Int = 1
    /// Number of inference steps to perform
    public var stepCount: Int = 50
    /// Random seed which to start generation
    public var seed: UInt32 = 0
    /// Controls the influence of the text prompt on sampling process (0=random images)
    public var guidanceScale: Float = 7.5
    /// List of Images for available ControlNet Models
    public var controlNetInputs: [CGImage] = []
    /// Safety checks are only performed if `self.canSafetyCheck && !disableSafety`
    public var disableSafety: Bool = false
    /// Enables progress updates to decode `currentImages` from denoised latent images for better previews
    public var useDenoisedIntermediates: Bool = false
    /// The type of Scheduler to use.
    public var schedulerType: StableDiffusionScheduler = .pndmScheduler
    /// The spacing to use for scheduler sigmas and time steps. Only supported when using `.dpmppScheduler`.
    public var schedulerTimestepSpacing: TimeStepSpacing = .linspace
    /// The type of RNG to use
    public var rngType: StableDiffusionRNG = .numpyRNG
    /// Scale factor to use on the latent after encoding
    public var encoderScaleFactor: Float32 = 0.18215
    /// Scale factor to use on the latent before decoding
    public var decoderScaleFactor: Float32 = 0.18215
    /// If `originalSize` is not the same as `targetSize` the image will appear to be down- or upsampled.
    /// Part of SDXL’s micro-conditioning as explained in section 2.2 of https://huggingface.co/papers/2307.01952.
    public var originalSize: Float32 = 1024
    /// `cropsCoordsTopLeft` can be used to generate an image that appears to be “cropped” from the position `cropsCoordsTopLeft` downwards.
    /// Favorable, well-centered images are usually achieved by setting `cropsCoordsTopLeft` to (0, 0).
    public var cropsCoordsTopLeft: Float32 = 0
    /// For most cases, `target_size` should be set to the desired height and width of the generated image.
    public var targetSize: Float32 = 1024
    /// Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
    public var aestheticScore: Float32 = 6
    /// Can be used to simulate an aesthetic score of the generated image by influencing the negative text condition.
    public var negativeAestheticScore: Float32 = 2.5

    /// Given the configuration, what mode will be used for generation
    public var mode: PipelineMode {
        guard startingImage != nil else {
            return .textToImage
        }
        guard strength < 1.0 else {
            return .textToImage
        }
        return .imageToImage
    }

    public init(
        prompt: String
    ) {
        self.prompt = prompt
    }

}


@available(iOS 16.2, macOS 13.1, *)
public extension StableDiffusionPipeline {

    /// Type of processing that will be performed to generate an image
    typealias Mode = PipelineMode

    /// Image generation configuration
    typealias Configuration = PipelineConfiguration
}
