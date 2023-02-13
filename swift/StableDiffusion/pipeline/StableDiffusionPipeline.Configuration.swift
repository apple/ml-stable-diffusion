// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreGraphics

@available(iOS 16.2, macOS 13.1, *)
extension StableDiffusionPipeline {
    
    /// Tyoe of processing that will be performed to generate an image
    public enum Mode {
        case textToImage
        case imageToImage
        // case inPainting
    }
    
    /// Image generation configuration
    public struct Configuration: Hashable {
        
        /// Text prompt to guide sampling
        public var prompt: String
        /// Negative text prompt to guide sampling
        public var negativePrompt: String = ""
        /// Starting image for image2image or in-painting
        public var startingImage: CGImage? = nil
        //public var maskImage: CGImage? = nil
        public var strength: Float = 1.0
        /// Number of images to generate
        public var imageCount: Int = 1
        /// Number of inference steps to perform
        public var stepCount: Int = 50
        /// Random seed which to start generation
        public var seed: UInt32 = 0
        /// Controls the influence of the text prompt on sampling process (0=random images)
        public var guidanceScale: Float = 7.5
        /// Safety checks are only performed if `self.canSafetyCheck && !disableSafety`
        public var disableSafety: Bool = false
        /// The type of Scheduler to use.
        public var schedulerType: StableDiffusionScheduler = .pndmScheduler
        /// The type of RNG to use
        public var rngType: StableDiffusionRNG = .numpyRNG
        
        /// Given the configuration, what mode will be used for generation
        public var mode: Mode {
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

}
