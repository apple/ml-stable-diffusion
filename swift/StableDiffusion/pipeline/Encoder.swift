// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// A encoder model which produces latent samples from RGB images
@available(iOS 16.2, macOS 13.1, *)
public struct Encoder: ResourceManaging {
    
    public enum Error: String, Swift.Error {
        case sampleInputShapeNotCorrect
    }
    
    /// VAE encoder model + post math and adding noise from schedular
    var model: ManagedMLModel
    
    /// Create encoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE encoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: An encoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }
    
    /// Ensure the model has been loaded into memory
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
       model.unloadResources()
    }
    
    /// Prediction queue
    let queue = DispatchQueue(label: "encoder.predict")

    /// Encode image into latent sample
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///    - scaleFactor: scalar multiplier on latents before encoding image
    ///    - random
    ///  - Returns: The encoded latent space as MLShapedArray
    public func encode(
        _ image: CGImage,
        scaleFactor: Float32,
        random: inout RandomSource
    ) throws -> MLShapedArray<Float32> {
        let imageData = try image.plannerRGBShapedArray
        guard imageData.shape == inputShape else {
            // TODO: Consider auto resizing and croping similar to how Vision or CoreML auto-generated Swift code can accomplish with `MLFeatureValue`
            throw Error.sampleInputShapeNotCorrect
        }
        let dict = [inputName: MLMultiArray(imageData)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)
        
        let result = try model.perform { model in
            try model.prediction(from: input)
        }
        let outputName = result.featureNames.first!
        let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
        let output = MLShapedArray<Float32>(outputValue)
        
        // DiagonalGaussianDistribution
        let mean = output[0][0..<4]
        let logvar = MLShapedArray<Float32>(
            scalars: output[0][4..<8].scalars.map { min(max($0, -30), 20) },
            shape: mean.shape
        )
        let std = MLShapedArray<Float32>(
            scalars: logvar.scalars.map { exp(0.5 * $0) },
            shape: logvar.shape
        )
        let latent = MLShapedArray<Float32>(
            scalars: zip(mean.scalars, std.scalars).map {
                Float32(random.nextNormal(mean: Double($0), stdev: Double($1)))
            },
            shape: logvar.shape
        )
        
        // Reference pipeline scales the latent after encoding
        let latentScaled = MLShapedArray<Float32>(
            scalars: latent.scalars.map { $0 * scaleFactor },
            shape: [1] + latent.shape
        )

        return latentScaled
    }
    
    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName["z"]!
        }
    }
    
    var inputName: String {
        inputDescription.name
    }
    
    /// The expected shape of the models latent sample input
    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
