// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

@available(iOS 16.2, macOS 13.1, *)
/// Encoder, currently supports image2image
public struct Encoder: ResourceManaging {
    
    public enum FeatureName: String {
        case sample = "sample"
        case diagonalNoise = "diagonal_noise"
        case noise = "noise"
        case sqrtAlphasCumprod = "sqrt_alphas_cumprod"
        case sqrtOneMinusAlphasCumprod = "sqrt_one_minus_alphas_cumprod"
    }
    
    public enum Error: String, Swift.Error {
        case latentOutputNotValid
        case batchLatentOutputEmpty
        case sampleInputShapeNotCorrect
        case noiseInputShapeNotCorrect
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

    /// Batch encode latent samples into images
    /// - Parameters:
    ///   - image: image used for image2image
    ///   - diagonalNoise: random noise for `DiagonalGaussianDistribution` operation
    ///   - noise: random noise for initial latent space based on strength argument
    ///   - alphasCumprodStep: calculations using the scheduler traditionally calculated in the pipeline in pyTorch Diffusers library.
    /// - Returns: The encoded latent space as MLShapedArray
    public func encode(
        image:  CGImage,
        diagonalNoise: MLShapedArray<Float32>,
        noise: MLShapedArray<Float32>,
        alphasCumprodStep: AlphasCumprodCalculation
    ) throws -> MLShapedArray<Float32> {
        let sample = try image.plannerRGBShapedArray
        let sqrtAlphasCumprod = MLShapedArray(scalars: [alphasCumprodStep.sqrtAlphasCumprod], shape: [1, 1])
        let sqrtOneMinusAlphasCumprod = MLShapedArray(scalars: [alphasCumprodStep.sqrtOneMinusAlphasCumprod], shape: [1, 1])
        
        let dict: [String: Any] = [
            FeatureName.sample.rawValue: MLMultiArray(sample),
            FeatureName.diagonalNoise.rawValue: MLMultiArray(diagonalNoise),
            FeatureName.noise.rawValue: MLMultiArray(noise),
            FeatureName.sqrtAlphasCumprod.rawValue: MLMultiArray(sqrtAlphasCumprod),
            FeatureName.sqrtOneMinusAlphasCumprod.rawValue: MLMultiArray(sqrtOneMinusAlphasCumprod),
        ]
        let featureProvider = try MLDictionaryFeatureProvider(dictionary: dict)
        
        let batch = MLArrayBatchProvider(array: [featureProvider])

        // Batch predict with model
        
        let results = try queue.sync {
            try model.perform { model in
                if let feature = model.modelDescription.inputDescriptionsByName[FeatureName.sample.rawValue],
                   let shape = feature.multiArrayConstraint?.shape as? [Int]
                {
                    guard sample.shape == shape else {
                        // TODO: Consider auto resizing and croping similar to how Vision or CoreML auto-generated Swift code can accomplish with `MLFeatureValue`
                        throw Error.sampleInputShapeNotCorrect
                    }
                }
                
                if let feature = model.modelDescription.inputDescriptionsByName[FeatureName.noise.rawValue],
                   let shape = feature.multiArrayConstraint?.shape as? [Int]
                {
                    guard noise.shape == shape else {
                        throw Error.noiseInputShapeNotCorrect
                    }
                }
                
                if let feature = model.modelDescription.inputDescriptionsByName[FeatureName.diagonalNoise.rawValue],
                   let shape = feature.multiArrayConstraint?.shape as? [Int]
                {
                    guard diagonalNoise.shape == shape else {
                        throw Error.noiseInputShapeNotCorrect
                    }
                }
                 
                return try model.predictions(fromBatch: batch)
            }
        }
        
        let batchLatents: [MLShapedArray<Float32>] = try (0..<results.count).compactMap { i in
            let result = results.features(at: i)
            guard
                let outputName = result.featureNames.first,
                let output = result.featureValue(for: outputName)?.multiArrayValue
            else {
                throw Error.latentOutputNotValid
            }
            return MLShapedArray(output)
        }
        
        guard let latents = batchLatents.first else {
            throw Error.batchLatentOutputEmpty
        }
        
        return latents
    }
    
}
