// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

@available(iOS 16.0, macOS 13.0, *)
/// Encoder, currently supports image2image
public struct Encoder {
    
    public enum Error: String, Swift.Error {
        case latentOutputNotValid
        case batchLatentOutputEmpty
    }
    
    /// VAE encoder model + post math and adding noise from schedular
    var model: MLModel
    
    /// Create decoder from Core ML model
    ///
    /// - Parameters
    ///     - model: Core ML model for VAE decoder
    public init(model: MLModel) {
        self.model = model
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
            "sample": MLMultiArray(sample),
            "diagonalNoise": MLMultiArray(diagonalNoise),
            "noise": MLMultiArray(noise),
            "sqrtAlphasCumprod": MLMultiArray(sqrtAlphasCumprod),
            "sqrtOneMinusAlphasCumprod": MLMultiArray(sqrtOneMinusAlphasCumprod),
        ]
        let featureProvider = try MLDictionaryFeatureProvider(dictionary: dict)
        
        let batch = MLArrayBatchProvider(array: [featureProvider])

        // Batch predict with model
        let results = try queue.sync { try model.predictions(fromBatch: batch) }
        
        let batchLatents: [MLShapedArray<Float32>] = try (0..<results.count).compactMap { i in
            let result = results.features(at: i)
            guard
                let outputName = result.featureNames.first,
                let output = result.featureValue(for: outputName)?.multiArrayValue
            else {
                throw Error.latentOutputNotValid
            }
            print("output.shape: \(output.shape)")
            return MLShapedArray(output)
        }
        
        guard let latents = batchLatents.first else {
            throw Error.batchLatentOutputEmpty
        }
        
        return latents
    }
    
}
