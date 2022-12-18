// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// A decoder model which produces RGB images from latent samples
@available(iOS 16.2, macOS 13.1, *)
public struct Decoder: ResourceManaging {

    /// VAE decoder model
    var model: ManagedMLModel

    /// Create decoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE decoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A decoder that will lazily load its required resources when needed or requested
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

    /// Batch decode latent samples into images
    ///
    ///  - Parameters:
    ///    - latents: Batch of latent samples to decode
    ///  - Returns: decoded images
    public func decode(_ latents: [MLShapedArray<Float32>]) throws -> [CGImage] {

        // Form batch inputs for model
        let inputs: [MLFeatureProvider] = try latents.map { sample in
            // Reference pipeline scales the latent samples before decoding
            let sampleScaled = MLShapedArray<Float32>(
                scalars: sample.scalars.map { $0 / 0.18215 },
                shape: sample.shape)

            let dict = [inputName: MLMultiArray(sampleScaled)]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Batch predict with model
        let results = try model.perform { model in
            try model.predictions(fromBatch: batch)
        }

        // Transform the outputs to CGImages
        let images: [CGImage] = try (0..<results.count).map { i in
            let result = results.features(at: i)
            let outputName = result.featureNames.first!
            let output = result.featureValue(for: outputName)!.multiArrayValue!
            return try CGImage.fromShapedArray(MLShapedArray<Float32>(output))
        }

        return images
    }

    var inputName: String {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.key
        }
    }

}
