// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

import CoreML

/// MMDiT noise prediction model for stable diffusion
@available(iOS 16.2, macOS 13.1, *)
public struct MultiModalDiffusionTransformer: ResourceManaging {
    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    var models: [ManagedMLModel]

    /// Creates a MMDiT noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single MMDiT compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: MMDiT model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration)
    {
        self.models = [ManagedMLModel(modelAt: url, configuration: configuration)]
    }

    /// Load resources.
    public func loadResources() throws {
        for model in models {
            try model.loadResources()
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }

    /// Pre-warm resources
    public func prewarmResources() throws {
        // Override default to pre-warm each model
        for model in models {
            try model.loadResources()
            model.unloadResources()
        }
    }

    var latentImageEmbeddingsDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["latent_image_embeddings"]!
        }
    }

    /// The expected shape of the models latent sample input
    public var latentImageEmbeddingsShape: [Int] {
        latentImageEmbeddingsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    var tokenLevelTextEmbeddingsDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["token_level_text_embeddings"]!
        }
    }

    /// The expected shape of the geometry conditioning
    public var tokenLevelTextEmbeddingsShape: [Int] {
        tokenLevelTextEmbeddingsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictNoise(
        latents: [MLShapedArray<Float32>],
        timeStep: Float,
        tokenLevelTextEmbeddings: MLShapedArray<Float32>,
        pooledTextEmbeddings: MLShapedArray<Float32>
    ) throws -> [MLShapedArray<Float32>] {
        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray<Float32>(scalars: [timeStep, timeStep], shape: [2])

        // Form batch input to model
        let inputs = try latents.enumerated().map {
            let dict: [String: Any] = [
                "latent_image_embeddings": MLMultiArray($0.element),
                "timestep": MLMultiArray(t),
                "token_level_text_embeddings": MLMultiArray(tokenLevelTextEmbeddings),
                "pooled_text_embeddings": MLMultiArray(pooledTextEmbeddings),
            ]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Make predictions
        let results = try models.predictions(from: batch)

        // Pull out the results in Float32 format
        let noise = (0..<results.count).map { i in

            let result = results.features(at: i)
            let outputName = result.featureNames.first!

            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!

            // To conform to this func return type make sure we return float32
            // Use the fact that the concatenating constructor for MLMultiArray
            // can do type conversion:
            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            return MLShapedArray<Float32>(fp32Noise)
        }

        return noise
    }
}
