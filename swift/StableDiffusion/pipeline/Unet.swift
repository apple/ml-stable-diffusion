// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// U-Net noise prediction model for stable diffusion
@available(iOS 16.2, macOS 13.1, *)
public struct Unet: ResourceManaging {

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    var models: [ManagedMLModel]

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.models = [ManagedMLModel(modelAt: url, configuration: configuration)]
    }

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - urls: Location of chunked U-Net via urls to each compiled chunk
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(chunksAt urls: [URL],
                configuration: MLModelConfiguration) {
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
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

    var latentSampleDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["sample"]!
        }
    }

    /// The expected shape of the models latent sample input
    public var latentSampleShape: [Int] {
        latentSampleDescription.multiArrayConstraint!.shape.map { $0.intValue }
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
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>
    ) throws -> [MLShapedArray<Float32>] {

        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray<Float32>(scalars:[Float(timeStep), Float(timeStep)],shape:[2])

        // Form batch input to model
        let inputs = try latents.map {
            let dict: [String: Any] = [
                "sample" : MLMultiArray($0),
                "timestep" : MLMultiArray(t),
                "encoder_hidden_states": MLMultiArray(hiddenStates)
            ]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Make predictions
        let results = try predictions(from: batch)

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

    func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {

        var results = try models.first!.perform { model in
            try model.predictions(fromBatch: batch)
        }

        if models.count == 1 {
            return results
        }

        // Manual pipeline batch prediction
        let inputs = batch.arrayOfFeatureValueDictionaries
        for stage in models.dropFirst() {

            // Combine the original inputs with the outputs of the last stage
            let next = try results.arrayOfFeatureValueDictionaries
                .enumerated().map { (index, dict) in
                    let nextDict =  dict.merging(inputs[index]) { (out, _) in out }
                    return try MLDictionaryFeatureProvider(dictionary: nextDict)
            }
            let nextBatch = MLArrayBatchProvider(array: next)

            // Predict
            results = try stage.perform { model in
                try model.predictions(fromBatch: nextBatch)
            }
        }

        return results
    }
}

extension MLFeatureProvider {
    var featureValueDictionary: [String : MLFeatureValue] {
        self.featureNames.reduce(into: [String : MLFeatureValue]()) { result, name in
            result[name] = self.featureValue(for: name)
        }
    }
}

extension MLBatchProvider {
    var arrayOfFeatureValueDictionaries: [[String : MLFeatureValue]] {
        (0..<self.count).map {
            self.features(at: $0).featureValueDictionary
        }
    }
}
