// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// U-Net noise prediction model for stable diffusion
public struct Unet {

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    var models: [MLModel]

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - model: U-Net held in single Core ML model
    /// - Returns: Ready for prediction
    public init(model: MLModel) {
        self.models = [model]
    }

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - chunks: U-Net held chunked into multiple Core ML models
    /// - Returns: Ready for prediction
    public init(chunks: [MLModel]) {
        self.models = chunks
    }

    var latentSampleDescription: MLFeatureDescription {
        models.first!.modelDescription.inputDescriptionsByName["sample"]!
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

    /// Prediction queue
    let queue = DispatchQueue(label: "unet.predict")

    func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {

        var results = try queue.sync {
            try models.first!.predictions(fromBatch: batch)
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
            results = try queue.sync {
                try stage.predictions(fromBatch: nextBatch)
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
