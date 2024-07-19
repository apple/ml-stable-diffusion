// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import CoreML

/// A class to manage and gate access to a Core ML model
///
/// It will automatically load a model into memory when needed or requested
/// It allows one to request to unload the model from memory
@available(iOS 16.2, macOS 13.1, *)
public final class ManagedMLModel: ResourceManaging {

    /// The location of the model
    var modelURL: URL

    /// The configuration to be used when the model is loaded
    var configuration: MLModelConfiguration

    /// The loaded model (when loaded)
    var loadedModel: MLModel?

    /// Queue to protect access to loaded model
    var queue: DispatchQueue

    /// Create a managed model given its location and desired loaded configuration
    ///
    /// - Parameters:
    ///     - url: The location of the model
    ///     - configuration: The configuration to be used when the model is loaded/used
    /// - Returns: A managed model that has not been loaded
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.modelURL = url
        self.configuration = configuration
        self.loadedModel = nil
        self.queue = DispatchQueue(label: "managed.\(url.lastPathComponent)")
    }

    /// Instantiation and load model into memory
    public func loadResources() throws {
        try queue.sync {
            try loadModel()
        }
    }

    /// Unload the model if it was loaded
    public func unloadResources() {
        queue.sync {
            loadedModel = nil
        }
    }

    /// Perform an operation with the managed model via a supplied closure.
    ///  The model will be loaded and supplied to the closure and should only be
    ///  used within the closure to ensure all resource management is synchronized
    ///
    /// - Parameters:
    ///     - body: Closure which performs and action on a loaded model
    /// - Returns: The result of the closure
    /// - Throws: An error if the model cannot be loaded or if the closure throws
    public func perform<R>(_ body: (MLModel) throws -> R) throws -> R {
        return try queue.sync {
            try autoreleasepool {
                try loadModel()
                return try body(loadedModel!)
            }
        }
    }

    private func loadModel() throws {
        if loadedModel == nil {
            loadedModel = try MLModel(contentsOf: modelURL,
                                      configuration: configuration)
        }
    }
}

@available(iOS 16.2, macOS 13.1, *)
public extension Array where Element == ManagedMLModel {
    /// Performs batch predictions using an array of `[ManagedMLModel]` instances in a pipeline.
    /// - Parameter batch: Inputs for btached predictions.
    /// - Returns: Final prediction results after processing through all models.
    /// - Throws: Errors if the array is empty, predictions fail, or results can't be combined.
    func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {
        var results = try self.first!.perform { model in
            try model.predictions(fromBatch: batch)
        }

        if self.count == 1 {
            return results
        }

        // Manual pipeline batch prediction
        let inputs = batch.arrayOfFeatureValueDictionaries
        for stage in self.dropFirst() {
            // Combine the original inputs with the outputs of the last stage
            let next = try results.arrayOfFeatureValueDictionaries
                .enumerated().map { index, dict in
                    let nextDict = dict.merging(inputs[index]) { out, _ in out }
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
