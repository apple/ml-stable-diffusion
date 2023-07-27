// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

@available(iOS 17.0, macOS 14.0, *)
public protocol TextEncoderXLModel: ResourceManaging {
    typealias TextEncoderXLOutput = (hiddenEmbeddings: MLShapedArray<Float32>, pooledOutputs: MLShapedArray<Float32>)
    func encode(_ text: String) throws -> TextEncoderXLOutput
}

///  A model for encoding text, suitable for SDXL
@available(iOS 17.0, macOS 14.0, *)
public struct TextEncoderXL: TextEncoderXLModel {

    /// Text tokenizer
    var tokenizer: BPETokenizer

    /// Embedding model
    var model: ManagedMLModel

    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - url: Location of compiled text encoding  Core ML model
    ///   - configuration: configuration to be used when the model is loaded
    /// - Returns: A text encoder that will lazily load its required resources when needed or requested
    public init(tokenizer: BPETokenizer,
                modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.tokenizer = tokenizer
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

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    public func encode(_ text: String) throws -> TextEncoderXLOutput {

        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize, padding to the expected length
        var (tokens, ids) = tokenizer.tokenize(input: text, minCount: inputLength)

        // Truncate if necessary
        if ids.count > inputLength {
            tokens = tokens.dropLast(tokens.count - inputLength)
            ids = ids.dropLast(ids.count - inputLength)
            let truncated = tokenizer.decode(tokens: tokens)
            print("Needed to truncate input '\(text)' to '\(truncated)'")
        }

        // Use the model to generate the embedding
        return try encode(ids: ids)
    }

    func encode(ids: [Int]) throws -> TextEncoderXLOutput {
        let inputName = inputDescription.name
        let inputShape = inputShape

        let floatIds = ids.map { Float32($0) }
        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
        let inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: [inputName: MLMultiArray(inputArray)])

        let result = try model.perform { model in
            try model.prediction(from: inputFeatures)
        }

        let embeddingFeature = result.featureValue(for: "hidden_embeds")
        let pooledFeature = result.featureValue(for: "pooled_outputs")
        return (MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!), MLShapedArray<Float32>(converting: pooledFeature!.multiArrayValue!))
    }

    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.value
        }
    }

    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
