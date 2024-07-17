// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import Tokenizers

@available(iOS 17.0, macOS 14.0, *)
public protocol TextEncoderT5Model: ResourceManaging {
    func encode(_ text: String) throws -> TextEncoderT5Output
}

@available(iOS 17.0, macOS 14.0, *)
public struct TextEncoderT5Output {
    public let encoderHiddenStates: MLShapedArray<Float32>
}

///  A model for encoding text, suitable for SD3
@available(iOS 17.0, macOS 14.0, *)
public struct TextEncoderT5: TextEncoderT5Model {

    /// Text tokenizer
    var tokenizer: Tokenizer

    /// Embedding model
    var model: ManagedMLModel

    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - url: Location of compiled text encoding  Core ML model
    ///   - configuration: configuration to be used when the model is loaded
    /// - Returns: A text encoder that will lazily load its required resources when needed or requested
    public init(tokenizer: Tokenizer,
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
    public func encode(_ text: String) throws -> TextEncoderT5Output {

        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize, padding to the expected length
        var tokens = tokenizer.tokenize(text: text)
        var ids = tokens.map { tokenizer.convertTokenToId($0) ?? 0 }

        // Truncate if necessary
        if ids.count > inputLength {
            tokens = tokens.dropLast(tokens.count - inputLength)
            ids = ids.dropLast(ids.count - inputLength)
            print("Needed to truncate input for TextEncoderT5")
        }

        // Use the model to generate the embedding
        let encodedText = try encode(ids: ids)
        return encodedText
    }

    func encode(ids: [Int]) throws -> TextEncoderT5Output {
        let inputName = "input_ids"
        let inputShape = inputShape
        let inputLength = inputShape[1]
                
        let bosToken = tokenizer.bosTokenId ?? 0
        let eosToken = tokenizer.eosTokenId ?? 1
        let padToken = bosToken
        let maskToken = eosToken

        // Truncate and pad input to the expected length
        let truncatedIds = ids.prefix(inputLength - 1) + [eosToken]
        let inputIds = truncatedIds + Array(repeating: padToken, count: inputLength - truncatedIds.count)

        let attentionMaskName = "attention_mask"
        var attentionMask: [Int] = inputIds.map { token in
            token == padToken ? maskToken : padToken
        }
        attentionMask[0] = bosToken

        let floatIds = inputIds.map { Float32($0) }
        let floatMask = attentionMask.map { Float32($0) }

        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
        let maskArray = MLShapedArray<Float32>(scalars: floatMask, shape: inputShape)
        let inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: [inputName: MLMultiArray(inputArray),
                         attentionMaskName: MLMultiArray(maskArray)])

        let result = try model.perform { model in
            try model.prediction(from: inputFeatures)
        }

        let embeddingFeature = result.featureValue(for: "encoder_hidden_states")
        return TextEncoderT5Output(encoderHiddenStates: MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!))
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
