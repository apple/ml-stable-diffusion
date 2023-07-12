// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Foundation
import NaturalLanguage
import CoreML

#if canImport(NaturalLanguage.NLContextualEmbedding)
@available(iOS 17.0, macOS 14.0, *)
public struct MultilingualTextEncoder: TextEncoderModel {
    let adapter: ManagedMLModel?

    let embeddingModel: NLContextualEmbedding

    // TODO: use maximum sequence length from embedding.
    let maximumEmbeddingSequenceLength = 256

    /// Creates a multilingual text encoder.
    ///
    /// - Parameters:
    ///   - url: The location of the compiled Core ML adapter model. The model is a linear projection layer that
    ///   transforms the contextual embedding size of 512 to the default text encoder CLIP size of 768.
    ///   - configuration: The configuration to be used when the model is loaded.
    ///   - script: The scipt of the contextual embedding.
    public init(
        modelAt url: URL? = nil,
        configuration: MLModelConfiguration = .init(),
        script: Script = .latin
    ) {
        if let url {
            self.adapter = ManagedMLModel(modelAt: url, configuration: configuration)
        } else {
            self.adapter = nil
        }
        self.embeddingModel = NLContextualEmbedding(script: script.asNLScript)!
        self.embeddingModel.requestAssets { _, _ in }
    }

    /// Loads model resources into memory.
    public func loadResources() throws {
        try adapter?.loadResources()
        try embeddingModel.load()
    }

    /// Unloads the model resources to free up memory.
    public func unloadResources() {
        adapter?.unloadResources()
        embeddingModel.unload()
    }

    /// Encodes the input text.
    ///
    ///  - Parameter text: The input text.
    ///  - Returns: An embedding shaped array.
    public func encode(_ text: String) throws -> MLShapedArray<Float> {
        guard embeddingModel.hasAvailableAssets else {
            throw Error.missingEmbeddingResource
        }

        // Create the text embedding result.
        let embedding = try embeddingModel.embeddingResult(for: text, language: nil)

        // Create embedding array from token vectors.
        var shapedEmbeddings = MLShapedArray<Double>(
            repeating: 0.0,
            shape: [1, maximumEmbeddingSequenceLength, embeddingModel.dimension]
        )
        shapedEmbeddings.withUnsafeMutableShapedBufferPointer { pointer, _, _ in
            var tokenIndex = 0
            embedding.enumerateTokenVectors(in: text.startIndex ..< text.endIndex) { (tokenEmbeddings, _) -> Bool in
                for tokenEmbeddingIndex in 0 ..< tokenEmbeddings.count {
                    pointer[tokenIndex * embeddingModel.dimension + tokenEmbeddingIndex] = tokenEmbeddings[tokenEmbeddingIndex]
                }
                tokenIndex += 1
                return true
            }
        }

        if adapter == nil {
            // Return embeddings with shape [1, 256, 512].
            return MLShapedArray(converting: shapedEmbeddings)
        } else {
            // Project the embeddings to the correct CLIP model input shape of [1, 768, 1, 256].
            return try projectEmbeddings(shapedEmbeddings)
        }
    }

    /// Creates the adapter model input feature provider.
    private func prepareProjectionInput(_ input: MLShapedArray<Double>) throws -> MLDictionaryFeatureProvider {
        guard let adapter else {
            fatalError("Cannot prepare projection input without an adapter.")
        }
        return try adapter.perform { model in
            guard let inputDescription = model.modelDescription.inputDescriptionsByName.first?.value else {
                throw Error.missingAdapterInput
            }
            return try MLDictionaryFeatureProvider(dictionary: [inputDescription.name: MLMultiArray(input)])
        }
    }

    /// Processes the adapter model output feature provider.
    private func processProjectionOutput(_ output: MLFeatureProvider) throws -> MLShapedArray<Float> {
        guard let adapter else {
            fatalError("Cannot process projection output without an adapter.")
        }
        return try adapter.perform { model in
            guard let outputDescription = model.modelDescription.outputDescriptionsByName.first?.value else {
                throw Error.missingAdapterOutput
            }
            guard let result = output
                .featureValue(for: outputDescription.name)?
                .multiArrayValue else {

                throw Error.incompatibleAdapterOutputDataFormat(
                    expected: .multiArray,
                    actual: outputDescription.type
                )
            }

            return MLShapedArray(converting: result)
        }
    }

    /// Projects the embeddings.
    private func projectEmbeddings(_ embeddings: MLShapedArray<Double>) throws -> MLShapedArray<Float> {
        guard let adapter else {
            fatalError("Cannot project embeddings without an adapter.")
        }
        let inputFeatureProvider = try prepareProjectionInput(embeddings)
        let projection = try adapter.perform { model in
            return try model.prediction(from: inputFeatureProvider)
        }
        return try processProjectionOutput(projection)
    }
}

@available(iOS 17.0, macOS 14.0, *)
extension MultilingualTextEncoder {
    /// A multilingual text encoder error.
    public enum Error: Swift.Error, LocalizedError, Equatable, CustomDebugStringConvertible {
        /// An error that indicates that the resource for the embedding is missing.
        case missingEmbeddingResource

        /// An error that indicates that the adapter model input data has the wrong format.
        case incompatibleAdapterInputDataFormat(expected: MLFeatureType, actual: MLFeatureType)

        /// An error that indicates that the adapter model output data has the wrong format.
        case incompatibleAdapterOutputDataFormat(expected: MLFeatureType, actual: MLFeatureType)

        /// An error that indicates that the adapter model is missing an input.
        case missingAdapterInput

        /// An error that indicates that the adapter model is missing an output.
        case missingAdapterOutput

        /// A debug description of the error.
        public var errorDescription: String? {
            debugDescription
        }

        /// A text representation of the error.
        public var debugDescription: String {
            switch self {
            case .missingEmbeddingResource:
                return "Resources required for generating embeddings are missing. Make sure that your device is connected to the internet and try again."
            case .incompatibleAdapterInputDataFormat(expected: let expected, actual: let actual):
                return "The adapter model input expected to be \(expected) but is \(actual)."
            case .incompatibleAdapterOutputDataFormat(expected: let expected, actual: let actual):
                return "The adapter model output expected to be \(expected) but is \(actual)."
            case .missingAdapterInput:
                return "The adapter model is missing an input."
            case .missingAdapterOutput:
                return "The adapter model is missing an output."
            }
        }
    }
}
#endif

@available(iOS 16.2, macOS 13.1, *)
public enum Script: String {
    case latin, cyrillic, cjk

#if canImport(NaturalLanguage.NLScript)
    @available(iOS 17.0, macOS 14.0, *)
    var asNLScript: NLScript {
        switch self {
        case .latin: return .latin
        case .cyrillic: return .cyrillic
        case .cjk: return .simplifiedChinese
        }
    }
#endif
}
