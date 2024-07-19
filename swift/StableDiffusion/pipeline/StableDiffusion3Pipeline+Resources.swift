// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.

import CoreML
import Foundation
import Tokenizers
import Hub

@available(iOS 17.0, macOS 14.0, *)
public extension StableDiffusion3Pipeline {
    struct ResourceURLs {
        public let textEncoderURL: URL
        public let textEncoder2URL: URL
        public let textEncoderT5URL: URL
        public let mmditURL: URL
        public let decoderURL: URL
        public let encoderURL: URL
        public let vocabURL: URL
        public let mergesURL: URL
        public let configT5URL: URL
        public let dataT5URL: URL

        public init(resourcesAt baseURL: URL) {
            textEncoderURL = baseURL.appending(path: "TextEncoder.mlmodelc")
            textEncoder2URL = baseURL.appending(path: "TextEncoder2.mlmodelc")
            textEncoderT5URL = baseURL.appending(path: "TextEncoderT5.mlmodelc")
            mmditURL = baseURL.appending(path: "MultiModalDiffusionTransformer.mlmodelc")
            decoderURL = baseURL.appending(path: "VAEDecoder.mlmodelc")
            encoderURL = baseURL.appending(path: "VAEEncoder.mlmodelc")
            vocabURL = baseURL.appending(path: "vocab.json")
            mergesURL = baseURL.appending(path: "merges.txt")
            configT5URL = baseURL.appending(path: "tokenizer_config.json")
            dataT5URL = baseURL.appending(path: "tokenizer.json")
        }
    }

    /// Create stable diffusion pipeline using model resources at a
    /// specified URL
    ///
    /// - Parameters:
    ///   - baseURL: URL pointing to directory holding all model and tokenization resources
    ///   - configuration: The configuration to load model resources with
    ///   - reduceMemory: Setup pipeline in reduced memory mode
    /// - Returns:
    ///  Pipeline ready for image generation if all  necessary resources loaded
    init(
        resourcesAt baseURL: URL,
        configuration config: MLModelConfiguration = .init(),
        reduceMemory: Bool = false
    ) throws {
        // Expect URL of each resource
        let urls = ResourceURLs(resourcesAt: baseURL)
        let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
        let textEncoder = TextEncoderXL(tokenizer: tokenizer, modelAt: urls.textEncoderURL, configuration: config)

        // padToken is different in the second XL text encoder
        let tokenizer2 = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL, padToken: "!")
        let textEncoder2 = TextEncoderXL(tokenizer: tokenizer2, modelAt: urls.textEncoder2URL, configuration: config)

        // Optional T5 encoder
        var textEncoderT5: TextEncoderT5?
        if FileManager.default.fileExists(atPath: urls.configT5URL.path),
           FileManager.default.fileExists(atPath: urls.dataT5URL.path),
           FileManager.default.fileExists(atPath: urls.textEncoderT5URL.path)
        {
            let tokenizerT5 = try PreTrainedTokenizer(tokenizerConfig: Config(fileURL: urls.configT5URL), tokenizerData: Config(fileURL: urls.dataT5URL))
            textEncoderT5 = TextEncoderT5(tokenizer: tokenizerT5, modelAt: urls.textEncoderT5URL, configuration: config)
        } else {
            textEncoderT5 = nil
        }

        // Denoiser model
        let mmdit = MultiModalDiffusionTransformer(modelAt: urls.mmditURL, configuration: config)

        // Image Decoder
        let decoder = Decoder(modelAt: urls.decoderURL, configuration: config)

        // Optional Image Encoder
        let encoder: Encoder?
        if FileManager.default.fileExists(atPath: urls.encoderURL.path) {
            encoder = Encoder(modelAt: urls.encoderURL, configuration: config)
        } else {
            encoder = nil
        }

        // Construct pipeline
        self.init(
            textEncoder: textEncoder,
            textEncoder2: textEncoder2,
            textEncoderT5: textEncoderT5,
            mmdit: mmdit,
            decoder: decoder,
            encoder: encoder,
            reduceMemory: reduceMemory
        )
    }
}
