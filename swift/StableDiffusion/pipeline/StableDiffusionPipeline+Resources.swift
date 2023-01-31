// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

@available(iOS 16.2, macOS 13.1, *)
public extension StableDiffusionPipeline {

    struct ResourceURLs {

        public let textEncoderURL: URL
        public let unetURL: URL
        public let unetChunk1URL: URL
        public let unetChunk2URL: URL
        public let decoderURL: URL
        public let encoderURL: URL
        public let safetyCheckerURL: URL
        public let vocabURL: URL
        public let mergesURL: URL

        public init(resourcesAt baseURL: URL) {
            textEncoderURL = baseURL.appending(path: "TextEncoder.mlmodelc")
            unetURL = baseURL.appending(path: "Unet.mlmodelc")
            unetChunk1URL = baseURL.appending(path: "UnetChunk1.mlmodelc")
            unetChunk2URL = baseURL.appending(path: "UnetChunk2.mlmodelc")
            decoderURL = baseURL.appending(path: "VAEDecoder.mlmodelc")
            encoderURL = baseURL.appending(path: "VAEEncoder.mlmodelc")
            safetyCheckerURL = baseURL.appending(path: "SafetyChecker.mlmodelc")
            vocabURL = baseURL.appending(path: "vocab.json")
            mergesURL = baseURL.appending(path: "merges.txt")
        }
    }

    /// Create stable diffusion pipeline using model resources at a
    /// specified URL
    ///
    /// - Parameters:
    ///    - baseURL: URL pointing to directory holding all model
    ///               and tokenization resources
    ///   - configuration: The configuration to load model resources with
    ///   - disableSafety: Load time disable of safety to save memory
    ///   - reduceMemory: Setup pipeline in reduced memory mode
    /// - Returns:
    ///  Pipeline ready for image generation if all  necessary resources loaded
    init(resourcesAt baseURL: URL,
         configuration config: MLModelConfiguration = .init(),
         disableSafety: Bool = false,
         reduceMemory: Bool = false) throws {

        /// Expect URL of each resource
        let urls = ResourceURLs(resourcesAt: baseURL)

        // Text tokenizer and encoder
        let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
        let textEncoder = TextEncoder(tokenizer: tokenizer,
                                      modelAt: urls.textEncoderURL,
                                      configuration: config)

        // Unet model
        let unet: Unet
        if FileManager.default.fileExists(atPath: urls.unetChunk1URL.path) &&
            FileManager.default.fileExists(atPath: urls.unetChunk2URL.path) {
            unet = Unet(chunksAt: [urls.unetChunk1URL, urls.unetChunk2URL],
                        configuration: config)
        } else {
            unet = Unet(modelAt: urls.unetURL, configuration: config)
        }

        // Image Decoder
        let decoder = Decoder(modelAt: urls.decoderURL, configuration: config)

        // Optional safety checker
        var safetyChecker: SafetyChecker? = nil
        if !disableSafety &&
            FileManager.default.fileExists(atPath: urls.safetyCheckerURL.path) {
            safetyChecker = SafetyChecker(modelAt: urls.safetyCheckerURL, configuration: config)
        }
        
        // Optional Image Encoder
        let encoder: Encoder?
        if FileManager.default.fileExists(atPath: urls.encoderURL.path) {
            encoder = Encoder(modelAt: urls.encoderURL, configuration: config)
        } else {
            encoder = nil
        }

        // Construct pipeline
        self.init(textEncoder: textEncoder,
                  unet: unet,
                  decoder: decoder,
                  encoder: encoder,
                  safetyChecker: safetyChecker,
                  reduceMemory: reduceMemory)
    }
}
