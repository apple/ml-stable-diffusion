// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import NaturalLanguage

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
        public let controlNetDirURL: URL
        public let controlledUnetURL: URL
        public let controlledUnetChunk1URL: URL
        public let controlledUnetChunk2URL: URL
        public let multilingualTextEncoderProjectionURL: URL

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
            controlNetDirURL = baseURL.appending(path: "controlnet")
            controlledUnetURL = baseURL.appending(path: "ControlledUnet.mlmodelc")
            controlledUnetChunk1URL = baseURL.appending(path: "ControlledUnetChunk1.mlmodelc")
            controlledUnetChunk2URL = baseURL.appending(path: "ControlledUnetChunk2.mlmodelc")
            multilingualTextEncoderProjectionURL = baseURL.appending(path: "MultilingualTextEncoderProjection.mlmodelc")
        }
    }

    /// Create stable diffusion pipeline using model resources at a
    /// specified URL
    ///
    /// - Parameters:
    ///   - baseURL: URL pointing to directory holding all model and tokenization resources
    ///   - controlNetModelNames: Specify ControlNet models to use in generation
    ///   - configuration: The configuration to load model resources with
    ///   - disableSafety: Load time disable of safety to save memory
    ///   - reduceMemory: Setup pipeline in reduced memory mode
    ///   - useMultilingualTextEncoder: Option to use system multilingual NLContextualEmbedding as encoder
    ///   - script: Optional natural language script to use for the text encoder.
    /// - Returns:
    ///  Pipeline ready for image generation if all  necessary resources loaded
    init(
        resourcesAt baseURL: URL,
        controlNet controlNetModelNames: [String],
        configuration config: MLModelConfiguration = .init(),
        disableSafety: Bool = false,
        reduceMemory: Bool = false,
        useMultilingualTextEncoder: Bool = false,
        script: Script? = nil
    ) throws {

        /// Expect URL of each resource
        let urls = ResourceURLs(resourcesAt: baseURL)
        let textEncoder: TextEncoderModel

#if canImport(NaturalLanguage.NLScript)
        if useMultilingualTextEncoder {
            guard #available(macOS 14.0, iOS 17.0, *) else { throw PipelineError.unsupportedOSVersion }
            textEncoder = MultilingualTextEncoder(
                modelAt: urls.multilingualTextEncoderProjectionURL,
                configuration: config,
                script: script ?? .latin
            )
        } else {
            let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
            textEncoder = TextEncoder(tokenizer: tokenizer, modelAt: urls.textEncoderURL, configuration: config)
        }
#else
        let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
        textEncoder = TextEncoder(tokenizer: tokenizer, modelAt: urls.textEncoderURL, configuration: config)
#endif

        // ControlNet model
        var controlNet: ControlNet? = nil
        let controlNetURLs = controlNetModelNames.map { model in
            let fileName = model + ".mlmodelc"
            return urls.controlNetDirURL.appending(path: fileName)
        }
        if !controlNetURLs.isEmpty {
            controlNet = ControlNet(modelAt: controlNetURLs, configuration: config)
        }

        // Unet model
        let unet: Unet
        let unetURL: URL, unetChunk1URL: URL, unetChunk2URL: URL
        
        // if ControlNet available, Unet supports additional inputs from ControlNet
        if controlNet == nil {
            unetURL = urls.unetURL
            unetChunk1URL = urls.unetChunk1URL
            unetChunk2URL = urls.unetChunk2URL
        } else {
            unetURL = urls.controlledUnetURL
            unetChunk1URL = urls.controlledUnetChunk1URL
            unetChunk2URL = urls.controlledUnetChunk2URL
        }
        
        if FileManager.default.fileExists(atPath: unetChunk1URL.path) &&
            FileManager.default.fileExists(atPath: unetChunk2URL.path) {
            unet = Unet(chunksAt: [unetChunk1URL, unetChunk2URL],
                        configuration: config)
        } else {
            unet = Unet(modelAt: unetURL, configuration: config)
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
        if #available(macOS 14.0, iOS 17.0, *) {
            self.init(
                textEncoder: textEncoder,
                unet: unet,
                decoder: decoder,
                encoder: encoder,
                controlNet: controlNet,
                safetyChecker: safetyChecker,
                reduceMemory: reduceMemory,
                useMultilingualTextEncoder: useMultilingualTextEncoder,
                script: script
            )
        } else {
            self.init(
                textEncoder: textEncoder,
                unet: unet,
                decoder: decoder,
                encoder: encoder,
                controlNet: controlNet,
                safetyChecker: safetyChecker,
                reduceMemory: reduceMemory
            )
        }
    }
}
