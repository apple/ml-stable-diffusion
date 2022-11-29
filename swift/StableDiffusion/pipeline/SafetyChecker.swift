// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import Accelerate

/// Image safety checking model
public struct SafetyChecker {

    /// Safety checking Core ML model
    var model: MLModel

    /// Creates safety checker
    ///
    /// - Parameters:
    ///     - model: Underlying model which performs the safety check
    /// - Returns: Safety checker ready from checks
    public init(model: MLModel) {
        self.model = model
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "safetycheker.predict")

    typealias PixelBufferPFx1 = vImage.PixelBuffer<vImage.PlanarF>
    typealias PixelBufferP8x1 = vImage.PixelBuffer<vImage.Planar8>
    typealias PixelBufferPFx3 = vImage.PixelBuffer<vImage.PlanarFx3>
    typealias PixelBufferP8x3 = vImage.PixelBuffer<vImage.Planar8x3>
    typealias PixelBufferIFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
    typealias PixelBufferI8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>
    typealias PixelBufferI8x4 = vImage.PixelBuffer<vImage.Interleaved8x4>

    enum SafetyCheckError: Error {
        case imageResizeFailure
        case imageToFloatFailure
        case modelInputFailure
        case unexpectedModelOutput
    }

    /// Check if image is safe
    ///
    /// - Parameters:
    ///     - image: Image to check
    /// - Returns: Whether the model considers the image to be safe
    public func isSafe(_ image: CGImage) throws -> Bool {

        let inputName = "clip_input"
        let adjustmentName = "adjustment"
        let imagesNames = "images"

        let inputInfo = model.modelDescription.inputDescriptionsByName
        let inputShape = inputInfo[inputName]!.multiArrayConstraint!.shape

        let width = inputShape[2].intValue
        let height = inputShape[3].intValue

        let resizedImage = try resizeToRGBA(image, width: width, height: height)

        let bufferP8x3 = try getRGBPlanes(of: resizedImage)

        let arrayPFx3 = normalizeToFloatShapedArray(bufferP8x3)

        guard let input = try? MLDictionaryFeatureProvider(
            dictionary:[
                // Input that is analyzed for safety
                inputName      : MLMultiArray(arrayPFx3),
                // No adjustment, use default threshold
                adjustmentName : MLMultiArray(MLShapedArray<Float32>(scalars: [0], shape: [1])),
                // Supplying dummy images to be filtered (will be ignored)
                imagesNames    : MLMultiArray(shape:[1, 512, 512, 3], dataType: .float16)
            ]
        ) else {
            throw SafetyCheckError.modelInputFailure
        }

        let result = try queue.sync { try model.prediction(from: input) }

        let output = result.featureValue(for: "has_nsfw_concepts")

        guard let unsafe = output?.multiArrayValue?[0].boolValue else {
            throw SafetyCheckError.unexpectedModelOutput
        }

        return !unsafe
    }

    func resizeToRGBA(_ image: CGImage,
                      width: Int, height: Int) throws -> CGImage {

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width*4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw SafetyCheckError.imageResizeFailure
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resizedImage = context.makeImage() else {
            throw SafetyCheckError.imageResizeFailure
        }

        return resizedImage
    }

    func getRGBPlanes(of rgbaImage: CGImage) throws -> PixelBufferP8x3 {
        // Reference as interleaved 8 bit vImage PixelBuffer
        var emptyFormat = vImage_CGImageFormat()
        guard let bufferI8x4 = try? PixelBufferI8x4(
            cgImage: rgbaImage,
            cgImageFormat:&emptyFormat) else {
            throw SafetyCheckError.imageToFloatFailure
        }

        // Drop the alpha channel, keeping RGB
        let bufferI8x3 = PixelBufferI8x3(width: rgbaImage.width, height:rgbaImage.height)
        bufferI8x4.convert(to: bufferI8x3, channelOrdering: .RGBA)

        // De-interleave into 8-bit planes
        return PixelBufferP8x3(interleavedBuffer: bufferI8x3)
    }

    func normalizeToFloatShapedArray(_ bufferP8x3: PixelBufferP8x3) -> MLShapedArray<Float32> {
        let width = bufferP8x3.width
        let height = bufferP8x3.height

        let means = [0.485, 0.456, 0.406] as [Float]
        let stds  = [0.229, 0.224, 0.225] as [Float]

        // Convert to normalized float 1x3xWxH input (plannar)
        let arrayPFx3 = MLShapedArray<Float32>(repeating: 0.0, shape: [1, 3, width, height])
        for c in 0..<3 {
            arrayPFx3[0][c].withUnsafeShapedBufferPointer { ptr, _, strides in
                let floatChannel = PixelBufferPFx1(data: .init(mutating: ptr.baseAddress!),
                                                   width: width, height: height,
                                                   byteCountPerRow: strides[0]*4)

                bufferP8x3.withUnsafePixelBuffer(at: c) { uint8Channel in
                    uint8Channel.convert(to: floatChannel) // maps [0 255] -> [0 1]
                    floatChannel.multiply(by: 1.0/stds[c],
                                          preBias: -means[c],
                                          postBias: 0.0,
                                          destination: floatChannel)
                }
            }
        }
        return arrayPFx3
    }
}
