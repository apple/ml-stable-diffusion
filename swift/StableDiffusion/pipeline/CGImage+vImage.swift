// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import Accelerate
import CoreML

@available(iOS 16.0, macOS 13.0, *)
extension CGImage {
    
    typealias PixelBufferPFx1 = vImage.PixelBuffer<vImage.PlanarF>
    typealias PixelBufferP8x3 = vImage.PixelBuffer<vImage.Planar8x3>
    typealias PixelBufferIFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
    typealias PixelBufferI8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>
    
    public enum ShapedArrayError: String, Swift.Error {
        case wrongNumberOfChannels
        case incorrectFormatsConvertingToShapedArray
        case vImageConverterNotInitialized
    }
    
    public static func fromShapedArray(_ array: MLShapedArray<Float32>) throws -> CGImage {
        
        // array is [N,C,H,W], where C==3
        let channelCount = array.shape[1]
        guard channelCount == 3 else {
            throw ShapedArrayError.wrongNumberOfChannels
        }
        
        let height = array.shape[2]
        let width = array.shape[3]

        // Normalize each channel into a float between 0 and 1.0
        let floatChannels = (0..<channelCount).map { i in

            // Normalized channel output
            let cOut = PixelBufferPFx1(width: width, height:height)

            // Reference this channel in the array and normalize
            array[0][i].withUnsafeShapedBufferPointer { ptr, _, strides in
                let cIn = PixelBufferPFx1(data: .init(mutating: ptr.baseAddress!),
                                          width: width, height: height,
                                          byteCountPerRow: strides[0]*4)
                // Map [-1.0 1.0] -> [0.0 1.0]
                cIn.multiply(by: 0.5, preBias: 1.0, postBias: 0.0, destination: cOut)
            }
            return cOut
        }

        // Convert to interleaved and then to UInt8
        let floatImage = PixelBufferIFx3(planarBuffers: floatChannels)
        let uint8Image = PixelBufferI8x3(width: width, height: height)
        floatImage.convert(to:uint8Image) // maps [0.0 1.0] -> [0 255] and clips

        // Convert to uint8x3 to RGB CGImage (no alpha)
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let cgImage = uint8Image.makeCGImage(cgImageFormat:
                .init(bitsPerComponent: 8,
                      bitsPerPixel: 3*8,
                      colorSpace: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: bitmapInfo)!)!

        return cgImage
    }
    
    public var plannerRGBShapedArray: MLShapedArray<Float32> {
        get throws {
            guard
                var sourceFormat = vImage_CGImageFormat(cgImage: self),
                var mediumFormat = vImage_CGImageFormat(
                    bitsPerComponent: 8 * MemoryLayout<UInt8>.size,
                    bitsPerPixel: 8 * MemoryLayout<UInt8>.size * 4,
                    colorSpace: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue)),
                let width = vImagePixelCount(exactly: self.width),
                let height = vImagePixelCount(exactly: self.height)
            else {
                throw ShapedArrayError.incorrectFormatsConvertingToShapedArray
            }
            
            var sourceImageBuffer = try vImage_Buffer(cgImage: self)
            
            var mediumDesination = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: mediumFormat.bitsPerPixel)
            
            let converter = vImageConverter_CreateWithCGImageFormat(
                &sourceFormat,
                &mediumFormat,
                nil,
                vImage_Flags(kvImagePrintDiagnosticsToConsole),
                nil)
            
            guard let converter = converter?.takeRetainedValue() else {
                throw ShapedArrayError.vImageConverterNotInitialized
            }
            
            vImageConvert_AnyToAny(converter, &sourceImageBuffer, &mediumDesination, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole))
            
            var destinationA = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
            var destinationR = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
            var destinationG = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
            var destinationB = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
            
            var minFloat: [Float] = [-1.0, -1.0, -1.0, -1.0]
            var maxFloat: [Float] = [1.0, 1.0, 1.0, 1.0]
            
            vImageConvert_ARGB8888toPlanarF(&mediumDesination, &destinationA, &destinationR, &destinationG, &destinationB, &maxFloat, &minFloat, .zero)
           
            let redData = Data(bytes: destinationR.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
            let greenData = Data(bytes: destinationG.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
            let blueData = Data(bytes: destinationB.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
            
            let imageData = redData + greenData + blueData

            let shapedArray = MLShapedArray<Float32>(data: imageData, shape: [1, 3, self.width, self.height])
            
            return shapedArray
        }
    }
}

