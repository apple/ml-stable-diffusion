// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import Accelerate
import CoreML
import CoreGraphics

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
                      colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: bitmapInfo)!)!

        return cgImage
    }
    
    public func planarRGBShapedArray(minValue: Float, maxValue: Float)
        throws -> MLShapedArray<Float32> {
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
            
            var mediumDestination = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: mediumFormat.bitsPerPixel)
            
            let converter = vImageConverter_CreateWithCGImageFormat(
                &sourceFormat,
                &mediumFormat,
                nil,
                vImage_Flags(kvImagePrintDiagnosticsToConsole),
                nil)

            guard let converter = converter?.takeRetainedValue() else {
                throw ShapedArrayError.vImageConverterNotInitialized
            }
            
            vImageConvert_AnyToAny(converter, &sourceImageBuffer, &mediumDestination, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole))
            
            var destinationA = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
            var destinationR = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
            var destinationG = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
            var destinationB = try vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))

            var minFloat: [Float] = Array(repeating: minValue, count: 4)
            var maxFloat: [Float] = Array(repeating: maxValue, count: 4)
            
            vImageConvert_ARGB8888toPlanarF(&mediumDestination, &destinationA, &destinationR, &destinationG, &destinationB, &maxFloat, &minFloat, .zero)
           
            let destAPtr = destinationA.data.assumingMemoryBound(to: Float.self)
            let destRPtr = destinationR.data.assumingMemoryBound(to: Float.self)
            let destGPtr = destinationG.data.assumingMemoryBound(to: Float.self)
            let destBPtr = destinationB.data.assumingMemoryBound(to: Float.self)

            for i in 0..<Int(width) * Int(height) {
                if destAPtr.advanced(by: i).pointee == 0 {
                    destRPtr.advanced(by: i).pointee = -1
                    destGPtr.advanced(by: i).pointee = -1
                    destBPtr.advanced(by: i).pointee = -1
                }
            }
            
            let redData = destinationR.unpaddedData()
            let greenData = destinationG.unpaddedData()
            let blueData = destinationB.unpaddedData()

            let imageData = redData + greenData + blueData

            let shapedArray = MLShapedArray<Float32>(data: imageData, shape: [1, 3, self.height, self.width])

            return shapedArray
    }

    private func normalizePixelValues(pixel: UInt8) -> Float {
        return (Float(pixel) / 127.5) - 1.0
    }

    public func toRGBShapedArray(minValue: Float, maxValue: Float)
        throws -> MLShapedArray<Float32> {
            let image = self
            let width = image.width
            let height = image.height
            let alphaMaskValue: Float = minValue

            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 4 * width, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
                  let ptr = context.data?.bindMemory(to: UInt8.self, capacity: width * height * 4) else {
                return []
            }

            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

            var redChannel = [Float](repeating: 0, count: width * height)
            var greenChannel = [Float](repeating: 0, count: width * height)
            var blueChannel = [Float](repeating: 0, count: width * height)

            for y in 0..<height {
                for x in 0..<width {
                    let i = 4 * (y * width + x)
                    if ptr[i+3] == 0 {
                        // Alpha mask for controlnets
                        redChannel[y * width + x] = alphaMaskValue
                        greenChannel[y * width + x] = alphaMaskValue
                        blueChannel[y * width + x] = alphaMaskValue
                    } else {
                        redChannel[y * width + x] = normalizePixelValues(pixel: ptr[i])
                        greenChannel[y * width + x] = normalizePixelValues(pixel: ptr[i+1])
                        blueChannel[y * width + x] = normalizePixelValues(pixel: ptr[i+2])
                    }
                }
            }

            let colorShape = [1, 1, height, width]
            let redShapedArray = MLShapedArray<Float32>(scalars: redChannel, shape: colorShape)
            let greenShapedArray = MLShapedArray<Float32>(scalars: greenChannel, shape: colorShape)
            let blueShapedArray = MLShapedArray<Float32>(scalars: blueChannel, shape: colorShape)

            let shapedArray = MLShapedArray<Float32>(concatenating: [redShapedArray, greenShapedArray, blueShapedArray], alongAxis: 1)

            return shapedArray
    }
}

extension vImage_Buffer {
    func unpaddedData() -> Data {
        let bytesPerPixel = self.rowBytes / Int(self.width)
        let bytesPerRow = Int(self.width) * bytesPerPixel

        var contiguousPixelData = Data(capacity: bytesPerRow * Int(self.height))
        for row in 0..<Int(self.height) {
            let rowStart = self.data!.advanced(by: row * self.rowBytes)
            let rowData = Data(bytes: rowStart, count: bytesPerRow)
            contiguousPixelData.append(rowData)
        }

        return contiguousPixelData
    }
}
