// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// A random source consistent with PyTorch
///
///  This implementation matches:
///  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/DistributionsHelper.h
///  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/DistributionTemplates.h
///  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/DistributionKernels.cpp
///  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TransformationHelper.h
///
@available(iOS 16.2, macOS 13.1, *)
struct TorchRandomSource: RandomNumberGenerator, RandomSource {

  struct State {
    var key = [UInt32](repeating: 0, count: 624)
    var pos: Int = 0
    var nextGauss: Double? = nil
  }

  var state: State

  /// Initialize with a random seed
  ///
  /// - Parameters
  ///     - seed: Seed for underlying Mersenne Twister 19937 generator
  /// - Returns random source
  init(seed: UInt32) {
    state = .init()
    var s = seed & 0xffff_ffff
    for i in 0..<state.key.count {
      state.key[i] = s
      s = UInt32((UInt64(1_812_433_253) * UInt64(s ^ (s >> 30)) + UInt64(i) + 1) & 0xffff_ffff)
    }
    state.pos = state.key.count
    state.nextGauss = nil
  }

  /// Generate next UInt32 using fast 32bit Mersenne Twister
  mutating func nextUInt32() -> UInt32 {
    let n = 624
    let m = 397
    let matrixA: UInt64 = 0x9908_b0df
    let upperMask: UInt32 = 0x8000_0000
    let lowerMask: UInt32 = 0x7fff_ffff

    var y: UInt32
    if state.pos == state.key.count {
      for i in 0..<(n - m) {
        y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
        state.key[i] = state.key[i + m] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      }
      for i in (n - m)..<(n - 1) {
        y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
        state.key[i] = state.key[i + (m - n)] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      }
      y = (state.key[n - 1] & upperMask) | (state.key[0] & lowerMask)
      state.key[n - 1] = state.key[m - 1] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      state.pos = 0
    }
    y = state.key[state.pos]
    state.pos += 1

    y ^= (y >> 11)
    y ^= (y << 7) & 0x9d2c_5680
    y ^= (y << 15) & 0xefc6_0000
    y ^= (y >> 18)

    return y
  }

  mutating func next() -> UInt64 {
    let high = nextUInt32()
    let low = nextUInt32()
    return (UInt64(high) << 32) | UInt64(low)
  }

  /// Generate next random double value
  mutating func nextDouble() -> Double {
    let a = next()
    return Double(a & 9_007_199_254_740_991) * (1.0 / 9007199254740992.0)
  }

  /// Generate next random float value
  mutating func nextFloat() -> Float {
    let a = nextUInt32()
    return Float(a & 16_777_215) * (1.0 / 16777216.0)
  }

  /// Generate next random value from a standard normal
  mutating func nextGauss() -> Double {
    if let nextGauss = state.nextGauss {
      state.nextGauss = nil
      return nextGauss
    }
    // Box-Muller transform
    let u1: Double = nextDouble()
    let u2: Double = 1 - nextDouble()
    let radius = sqrt(-2.0 * log(u2))
    let theta = 2.0 * .pi * u1
    state.nextGauss = radius * sin(theta)
    return radius * cos(theta)
  }

  /// Generates an array of random values from a normal distribution with given mean and standard deviation.
  /// This simulates torch.randn([1, 4, 64, 64], dtype=torch.float), note that for dtype=torch.double, it
  /// will be slightly different.
  mutating func normalArray(count: Int, mean: Double = 0.0, stdev: Double = 1.0) -> [Double] {
    // If it is smaller than 16 elements, Torch generates from Box-Muller transform directly.
    // Note that even if this is used to generate Float, it will use Double underneath.
    guard count >= 16 else {
      return (0..<count).map { _ in nextGauss() * stdev + mean }
    }
    // Otherwise, Torch first fill a uniform distribution array, then do Box-Muller
    // transformation over this array.
    var data = (0..<count).map { _ in Double(nextFloat()) }
    for i in stride(from: 0, to: count - 15, by: 16) {
      for j in 0..<8 {
        let u1 = 1 - data[i + j]
        let u2 = data[i + j + 8]
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * .pi * u2
        data[i + j] = radius * cos(theta) * stdev + mean
        data[i + j + 8] = radius * sin(theta) * stdev + mean
      }
    }
    if count % 16 != 0 {
      for i in (count - 16)..<count {
        data[i] = nextDouble()
      }
      let i = count - 16
      for j in 0..<8 {
        let u1 = 1 - data[i + j]
        let u2 = data[i + j + 8]
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * .pi * u2
        data[i + j] = radius * cos(theta) * stdev + mean
        data[i + j + 8] = radius * sin(theta) * stdev + mean
      }
    }
    return data
  }

  /// Generate a shaped array with scalars from a normal distribution with given mean and standard deviation.
  mutating func normalShapedArray(_ shape: [Int], mean: Double = 0.0, stdev: Double = 1.0) -> MLShapedArray<Double> {
    let count = shape.reduce(1, *)
    return .init(scalars: normalArray(count: count, mean: mean, stdev: stdev), shape: shape)
  }
}
