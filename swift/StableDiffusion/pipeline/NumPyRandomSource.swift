// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// A random source consistent with NumPy
///
///  This implementation matches:
///  [NumPy's older randomkit.c](https://github.com/numpy/numpy/blob/v1.0/numpy/random/mtrand/randomkit.c)
///
@available(iOS 16.2, macOS 13.1, *)
struct NumPyRandomSource: RandomNumberGenerator, RandomSource {

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
        var s = seed & 0xffffffff
        for i in 0 ..< state.key.count {
            state.key[i] = s
            s = UInt32((UInt64(1812433253) * UInt64(s ^ (s >> 30)) + UInt64(i) + 1) & 0xffffffff)
        }
        state.pos = state.key.count
        state.nextGauss = nil
    }

    /// Generate next UInt32 using fast 32bit Mersenne Twister
    mutating func nextUInt32() -> UInt32 {
        let n = 624
        let m = 397
        let matrixA: UInt64    = 0x9908b0df
        let upperMask: UInt32  = 0x80000000
        let lowerMask: UInt32  = 0x7fffffff

        var y: UInt32
        if state.pos == state.key.count {
            for i in 0 ..< (n - m) {
                y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
                state.key[i] = state.key[i + m] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
            }
            for i in (n - m) ..< (n - 1) {
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
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)

        return y
    }

    mutating func next() -> UInt64 {
        let low = nextUInt32()
        let high = nextUInt32()
        return (UInt64(high) << 32) | UInt64(low)
    }

    /// Generate next random double value
    mutating func nextDouble() -> Double {
        let a = Double(nextUInt32() >> 5)
        let b = Double(nextUInt32() >> 6)
        return (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// Generate next random value from a standard normal
    mutating func nextGauss() -> Double {
        if let nextGauss = state.nextGauss {
            state.nextGauss = nil
            return nextGauss
        }
        var x1, x2, r2: Double
        repeat {
            x1 = 2.0 * nextDouble() - 1.0
            x2 = 2.0 * nextDouble() - 1.0
            r2 = x1 * x1 + x2 * x2
        } while r2 >= 1.0 || r2 == 0.0

        // Box-Muller transform
        let f = sqrt(-2.0 * log(r2) / r2)
        state.nextGauss = f * x1
        return f * x2
    }

    /// Generates a random value from a normal distribution with given mean and standard deviation.
    mutating func nextNormal(mean: Double = 0.0, stdev: Double = 1.0) -> Double {
        nextGauss() * stdev + mean
    }

    /// Generates an array of random values from a normal distribution with given mean and standard deviation.
    mutating func normalArray(count: Int, mean: Double = 0.0, stdev: Double = 1.0) -> [Double] {
        (0 ..< count).map { _ in nextNormal(mean: mean, stdev: stdev) }
    }

    /// Generate a shaped array with scalars from a normal distribution with given mean and standard deviation.
    mutating func normalShapedArray(_ shape: [Int], mean: Double = 0.0, stdev: Double = 1.0) -> MLShapedArray<Double> {
        let count = shape.reduce(1, *)
        return .init(scalars: normalArray(count: count, mean: mean, stdev: stdev), shape: shape)
    }
}
