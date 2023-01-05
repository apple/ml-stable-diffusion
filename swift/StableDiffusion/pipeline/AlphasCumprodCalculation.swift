// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

public struct AlphasCumprodCalculation {
    public var sqrtAlphasCumprod: Float
    public var sqrtOneMinusAlphasCumprod: Float
    
    public init(
        sqrtAlphasCumprod: Float,
        sqrtOneMinusAlphasCumprod: Float
    ) {
        self.sqrtAlphasCumprod = sqrtAlphasCumprod
        self.sqrtOneMinusAlphasCumprod = sqrtOneMinusAlphasCumprod
    }
    
    public init(
        alphasCumprod: [Float],
        timesteps: Int = 1_000,
        steps: Int,
        strength: Float
    ) {
        let tEnc = Int(strength * Float(steps))
        let initTimestep = min(max(0, timesteps - timesteps / steps * (steps - tEnc) + 1), timesteps - 1)
        self.sqrtAlphasCumprod = alphasCumprod[initTimestep].squareRoot()
        self.sqrtOneMinusAlphasCumprod = (1 - alphasCumprod[initTimestep]).squareRoot()
    }
}
