// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. and The HuggingFace Team. All Rights Reserved.

import Accelerate
import CoreML

/// A Scheduler used to compute a de-noised image
///
/// This inplementation matches:
/// [Hugging Face Diffusers EulerAncestralDiscreteScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py)
///
/// It is based on the [original k-diffusion implementation by Katherine Crowson](https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L72)
/// Limitations:
///  - Only implemented for Euler A algorithm (not Euler)
///  - Assumes model predicts epsilon
@available(iOS 16.2, macOS 13.1, *)
public final class EulerAncestralDiscreteScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let timeSteps: [Int]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let sigmas: [Float]
    public let initNoiseSigma: Float
    private(set) var randomSource: RandomSource
        
    public init(
        randomSource: RandomSource,
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .linear,
        betaStart: Float = 0.0001,
        betaEnd: Float = 0.02
    ) {
        self.randomSource = randomSource
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        
        switch betaSchedule {
        case .linear:
            self.betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map({ $0 * $0 })
        }
        
        self.alphas = betas.map({ 1.0 - $0 })
        
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i -  1]
        }
        self.alphasCumProd = alphasCumProd
        
        var sigmas = vForce.sqrt(vDSP.divide(vDSP.subtract([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd), self.alphasCumProd))
        sigmas.reverse()
        sigmas.append(0)
        self.sigmas = sigmas
      
        self.initNoiseSigma = sigmas.max()!
        
        self.timeSteps = linspace(0, Float(trainStepCount - 1), trainStepCount).reversed().map { Int(round($0)) }
    }
    
    public func step(output: MLShapedArray<Float32>, timeStep t: Int, sample s: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: t)!
        let sigma = sigmas[stepIndex]
        
        // compute predicted original sample (x0) from sigma-scaled predicted noise (for epsilon):
        // sample - sigma * output
        let predOriginalSample = weightedSum([1.0, Double(-1.0 * sigma)], [s, output])
        
        let sigmaFrom = sigmas[stepIndex]
        let sigmaTo = sigmas[stepIndex + 1]
        let sigmaUp = sqrt(pow(sigmaTo, 2) * (pow(sigmaFrom, 2) - pow(sigmaTo, 2)) / pow(sigmaFrom, 2))
        let sigmaDown = sqrt(pow(sigmaTo, 2) - pow(sigmaUp, 2))
        
        // Convert to an ODE derivative:
        // derivative = (sample - predOriginalSample) / sigma
        // prevSample = sample + derivative * dt
        let derivative = weightedSum([Double(1 / sigma), Double(-1 / sigma)], [s, predOriginalSample])
        let dt = sigmaDown - sigma
        let prevSample = weightedSum([1.0, Double(dt)], [s, derivative])
        
        // Introduce noise
        let noise = MLShapedArray<Float32>(converting: randomSource.normalShapedArray(output.shape, mean: 0.0, stdev: Double(initNoiseSigma)))
        
        return weightedSum([1, Double(sigmaUp)], [prevSample, noise])  // output = prevSample + noise * sigmaUp
    }
}
