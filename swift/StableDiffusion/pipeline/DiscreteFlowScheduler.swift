// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.

import CoreML

/// A scheduler used to compute a de-noised image
@available(iOS 16.2, macOS 13.1, *)
public final class DiscreteFlowScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public var timeSteps = [Int]()
    public var betas = [Float]()
    public var alphas = [Float]()
    public var alphasCumProd = [Float]()

    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    var trainSteps: Float
    var shift: Float
    var counter: Int
    var sigmas = [Float]()

    /// Create a scheduler that uses a second order DPM-Solver++ algorithm.
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - timeStepShift: Amount to shift the timestep schedule
    /// - Returns: A scheduler ready for its first step
    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        timeStepShift: Float = 3.0
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.trainSteps = Float(trainStepCount)
        self.shift = timeStepShift
        self.counter = 0

        let sigmaDistribution = linspace(1, trainSteps, Int(trainSteps)).map { sigmaFromTimestep($0) }
        let timeStepDistribution = linspace(sigmaDistribution.first!, sigmaDistribution.last!, stepCount).reversed()
        self.timeSteps = timeStepDistribution.map { Int($0 * trainSteps) }
        self.sigmas = timeStepDistribution.map { sigmaFromTimestep($0 * trainSteps) }
    }

    func sigmaFromTimestep(_ timestep: Float) -> Float {
        if shift == 1.0 {
            return timestep / trainSteps
        } else {
            // shift * timestep / (1 + (shift - 1) * timestep)
            let t = timestep / trainSteps
            return shift * t / (1 + (shift - 1) * t)
        }
    }

    func timestepsFromSigmas() -> [Float] {
        return sigmas.map { $0 * trainSteps }
    }

    /// Convert the model output to the corresponding type the algorithm needs.
    func convertModelOutput(modelOutput: MLShapedArray<Float32>, timestep: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        assert(modelOutput.scalarCount == sample.scalarCount)
        let stepIndex = timeSteps.firstIndex(of: timestep) ?? counter
        let sigma = sigmas[stepIndex]

        return MLShapedArray<Float>(unsafeUninitializedShape: modelOutput.shape) { result, _ in
            modelOutput.withUnsafeShapedBufferPointer { noiseScalars, _, _ in
                sample.withUnsafeShapedBufferPointer { latentScalars, _, _ in
                    for i in 0..<result.count {
                        let denoised = latentScalars[i] - noiseScalars[i] * sigma
                        result.initializeElement(
                            at: i,
                            to: denoised
                        )
                    }
                }
            }
        }
    }

    public func calculateTimestepsFromSigmas(strength: Float?) -> [Float] {
        guard let strength else { return timestepsFromSigmas() }
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        let actualTimesteps = Array(timestepsFromSigmas()[startStep...])
        return actualTimesteps
    }

    public func step(output: MLShapedArray<Float32>, timeStep t: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: t) ?? counter // TODO: allow float timesteps in scheduler step protocol
        let modelOutput = convertModelOutput(modelOutput: output, timestep: t, sample: sample)
        modelOutputs.append(modelOutput)

        let sigma = sigmas[stepIndex]
        var dt = sigma
        var prevSigma: Float = 0
        if stepIndex < sigmas.count - 1 {
            prevSigma = sigmas[stepIndex + 1]
            dt = prevSigma - sigma
        }

        let prevSample: MLShapedArray<Float32> = MLShapedArray<Float>(unsafeUninitializedShape: modelOutput.shape) { result, _ in
            modelOutput.withUnsafeShapedBufferPointer { noiseScalars, _, _ in
                sample.withUnsafeShapedBufferPointer { latentScalars, _, _ in
                    for i in 0..<result.count {
                        let denoised = noiseScalars[i]
                        let x = latentScalars[i]

                        let d = (x - denoised) / sigma
                        let prev_x = x + d * dt
                        result.initializeElement(
                            at: i,
                            to: prev_x
                        )
                    }
                }
            }
        }

        counter += 1
        return prevSample
    }
}
