// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. and The HuggingFace Team. All Rights Reserved.

import Accelerate
import CoreML

/// How to space timesteps for inference
public enum TimeStepSpacing {
    case linspace
    case leading
    case karras
}

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers DPMSolverMultistepScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)
///
/// It uses the DPM-Solver++ algorithm: [code](https://github.com/LuChengTHU/dpm-solver) [paper](https://arxiv.org/abs/2211.01095).
/// Limitations:
///  - Only implemented for DPM-Solver++ algorithm (not DPM-Solver).
///  - Second order only.
///  - Assumes the model predicts epsilon.
///  - No dynamic thresholding.
///  - `midpoint` solver algorithm.
@available(iOS 16.2, macOS 13.1, *)
public final class DPMSolverMultistepScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Int]

    public let alpha_t: [Float]
    public let sigma_t: [Float]
    public let lambda_t: [Float]
    
    public let solverOrder = 2
    private(set) var lowerOrderStepped = 0
    
    private var usingKarrasSigmas = false

    /// Whether to use lower-order solvers in the final steps. Only valid for less than 15 inference steps.
    /// We empirically find this trick can stabilize the sampling of DPM-Solver, especially with 10 or fewer steps.
    public let useLowerOrderFinal = true
    
    // Stores solverOrder (2) items
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    /// Create a scheduler that uses a second order DPM-Solver++ algorithm.
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    ///   - timeStepSpacing: How to space time steps
    /// - Returns: A scheduler ready for its first step
    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        timeStepSpacing: TimeStepSpacing = .linspace
    ) {
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

        switch timeStepSpacing {
        case .linspace:
            self.timeSteps = linspace(0, Float(self.trainStepCount-1), stepCount+1).dropFirst().reversed().map { Int(round($0)) }
            self.alpha_t = vForce.sqrt(self.alphasCumProd)
            self.sigma_t = vForce.sqrt(vDSP.subtract([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd))
        case .leading:
            let lastTimeStep = trainStepCount - 1
            let stepRatio = lastTimeStep / (stepCount + 1)
            // Creates integer timesteps by multiplying by ratio
            self.timeSteps = (0...stepCount).map { 1 + $0 * stepRatio }.dropFirst().reversed()
            self.alpha_t = vForce.sqrt(self.alphasCumProd)
            self.sigma_t = vForce.sqrt(vDSP.subtract([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd))
        case .karras:
            // sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            let scaled = vDSP.multiply(
                subtraction: ([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd),
                subtraction: (vDSP.divide(1, self.alphasCumProd), [Float](repeating: 0, count: self.alphasCumProd.count))
            )
            let sigmas = vForce.sqrt(scaled)
            let logSigmas = sigmas.map { log($0) }

            let sigmaMin = sigmas.first!
            let sigmaMax = sigmas.last!
            let rho: Float = 7
            let ramp = linspace(0, 1, stepCount)
            let minInvRho = pow(sigmaMin, (1 / rho))
            let maxInvRho = pow(sigmaMax, (1 / rho))

            var karrasSigmas = ramp.map { pow(maxInvRho + $0 * (minInvRho - maxInvRho), rho) }
            let karrasTimeSteps = karrasSigmas.map { sigmaToTimestep(sigma: $0, logSigmas: logSigmas) }
            self.timeSteps = karrasTimeSteps

            karrasSigmas.append(karrasSigmas.last!)

            self.alpha_t = vDSP.divide(1, vForce.sqrt(vDSP.add(1, vDSP.square(karrasSigmas))))
            self.sigma_t = vDSP.multiply(karrasSigmas, self.alpha_t)
            usingKarrasSigmas = true
        }

        self.lambda_t = zip(self.alpha_t, self.sigma_t).map { α, σ in log(α) - log(σ) }
    }
    
    func timestepToIndex(_ timestep: Int) -> Int {
        guard usingKarrasSigmas else { return timestep }
        return self.timeSteps.firstIndex(of: timestep) ?? 0
    }
    
    /// Convert the model output to the corresponding type the algorithm needs.
    /// This implementation is for second-order DPM-Solver++ assuming epsilon prediction.
    func convertModelOutput(modelOutput: MLShapedArray<Float32>, timestep: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        assert(modelOutput.scalarCount == sample.scalarCount)
        let scalarCount = modelOutput.scalarCount
        let sigmaIndex = timestepToIndex(timestep)
        let (alpha_t, sigma_t) = (self.alpha_t[sigmaIndex], self.sigma_t[sigmaIndex])

        return MLShapedArray(unsafeUninitializedShape: modelOutput.shape) { scalars, _ in
            assert(scalars.count == scalarCount)
            modelOutput.withUnsafeShapedBufferPointer { modelOutput, _, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    for i in 0 ..< scalarCount {
                        scalars.initializeElement(at: i, to: (sample[i] - modelOutput[i] * sigma_t) / alpha_t)
                    }
                }
            }
        }
    }

    /// One step for the first-order DPM-Solver (equivalent to DDIM).
    /// See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func firstOrderUpdate(
        modelOutput: MLShapedArray<Float32>,
        timestep: Int,
        prevTimestep: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let prevIndex = timestepToIndex(prevTimestep)
        let currIndex = timestepToIndex(timestep)
        let (p_lambda_t, lambda_s) = (Double(lambda_t[prevIndex]), Double(lambda_t[currIndex]))
        let p_alpha_t = Double(alpha_t[prevIndex])
        let (p_sigma_t, sigma_s) = (Double(sigma_t[prevIndex]), Double(sigma_t[currIndex]))
        let h = p_lambda_t - lambda_s
        // x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        let x_t = weightedSum(
            [p_sigma_t / sigma_s, -p_alpha_t * (exp(-h) - 1)],
            [sample, modelOutput]
        )
        return x_t
    }

    /// One step for the second-order multistep DPM-Solver++ algorithm, using the midpoint method.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func secondOrderUpdate(
        modelOutputs: [MLShapedArray<Float32>],
        timesteps: [Int],
        prevTimestep t: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let (s0, s1) = (timesteps[back: 1], timesteps[back: 2])
        let (m0, m1) = (modelOutputs[back: 1], modelOutputs[back: 2])
        let (p_lambda_t, lambda_s0, lambda_s1) = (
            Double(lambda_t[timestepToIndex(t)]),
            Double(lambda_t[timestepToIndex(s0)]),
            Double(lambda_t[timestepToIndex(s1)])
        )
        let p_alpha_t = Double(alpha_t[timestepToIndex(t)])
        let (p_sigma_t, sigma_s0) = (Double(sigma_t[timestepToIndex(t)]), Double(sigma_t[timestepToIndex(s0)]))
        let (h, h_0) = (p_lambda_t - lambda_s0, lambda_s0 - lambda_s1)
        let r0 = h_0 / h
        let D0 = m0
        
        // D1 = (1.0 / r0) * (m0 - m1)
        let D1 = weightedSum(
            [1/r0, -1/r0],
            [m0, m1]
        )
        
        // See https://arxiv.org/abs/2211.01095 for detailed derivations
        // x_t = (
        //     (sigma_t / sigma_s0) * sample
        //     - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        //     - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        // )
        let x_t = weightedSum(
            [p_sigma_t/sigma_s0, -p_alpha_t * (exp(-h) - 1), -0.5 * p_alpha_t * (exp(-h) - 1)],
            [sample, D0, D1]
        )
        return x_t
    }

    public func step(output: MLShapedArray<Float32>, timeStep t: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let prevTimestep = stepIndex == timeSteps.count - 1 ? 0 : timeSteps[stepIndex + 1]

        let lowerOrderFinal = useLowerOrderFinal && stepIndex == timeSteps.count - 1 && timeSteps.count < 15
        let lowerOrderSecond = useLowerOrderFinal && stepIndex == timeSteps.count - 2 && timeSteps.count < 15
        let lowerOrder = lowerOrderStepped < 1 || lowerOrderFinal || lowerOrderSecond
        
        let modelOutput = convertModelOutput(modelOutput: output, timestep: t, sample: sample)
        if modelOutputs.count == solverOrder { modelOutputs.removeFirst() }
        modelOutputs.append(modelOutput)
        
        let prevSample: MLShapedArray<Float32>
        if lowerOrder {
            prevSample = firstOrderUpdate(modelOutput: modelOutput, timestep: t, prevTimestep: prevTimestep, sample: sample)
        } else {
            prevSample = secondOrderUpdate(
                modelOutputs: modelOutputs,
                timesteps: [timeSteps[stepIndex - 1], t],
                prevTimestep: prevTimestep,
                sample: sample
            )
        }
        if lowerOrderStepped < solverOrder {
            lowerOrderStepped += 1
        }
        
        return prevSample
    }
}

func sigmaToTimestep(sigma: Float, logSigmas: [Float]) -> Int {
    let logSigma = log(sigma)
    let dists = logSigmas.map { logSigma - $0 }

    // last index that is not negative, clipped to last index - 1
    var lowIndex = dists.reduce(-1) { partialResult, dist in
        return dist >= 0 && partialResult < dists.endIndex-2 ? partialResult + 1 : partialResult
    }
    lowIndex = max(lowIndex, 0)
    let highIndex = lowIndex + 1

    let low = logSigmas[lowIndex]
    let high = logSigmas[highIndex]

    // Interpolate sigmas
    let w = ((low - logSigma) / (low - high)).clipped(to: 0...1)

    // transform interpolated value to time range
    let t = (1 - w) * Float(lowIndex) + w * Float(highIndex)
    return Int(round(t))
}

extension FloatingPoint {
    func clipped(to range: ClosedRange<Self>) -> Self {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}
