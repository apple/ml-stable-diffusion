import Accelerate
import CoreML

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
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    public init(
        randomSource: RandomSource,
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaStart: Float = 0.0001,
        betaEnd: Float = 0.02,
        betaSchedule: BetaSchedule = .scaledLinear,
        timestepSpacing: TimestepSpacing = .leading,
        stepsOffset: Int = 1
    ) {
        self.randomSource = randomSource
        self.trainStepCount = trainStepCount
        inferenceStepCount = stepCount

        switch betaSchedule {
        case .linear:
            betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            betas = linspace(sqrt(betaStart), sqrt(betaEnd), trainStepCount).map { $0 * $0 }
        }

        alphas = betas.map { 1.0 - $0 }
        var alphasCumProd = alphas
        for i in 1 ..< alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i - 1]
        }
        self.alphasCumProd = alphasCumProd

        var sigmas = vForce.sqrt(
            vDSP.divide(
                vDSP.subtract(
                    [Float](repeating: 1, count: alphasCumProd.count),
                    alphasCumProd
                ),
                alphasCumProd
            )
        )

        var timeSteps = [Float](repeating: 0.0, count: stepCount)
        switch timestepSpacing {
        case .linspace:
            timeSteps = linspace(0, Float(trainStepCount - 1), stepCount)
        case .leading:
            let stepRatio = trainStepCount / stepCount
            timeSteps = (0 ..< stepCount).map { Float($0 * stepRatio + stepsOffset) }
        case .trailing:
            let stepRatio = trainStepCount / stepCount
            timeSteps = (1 ... stepCount).map { Float($0 * stepRatio - 1) }
        }
        timeSteps.reverse()
        var sigmasInt = [Float](repeating: 0.0, count: timeSteps.count)
        vDSP_vlint(&sigmas, &timeSteps, vDSP_Stride(1), &sigmasInt, vDSP_Stride(1), vDSP_Length(timeSteps.count), vDSP_Length(sigmas.count))
        sigmasInt.append(0.0)
        initNoiseSigma = sigmasInt.max()!
        self.timeSteps = timeSteps.map { Int($0) }
        self.sigmas = sigmasInt
    }

    public func step(
        output: MLShapedArray<Float32>,
        timeStep: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: timeStep) ?? timeSteps.count - 1
        let sigma = sigmas[stepIndex]
        let predOriginalSample = weightedSum([1.0, Double(-1.0 * sigma)], [sample, output])

        let sigmaFrom = sigmas[stepIndex]
        let sigmaTo = sigmas[stepIndex + 1]
        let sigmaUp = sqrt(pow(sigmaTo, 2.0) * (pow(sigmaFrom, 2.0) - pow(sigmaTo, 2.0)) / pow(sigmaFrom, 2.0))
        let sigmaDown = sqrt(pow(sigmaTo, 2.0) - pow(sigmaUp, 2.0))

        // Convert to an ODE derivative:
        let derivative = weightedSum([Double(1.0 / sigma), Double(-1.0 / sigma)], [sample, predOriginalSample])
        let dt = sigmaDown - sigma
        let prevSample = weightedSum([1.0, Double(dt)], [sample, derivative])
        let noise = MLShapedArray<Float32>(
            converting: randomSource.normalShapedArray(
                output.shape,
                mean: 0.0,
                stdev: 1.0
            )
        )
        return weightedSum([1.0, Double(sigmaUp)], [prevSample, noise])
    }

    public func scaleModelInput(
        sample: MLShapedArray<Float32>,
        timeStep: Int
    ) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: timeStep) ?? timeSteps.count - 1
        let sigma = sigmas[stepIndex]
        let scale = sqrt(pow(sigma, 2.0) + 1.0)
        let scalarCount = sample.scalarCount
        return MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
            assert(scalars.count == scalarCount)
            sample.withUnsafeShapedBufferPointer { sample, _, _ in
                for i in 0 ..< scalarCount {
                    scalars.initializeElement(at: i, to: sample[i] / scale)
                }
            }
        }
    }
}
