// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import XCTest
import CoreML
@testable import StableDiffusion

@available(iOS 16.2, macOS 14.0, *)
final class EulerAncestralDiscreteSchedulerTests: XCTestCase {
    func testScal() throws {
        let scheduler = EulerAncestralDiscreteScheduler(randomSource: TorchRandomSource(seed: 0), stepCount: 25)
        let scaledSample = scheduler.scaleModelInput(
            sample: MLShapedArray(
                scalars: [0.1, 0.2, 0.3, 0.4, 0.5],
                shape: [5]
            ),
            timeStep: 960)
        XCTAssertTrue(true)
    }
    
    func testLinspaceStep() throws {
        let scheduler = EulerAncestralDiscreteScheduler(randomSource: TorchRandomSource(seed: 0), stepCount: 25)
        let preSample = scheduler.step(
            output: MLShapedArray<Float32>(repeating: 0.5, shape: [5]),
            timeStep: 960,
            sample: MLShapedArray<Float32>(repeating: 0.5, shape: [5])
        )
        XCTAssertEqual(
            preSample,
            MLShapedArray(
                scalars: [18.519714, -15.129302, -49.71263, 0.6798735, -29.640398],
                shape: [5]
            )
        )
    }
}
