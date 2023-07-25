// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import Accelerate

@available(iOS 16.2, macOS 13.1, *)
public struct ControlNet: ResourceManaging {
    
    var models: [ManagedMLModel]
    
    public init(modelAt urls: [URL],
                configuration: MLModelConfiguration) {
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
    }
    
    /// Load resources.
    public func loadResources() throws {
        for model in models {
            try model.loadResources()
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }
    
    /// Pre-warm resources
    public func prewarmResources() throws {
        // Override default to pre-warm each model
        for model in models {
            try model.loadResources()
            model.unloadResources()
        }
    }
    
    var inputImageDescriptions: [MLFeatureDescription] {
        models.map { model in
            try! model.perform {
                $0.modelDescription.inputDescriptionsByName["controlnet_cond"]!
            }
        }
    }
    
    /// The expected shape of the models image input
    public var inputImageShapes: [[Int]] {
        inputImageDescriptions.map { desc in
            desc.multiArrayConstraint!.shape.map { $0.intValue }
        }
    }
    
    /// Calculate additional inputs for Unet to generate intended image following provided images
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    ///   - images: Images for each ControlNet
    /// - Returns: Array of predicted noise residuals
    func execute(
        latents: [MLShapedArray<Float32>],
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>,
        images: [MLShapedArray<Float32>]
    ) throws -> [[String: MLShapedArray<Float32>]] {
        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray(scalars: [Float(timeStep), Float(timeStep)], shape: [2])
        
        var outputs: [[String: MLShapedArray<Float32>]] = []
        
        for (modelIndex, model) in models.enumerated() {
            let inputs = try latents.map { latent in
                let dict: [String: Any] = [
                    "sample": MLMultiArray(latent),
                    "timestep": MLMultiArray(t),
                    "encoder_hidden_states": MLMultiArray(hiddenStates),
                    "controlnet_cond": MLMultiArray(images[modelIndex])
                ]
                return try MLDictionaryFeatureProvider(dictionary: dict)
            }
            
            let batch = MLArrayBatchProvider(array: inputs)
            
            let results = try model.perform {
                try $0.predictions(fromBatch: batch)
            }
            
            // pre-allocate MLShapedArray with a specific shape in outputs
            if outputs.isEmpty {
                outputs = initOutputs(
                    batch: latents.count,
                    shapes: results.features(at: 0).featureValueDictionary
                )
            }
            
            for n in 0..<results.count {
                let result = results.features(at: n)
                for k in result.featureNames {
                    let newValue = result.featureValue(for: k)!.multiArrayValue!
                    if modelIndex == 0 {
                        outputs[n][k] = MLShapedArray<Float32>(newValue)
                    } else {
                        let outputArray = MLMultiArray(outputs[n][k]!)
                        let count = newValue.count
                        let inputPointer = newValue.dataPointer.assumingMemoryBound(to: Float.self)
                        let outputPointer = outputArray.dataPointer.assumingMemoryBound(to: Float.self)
                        vDSP_vadd(inputPointer, 1, outputPointer, 1, outputPointer, 1, vDSP_Length(count))
                    }
                }
            }
        }
        
        return outputs
    }
    
    private func initOutputs(batch: Int, shapes: [String: MLFeatureValue]) -> [[String: MLShapedArray<Float32>]] {
        var output: [String: MLShapedArray<Float32>] = [:]
        for (outputName, featureValue) in shapes {
            output[outputName] = MLShapedArray<Float32>(
                repeating: 0.0,
                shape: featureValue.multiArrayValue!.shape.map { $0.intValue }
            )
        }
        return Array(repeating: output, count: batch)
    }
}
