// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

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
        
        var currentOutputs: [[String: MLMultiArray]] = []
        
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
            
            for n in 0..<results.count {
                let result = results.features(at: n)
                if currentOutputs.count < results.count {
                    let initOutput = result.featureNames.reduce(into: [String: MLMultiArray]()) { output, k in
                        output[k] = MLMultiArray(
                            concatenating: [result.featureValue(for: k)!.multiArrayValue!],
                            axis: 0,
                            dataType: .float32
                        )
                    }
                    currentOutputs.append(initOutput)
                } else {
                    var currentOutput = currentOutputs[n]
                    for k in result.featureNames {
                        let newValue = result.featureValue(for: k)!
                        let currentValue = currentOutput[k]!
                        let multiArray = MLMultiArray(
                            concatenating: [newValue.multiArrayValue!],
                            axis: 0,
                            dataType: .float32
                        )
                        for i in 0..<multiArray.count {
                            currentValue[i] = NSNumber(value: currentValue[i].floatValue + multiArray[i].floatValue)
                        }
                        currentOutput[k] = currentValue
                    }
                    currentOutputs[n] = currentOutput
                }
            }
        }
        
        var outputs: [[String: MLShapedArray<Float32>]] = []
        let batchCount = currentOutputs.count
        
        for _ in 0..<batchCount {
            let output = currentOutputs.remove(at: 0)
            var newOutput: [String: MLShapedArray<Float32>] = [:]
            for (k, v) in output {
                newOutput[k] = MLShapedArray(v)
            }
            outputs.append(newOutput)
        }
        
        return outputs
    }
}
