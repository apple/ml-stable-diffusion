// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import ArgumentParser
import CoreGraphics
import CoreML
import Foundation
import StableDiffusion
import UniformTypeIdentifiers
import Cocoa

@available(iOS 16.2, macOS 13.1, *)
struct StableDiffusionSample: ParsableCommand {

    static let configuration = CommandConfiguration(
        abstract: "Run stable diffusion to generate images guided by a text prompt",
        version: "0.1"
    )

    @Argument(help: "Input string prompt")
    var prompt: String

    @Option(help: "Input string negative prompt")
    var negativePrompt: String = ""

    @Option(
        help: ArgumentHelp(
            "Path to stable diffusion resources.",
            discussion: "The resource directory should contain\n" +
                " - *compiled* models: {TextEncoder,Unet,VAEDecoder}.mlmodelc\n" +
                " - tokenizer info: vocab.json, merges.txt",
            valueName: "directory-path"
        )
    )
    var resourcePath: String = "./"
    
    @Option(help: "Path to starting image.")
    var image: String? = nil
    
    @Option(help: "Strength for image2image.")
    var strength: Float = 0.5

    @Option(help: "Number of images to sample / generate")
    var imageCount: Int = 1

    @Option(help: "Number of diffusion steps to perform")
    var stepCount: Int = 50

    @Option(
        help: ArgumentHelp(
            "How often to save samples at intermediate steps",
            discussion: "Set to 0 to only save the final sample"
        )
    )
    var saveEvery: Int = 0

    @Option(help: "Output path")
    var outputPath: String = "./"

    @Option(help: "Random seed")
    var seed: UInt32 = UInt32.random(in: 0...UInt32.max)

    @Option(help: "Controls the influence of the text prompt on sampling process (0=random images)")
    var guidanceScale: Float = 7.5

    @Option(help: "Compute units to load model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var computeUnits: ComputeUnits = .all

    @Option(help: "Scheduler to use, one of {pndm, dpmpp}")
    var scheduler: SchedulerOption = .pndm

    @Option(help: "Random number generator to use, one of {numpy, torch}")
    var rng: RNGOption = .numpy

    @Flag(help: "Disable safety checking")
    var disableSafety: Bool = false

    @Flag(help: "Reduce memory usage")
    var reduceMemory: Bool = false

    mutating func run() throws {
        guard FileManager.default.fileExists(atPath: resourcePath) else {
            throw RunError.resources("Resource path does not exist \(resourcePath)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.asMLComputeUnits
        let resourceURL = URL(filePath: resourcePath)

        log("Loading resources and creating pipeline\n")
        log("(Note: This can take a while the first time using these resources)\n")
        let pipeline = try StableDiffusionPipeline(resourcesAt: resourceURL,
                                                   configuration: config,
                                                   disableSafety: disableSafety,
                                                   reduceMemory: reduceMemory)
        try pipeline.loadResources()
        
        let startingImage: CGImage?
        if let image {
            let imageURL = URL(filePath: image)
            do {
                let imageData = try Data(contentsOf: imageURL)
                guard
                    let nsImage = NSImage(data: imageData),
                    let loadedImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
                else {
                    throw RunError.resources("Starting Image not available \(resourcePath)")
                }
                startingImage = loadedImage
            } catch let error {
                throw RunError.resources("Starting image not found \(imageURL), error: \(error)")
            }
            
        } else {
            startingImage = nil
        }

        log("Sampling ...\n")
        let sampleTimer = SampleTimer()
        sampleTimer.start()

        var pipelineConfig = StableDiffusionPipeline.Configuration(prompt: prompt)
        
        pipelineConfig.negativePrompt = negativePrompt
        pipelineConfig.startingImage = startingImage
        pipelineConfig.strength = strength
        pipelineConfig.imageCount = imageCount
        pipelineConfig.stepCount = stepCount
        pipelineConfig.seed = seed
        pipelineConfig.guidanceScale = guidanceScale
        pipelineConfig.schedulerType = scheduler.stableDiffusionScheduler
        pipelineConfig.rngType = rng.stableDiffusionRNG
        
        let images = try pipeline.generateImages(
            configuration: pipelineConfig,
            progressHandler: { progress in
                sampleTimer.stop()
                handleProgress(progress,sampleTimer)
                if progress.stepCount != progress.step {
                    sampleTimer.start()
                }
                return true
            })

        _ = try saveImages(images, logNames: true)
    }

    func handleProgress(
        _ progress: StableDiffusionPipeline.Progress,
        _ sampleTimer: SampleTimer
    ) {
        log("\u{1B}[1A\u{1B}[K")
        log("Step \(progress.step) of \(progress.stepCount) ")
        log(" [")
        log(String(format: "mean: %.2f, ", 1.0/sampleTimer.mean))
        log(String(format: "median: %.2f, ", 1.0/sampleTimer.median))
        log(String(format: "last %.2f", 1.0/sampleTimer.allSamples.last!))
        log("] step/sec")

        if saveEvery > 0, progress.step % saveEvery == 0 {
            let saveCount = (try? saveImages(progress.currentImages, step: progress.step)) ?? 0
            log(" saved \(saveCount) image\(saveCount != 1 ? "s" : "")")
        }
        log("\n")
    }

    func saveImages(
        _ images: [CGImage?],
        step: Int? = nil,
        logNames: Bool = false
    ) throws -> Int {
        let url = URL(filePath: outputPath)
        var saved = 0
        for i in 0 ..< images.count {

            guard let image = images[i] else {
                if logNames {
                    log("Image \(i) failed safety check and was not saved")
                }
                continue
            }

            let name = imageName(i, step: step)
            let fileURL = url.appending(path:name)

            guard let dest = CGImageDestinationCreateWithURL(fileURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                throw RunError.saving("Failed to create destination for \(fileURL)")
            }
            CGImageDestinationAddImage(dest, image, nil)
            if !CGImageDestinationFinalize(dest) {
                throw RunError.saving("Failed to save \(fileURL)")
            }
            if logNames {
                log("Saved \(name)\n")
            }
            saved += 1
        }
        return saved
    }

    func imageName(_ sample: Int, step: Int? = nil) -> String {
        let fileCharLimit = 75
        var name = prompt.prefix(fileCharLimit).replacingOccurrences(of: " ", with: "_")
        if imageCount != 1 {
            name += ".\(sample)"
        }
        
        if image != nil {
            name += ".str\(Int(strength * 100))"
        }

        name += ".\(seed)"

        if let step = step {
            name += ".\(step)"
        } else {
            name += ".final"
        }
        name += ".png"
        return name
    }

    func log(_ str: String, term: String = "") {
        print(str, terminator: term)
    }
}

enum RunError: Error {
    case resources(String)
    case saving(String)
}

@available(iOS 16.2, macOS 13.1, *)
enum ComputeUnits: String, ExpressibleByArgument, CaseIterable {
    case all, cpuAndGPU, cpuOnly, cpuAndNeuralEngine
    var asMLComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }
}

@available(iOS 16.2, macOS 13.1, *)
enum SchedulerOption: String, ExpressibleByArgument {
    case pndm, dpmpp
    var stableDiffusionScheduler: StableDiffusionScheduler {
        switch self {
        case .pndm: return .pndmScheduler
        case .dpmpp: return .dpmSolverMultistepScheduler
        }
    }
}

@available(iOS 16.2, macOS 13.1, *)
enum RNGOption: String, ExpressibleByArgument {
    case numpy, torch
    var stableDiffusionRNG: StableDiffusionRNG {
        switch self {
        case .numpy: return .numpyRNG
        case .torch: return .torchRNG
        }
    }
}

if #available(iOS 16.2, macOS 13.1, *) {
    StableDiffusionSample.main()
} else {
    print("Unsupported OS")
}
