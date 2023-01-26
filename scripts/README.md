# Download models and run inference

## Download models

Modify `scripts/download_model.py` to choose the model you'd like to download.
Then run it from the command line: `python scripts/download_model.py`

Check the `models/<output>` folder to see the model. If it is a https://huggingface.co/coreml model (models converted to Core ML), then unzip it first.

## Run inference

```shell
# MODEL=coreml-stable-diffusion-v1-4_original_compiled
# MODEL=coreml-stable-diffusion-v1-5_original_compiled
MODEL=coreml-stable-diffusion-2-1-base_original
# MODEL=coreml-stable-diffusion-2-1-base_split_einsum
# COMPUTE_UNITS=all # "split_einsum" models
COMPUTE_UNITS=cpuAndGPU # on "original" models
OUTPUT_PATH=output_images/$MODEL
mkdir -p $OUTPUT_PATH

PROMPT="a photograph of an astronaut riding on a horse"
SEED=42 # 93 is the default
echo "Generating \"$PROMPT\" on $MODEL with seed $SEED"
time swift run StableDiffusionSample $PROMPT --resource-path models/$MODEL --compute-units $COMPUTE_UNITS --output-path $OUTPUT_PATH --seed $SEED
```

Available commands:

```shell
swift run StableDiffusionSample --help
# USAGE: stable-diffusion-sample [<options>] <prompt>

# ARGUMENTS:
#   <prompt>                Input string prompt

# OPTIONS:
#   --negative-prompt <negative-prompt>
#                           Input string negative prompt
#   --resource-path <directory-path>
#                           Path to stable diffusion resources. (default: ./)
#         The resource directory should contain
#          - *compiled* models: {TextEncoder,Unet,VAEDecoder}.mlmodelc
#          - tokenizer info: vocab.json, merges.txt
#   --image-count <image-count>
#                           Number of images to sample / generate (default: 1)
#   --step-count <step-count>
#                           Number of diffusion steps to perform (default: 50)
#   --save-every <save-every>
#                           How often to save samples at intermediate steps (default: 0)
#         Set to 0 to only save the final sample
#   --output-path <output-path>
#                           Output path (default: ./)
#   --seed <seed>           Random seed (default: 93)
#   --guidance-scale <guidance-scale>
#                           Controls the influence of the text prompt on sampling process (0=random
#                           images) (default: 7.5)
#   --compute-units <compute-units>
#                           Compute units to load model with
#                           {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine} (default: all)
#   --scheduler <scheduler> Scheduler to use, one of {pndm, dpmpp} (default: pndm)
#   --disable-safety        Disable safety checking
#   --reduce-memory         Reduce memory usage
#   --version               Show the version.
#   -h, --help              Show help information.
```

## References

- https://huggingface.co/blog/diffusers-coreml
- https://huggingface.co/apple
- https://huggingface.co/coreml
