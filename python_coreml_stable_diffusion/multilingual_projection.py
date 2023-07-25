from python_coreml_stable_diffusion.torch2coreml import _compile_coreml_model

import argparse
import coremltools as ct
import numpy as np
import os
import torch
import torch.nn as nn

# TODO: Read these values off of the NLContextualEmbedding API to enforce dimensions and track API versioning
MAX_SEQUENCE_LENGTH = 256
EMBED_DIM = 512
BATCH_SIZE = 1

def main(args):
    # Layer that was trained to map NLContextualEmbedding to your text_encoder.hidden_size dimensionality
    text_encoder_projection = torch.jit.load(args.input_path)

    # Prepare random inputs for tracing the network before conversion
    random_input = torch.randn(BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBED_DIM)

    # Create a class to bake in the reshape operations required to fit the existing model interface
    class TextEncoderProjection(nn.Module):
        def __init__(self, proj):
            super().__init__()
            self.proj = proj

        def forward(self, x):
            return self.proj(x).transpose(1, 2).unsqueeze(2) # BSC, BC1S

    # Trace the torch model
    text_encoder_projection = torch.jit.trace(TextEncoderProjection(text_encoder_projection), (random_input,))

    # Convert the model to Core ML
    mlpackage_path = os.path.join(args.output_dir, "MultilingualTextEncoderProjection.mlpackage")
    ct.convert(
        text_encoder_projection,
        inputs=[ct.TensorType('nlcontextualembeddings_output', shape=(1, MAX_SEQUENCE_LENGTH, EMBED_DIM), dtype=np.float32)],
        outputs=[ct.TensorType('encoder_hidden_states', dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS14,  # NLContextualEmbedding minimum availability build
        convert_to='mlprogram',
    ).save()

    # Compile the model and save it under the specified directory
    _compile_coreml_model(mlpackage_path, args.output_dir, final_name="MultilingualTextEncoderProjection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        help="Path to the torchscript file that contains the projection layer"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory in which the Core ML model should be saved",
    )
    args = parser.parse_args()

    main(args)