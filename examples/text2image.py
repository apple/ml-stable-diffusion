from diffusers import DiffusionPipeline
from datetime import datetime

class Text2Image:
    def __init__(self, model_ref) -> None:
        """
        :param model_ref: str, the reference to load the mdoel from Huggingface
        """
        self.pipe = DiffusionPipeline.from_pretrained(model_ref).to("mps")

        # Recommended if your computer has < 64 GB of RAM
        self.pipe.enable_attention_slicing()

    def imagine(self, prompt):
        """
        Create an image based on prompt text.
        :param prompt: str, prompt text
        :return Image
        """
        # First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
        _ = self.pipe(prompt, num_inference_steps=1)

        # Results match those from the CPU device after the warmup pass.
        return self.pipe(prompt).images[0]

def main():
    model_ref = "runwayml/stable-diffusion-v1-5"
    prompt = "a photo of an astronaut riding a horse on mars"

    image = Text2Image(model_ref).imagine(prompt)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"{prompt.replace(' ','_')}_{time_str}.png"
    image.save(file_path)

if __name__ == "__main__":
    main()