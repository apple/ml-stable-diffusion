import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import logging

logging.basicConfig(level=logging.INFO)
class Image2Image:
    def __init__(self, model_ref, max_height=512):
        """
        :param model_ref: str, the reference to load the mdoel from Huggingface
        :param max_height: int, the image will be resized up to this value in height and proportionally for width. With the default value of 512, image generation typically finishes within 1 min.
        """
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_ref).to("mps")
        self.max_height = max_height

        # Recommended if your computer has < 64 GB of RAM
        self.pipe.enable_attention_slicing()

    def promptWithImage(self, prompt, image_path):
        """
        Create an image based on prompt text and the base image.
        :param prompt: str
        :image_path: str, path to load the image, load from web if starting with "http://" or "https://". Otherwise, load from the local file system.
        :return Image
        """
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image_content = requests.get(image_path).content
            init_image = Image.open(BytesIO(image_content)).convert("RGB")
        else:
            init_image = Image.open(image_path).convert("RGB")
        
        # resize to max_height and for width with same scale.
        width,height = init_image.size
        new_width,new_height = int(width * (self.max_height) / height),self.max_height
        init_image = init_image.resize((new_width, new_height))
        logging.info(f"resize image: ({width,height}) -> ({new_width,new_height})")

        images = self.pipe(prompt=prompt, image=init_image).images
        return images[0]
    
def main():
    model_ref = "runwayml/stable-diffusion-v1-5"

    prompt = "A fantasy landscape, trending on artstation"
    image_path = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    image = Image2Image(model_ref).promptWithImage(prompt, image_path)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"{prompt.replace(' ','_')}_{time_str}.png"
    image.save(file_path)

if __name__ == "__main__":
    main()