from huggingface_hub import snapshot_download
from huggingface_hub.file_download import repo_folder_name
from pathlib import Path
import shutil

# From apple: https://huggingface.co/apple
# repo_id = "apple/coreml-stable-diffusion-v1-5"
# repo_id = "apple/coreml-stable-diffusion-v1-4"
# repo_id = "apple/coreml-stable-diffusion-2-base"

# For Swift
# variant = "original/compiled"
# For Python
# variant = "original/packages"

# From coreml: https://huggingface.co/coreml
repo_id = "coreml/coreml-stable-diffusion-2-1-base"
variant = "original"


def download_model(repo_id, variant, output_dir):
    destination = Path(output_dir) / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
    if destination.exists():
        raise Exception(f"Model already exists at {destination}")

    # Download and copy without symlinks
    downloaded = snapshot_download(repo_id, allow_patterns=f"{variant}/*", cache_dir=output_dir)
    downloaded_bundle = Path(downloaded) / variant
    shutil.copytree(downloaded_bundle, destination)

    # Remove all downloaded files
    cache_folder = Path(output_dir) / repo_folder_name(repo_id=repo_id, repo_type="model")
    shutil.rmtree(cache_folder)
    return destination

model_path = download_model(repo_id, variant, output_dir="./models")
print(f"Model downloaded at {model_path}")
