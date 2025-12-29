import asyncio
import os
from pathlib import Path
from huggingface_hub import HfApi
from mcp_servers.lib.hf_utils import get_dataset_repo_id, get_hf_config

async def upload_assets():
    # Load Config
    config = get_hf_config()
    repo_id = get_dataset_repo_id(config)
    token = config["token"]
    
    if not token:
        print("Error: No HUGGING_FACE_TOKEN found.")
        return

    api = HfApi(token=token)
    project_root = Path(os.getcwd())
    
    print(f"Target Repo: {repo_id}")
    
    # 1. Upload Images
    images_dir = project_root / "README-MARKDOWN-IMAGES"
    print(f"Uploading images from {images_dir} to images/ ...")
    
    if images_dir.exists():
        # process: we want to upload all files in this dir to 'images/' path in repo
        future_images = asyncio.to_thread(
            api.upload_folder,
            folder_path=str(images_dir),
            path_in_repo="images",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload Architecture Diagrams (SVG)"
        )
        await future_images
        print("✅ Images uploaded.")
    else:
        print("❌ Images directory not found!")

    # 2. Upload README
    readme_hf = project_root / "README_HF.md"
    print(f"Uploading {readme_hf} as README.md ...")
    
    if readme_hf.exists():
        future_readme = asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=str(readme_hf),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update Root README with Diagram Images"
        )
        await future_readme
        print("✅ README.md updated.")
    else:
        print("❌ README_HF.md not found!")

if __name__ == "__main__":
    asyncio.run(upload_assets())
