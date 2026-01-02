#============================================
# scripts/hf_upload_assets.py
# Purpose: Synchronizes staged landing-page assets with the Hugging Face Hub.
# Role:
#   1. Uploads the final, metadata-rich README.md to the repository root.
#   2. Ensures the public Dataset Card is up to date with the latest protocol evolution.
#   3. Complements cortex_cli.py (which handles the machine-readable data).
# ADR: 081 - Content Harmonization & Integrity
#============================================
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

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
    
    # 1. Upload README
    readme_path = project_root / "hugging_face_dataset_repo" / "README.md"
    print(f"Uploading {readme_path} as README.md ...")
    
    if readme_path.exists():
        future_readme = asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update Root README from Staging"
        )
        await future_readme
        print("✅ README.md updated.")
    else:
        print("❌ Staged README.md not found!")

if __name__ == "__main__":
    asyncio.run(upload_assets())
