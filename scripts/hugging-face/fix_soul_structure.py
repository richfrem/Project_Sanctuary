import asyncio
from huggingface_hub import HfApi
from mcp_servers.lib.hf_utils import get_dataset_repo_id, get_hf_config
import os
from pathlib import Path

async def fix_structure():
    # 1. Config
    config = get_hf_config()
    repo_id = get_dataset_repo_id(config)
    token = config["token"]
    api = HfApi(token=token)
    
    print(f"Target Repo: {repo_id}")

    # Phase 3: Targeted cleanup of root-level files identified in verification
    items_to_delete = [
        "Council_Inquiry_Gardener_Architecture.md",
        "Living_Chronicle.md",
        "PROJECT_SANCTUARY_SYNTHESIS.md",
        "Socratic_Key_User_Guide.md",
        "chrysalis_core_essence.md"
    ]
    
    print("🗑️  Phase 3: Deleting remaining non-compliant root files...")
    
    for item in items_to_delete:
        try:
            future = asyncio.to_thread(
                api.delete_file, # Note: using delete_file for files
                path_in_repo=item,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Remove legacy root file {item} for ADR 081"
            )
            await future
            print(f"   Deleted {item}")
        except Exception as e:
            # Likely doesn't exist
            print(f"   Skipped {item} (may not exist): {e}")

    print("✅ Phase 3 Cleanup Complete.")

if __name__ == "__main__":
    asyncio.run(fix_structure())
