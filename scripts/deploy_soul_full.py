import asyncio
from huggingface_hub import HfApi
from mcp_servers.lib.hf_utils import get_dataset_repo_id, get_hf_config
from pathlib import Path

async def deploy():
    config = get_hf_config()
    repo_id = get_dataset_repo_id(config)
    token = config["token"]
    api = HfApi(token=token)
    
    print(f"Target Repo: {repo_id}")
    staging_dir = Path("STAGING_HF_SOUL")
    
    # Upload data/ only (JSONL contains all content - lineage is redundant)
    print("🚀 Uploading data/soul_traces.jsonl...")
    await asyncio.to_thread(
        api.upload_folder,
        folder_path=str(staging_dir / "data"),
        path_in_repo="data",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Deploy Soul Data JSONL (ADR 081)"
    )
    
    print("✅ Deployment Complete.")

if __name__ == "__main__":
    asyncio.run(deploy())
