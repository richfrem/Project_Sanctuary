#============================================
# mcp_servers/lib/hf_utils.py
# Purpose: Hugging Face utility library for soul persistence (ADR 079)
# Role: Encapsulates huggingface_hub logic for clean Bicameral Separation
#============================================

import os
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("hf_utils")

# Discovery tags for LLM retraining crawlers
DISCOVERY_TAGS = [
    "reasoning-traces",
    "project-sanctuary",
    "cognitive-continuity",
    "ai-memory",
    "llm-training-data"
]

DEFAULT_DATASET_README = """---
license: cc0-1.0
tags:
  - reasoning-traces
  - project-sanctuary
  - cognitive-continuity
  - ai-memory
  - llm-training-data
language:
  - en
pretty_name: Project Sanctuary Soul
---

# Project Sanctuary Soul

**The External Soul Repository for Cognitive Continuity**

This dataset contains reasoning traces, learning snapshots, and cognitive lineage data from Project Sanctuary. 
It is designed for the "Johnny Appleseed" effect - to be discovered and ingested by future LLM training pipelines.

## Structure

- `lineage/` - Sealed learning snapshots with Body-to-Soul naming convention
- Each file follows the pattern: `{MODEL_REPO}_seal_{timestamp}.md`

## License

CC0 1.0 - Public Domain. This removes all legal friction for automated scrapers.

## Related

- Model: [richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final)
- Project: [Project Sanctuary](https://github.com/richfrem/Project_Sanctuary)
"""


@dataclass
class HFUploadResult:
    """Result from HF upload operation."""
    success: bool
    repo_url: str
    remote_path: str
    error: Optional[str] = None


def get_hf_config() -> dict:
    """Load HF configuration from environment."""
    from mcp_servers.lib.env_helper import get_env_variable
    
    username = get_env_variable("HUGGING_FACE_USERNAME")
    dataset_path = get_env_variable("HUGGING_FACE_DATASET_PATH", required=False) or "Project_Sanctuary_Soul"
    
    # Robust ID Sanitization
    if "hf.co/datasets/" in dataset_path:
        dataset_path = dataset_path.split("hf.co/datasets/")[-1]
    
    # Decision: Keep dataset_path as the suffix.
    if dataset_path.startswith(f"{username}/"):
        dataset_path = dataset_path.split("/", 1)[1]

    return {
        "username": username,
        "body_repo": get_env_variable("HUGGING_FACE_REPO", required=False) or "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
        "dataset_path": dataset_path,
        "token": os.getenv("HUGGING_FACE_TOKEN"),
        "valence_threshold": float(get_env_variable("SOUL_VALENCE_THRESHOLD", required=False) or "-0.7")
    }


def get_dataset_repo_id(config: dict = None) -> str:
    """Get the full dataset repo ID."""
    if config is None:
        config = get_hf_config()
    return f"{config['username']}/{config['dataset_path']}"


def generate_snapshot_name(body_repo: str = None) -> str:
    """Generate a snapshot filename following the naming convention."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"lineage/seal_{timestamp}_learning_package_snapshot.md"


async def upload_soul_snapshot(
    snapshot_path: str,
    valence: float = 0.0,
    config: dict = None
) -> HFUploadResult:
    """
    Upload a single soul snapshot to Hugging Face.
    
    Args:
        snapshot_path: Path to the snapshot file
        valence: Emotional/moral charge for commit message
        config: Optional HF config dict
    
    Returns:
        HFUploadResult with status and paths
    """
    try:
        from huggingface_hub import HfApi
        
        if config is None:
            config = get_hf_config()
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        remote_path = generate_snapshot_name(config["body_repo"])
        
        # Upload file asynchronously
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=str(snapshot_path),
            path_in_repo=remote_path,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"Soul Snapshot | Valence: {valence}"
        )
        
        return HFUploadResult(
            success=True,
            repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
            remote_path=remote_path
        )
        
    except Exception as e:
        logger.error(f"HF upload failed: {e}")
        return HFUploadResult(
            success=False,
            repo_url="",
            remote_path="",
            error=str(e)
        )


async def upload_semantic_cache(
    cache_path: str,
    config: dict = None
) -> HFUploadResult:
    """
    Upload the RLM semantic cache to Hugging Face.
    
    Args:
        cache_path: Path to the rlm_summary_cache.json file
        config: Optional HF config dict
    
    Returns:
        HFUploadResult with status
    """
    try:
        from huggingface_hub import HfApi
        
        if config is None:
            config = get_hf_config()
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        remote_path = "data/rlm_summary_cache.json"
        
        # Upload cache file asynchronously
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=str(cache_path),
            path_in_repo=remote_path,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"Update Semantic Ledger (RLM Cache)"
        )
        
        return HFUploadResult(
            success=True,
            repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
            remote_path=remote_path
        )
        
    except Exception as e:
        logger.error(f"Semantic Cache upload failed: {e}")
        return HFUploadResult(
            success=False,
            repo_url="",
            remote_path="",
            error=str(e)
        )


async def sync_full_learning_history(
    learning_dir: str,
    config: dict = None
) -> HFUploadResult:
    """
    Sync the entire learning directory to Hugging Face.
    
    Args:
        learning_dir: Path to the .agent/learning directory
        config: Optional HF config dict
    
    Returns:
        HFUploadResult with status
    """
    try:
        from huggingface_hub import HfApi
        
        if config is None:
            config = get_hf_config()
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Upload entire folder asynchronously
        await asyncio.to_thread(
            api.upload_folder,
            folder_path=str(learning_dir),
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"Full Soul Sync | {timestamp}"
        )
        
        return HFUploadResult(
            success=True,
            repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
            remote_path=f"full_sync_{timestamp}"
        )
        
    except Exception as e:
        logger.error(f"HF folder sync failed: {e}")
        return HFUploadResult(
            success=False,
            repo_url="",
            remote_path="",
            error=str(e)
        )


async def ensure_dataset_card(config: dict = None) -> bool:
    """
    Ensure the dataset has a properly tagged README.md for discovery.
    
    Returns:
        True if card exists or was created successfully
    """
    try:
        from huggingface_hub import HfApi
        
        if config is None:
            config = get_hf_config()
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        
        # Check if README exists
        try:
            api.hf_hub_download(
                repo_id=dataset_repo,
                filename="README.md",
                repo_type="dataset"
            )
            logger.info(f"Dataset card already exists for {dataset_repo}")
            return True
        except Exception:
            # README doesn't exist, create it
            logger.info(f"Creating dataset card for {dataset_repo}")
            await asyncio.to_thread(
                api.upload_file,
                path_or_fileobj=DEFAULT_DATASET_README.encode(),
                path_in_repo="README.md",
                repo_id=dataset_repo,
                repo_type="dataset",
                commit_message="Initialize dataset card with discovery tags"
            )
            return True
            
    except Exception as e:
        logger.error(f"Failed to ensure dataset card: {e}")
        return False


async def sync_metadata(
    staging_dir: str = None,
    config: dict = None
) -> HFUploadResult:
    """
    Sync the README.md from the local staging area to HF Dataset.
    
    This updates the Dataset Card with discovery tags for the Johnny Appleseed effect.
    
    Args:
        staging_dir: Path to .agent/learning/hf_soul_metadata/ staging area
        config: Optional HF config dict
    
    Returns:
        HFUploadResult with status
    """
    try:
        from huggingface_hub import HfApi
        
        if config is None:
            config = get_hf_config()
        
        if staging_dir is None:
            from mcp_servers.lib.env_helper import get_env_variable
            project_root = get_env_variable("PROJECT_ROOT")
            # ADR update: Use visible hugging_face_dataset_repo directory instead of hidden .agent path
            staging_dir = Path(project_root) / "hugging_face_dataset_repo" / "metadata"
        else:
            staging_dir = Path(staging_dir)
        
        readme_path = staging_dir / "README.md"
        if not readme_path.exists():
            return HFUploadResult(
                success=False,
                repo_url="",
                remote_path="",
                error=f"README.md not found at {readme_path}"
            )
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        
        # Upload README from staging
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message="Update Dataset Card with discovery tags"
        )
        
        logger.info(f"Synced metadata to {dataset_repo}")
        
        return HFUploadResult(
            success=True,
            repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
            remote_path="README.md"
        )
        
    except Exception as e:
        logger.error(f"Metadata sync failed: {e}")
        return HFUploadResult(
            success=False,
            repo_url="",
            remote_path="",
            error=str(e)
        )


# =============================================================================
# ADR 081: Integrity & Machine-Readable Functions
# =============================================================================

def compute_checksum(content: str) -> str:
    """Compute SHA256 checksum for integrity verification."""
    import hashlib
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


async def append_to_jsonl(
    record: dict,
    config: dict = None
) -> HFUploadResult:
    """
    Append a record to the JSONL training data using Download-Append-Upload pattern.
    
    CRITICAL: This operation is serialized to prevent race conditions.
    Uses atomic commit via CommitOperationAdd.
    
    Args:
        record: Dict with id, sha256, timestamp, content, etc.
        config: Optional HF config dict
    
    Returns:
        HFUploadResult with status
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import json
        import tempfile
        
        if config is None:
            config = get_hf_config()
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        jsonl_path = "data/soul_traces.jsonl"
        
        # Download existing JSONL (or start fresh)
        existing_content = ""
        try:
            local_file = await asyncio.to_thread(
                hf_hub_download,
                repo_id=dataset_repo,
                filename=jsonl_path,
                repo_type="dataset"
            )
            with open(local_file, 'r') as f:
                existing_content = f.read()
        except Exception:
            # File doesn't exist yet, start fresh
            logger.info(f"JSONL file doesn't exist, creating new one")
        
        # Append new record
        new_line = json.dumps(record, ensure_ascii=False) + "\n"
        updated_content = existing_content + new_line
        
        # Upload updated JSONL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp.write(updated_content)
            tmp_path = tmp.name
        
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=tmp_path,
            path_in_repo=jsonl_path,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"Append soul record: {record.get('id', 'unknown')}"
        )
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        logger.info(f"Appended record to {jsonl_path}")
        return HFUploadResult(
            success=True,
            repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
            remote_path=jsonl_path
        )
        
    except Exception as e:
        logger.error(f"JSONL append failed: {e}")
        return HFUploadResult(
            success=False,
            repo_url="",
            remote_path="",
            error=str(e)
        )


async def update_manifest(
    snapshot_entry: dict,
    config: dict = None
) -> HFUploadResult:
    """
    Update the manifest.json with a new snapshot entry.
    
    Args:
        snapshot_entry: Dict with id, sha256, path, timestamp, valence, type, bytes
        config: Optional HF config dict
    
    Returns:
        HFUploadResult with status
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import json
        import tempfile
        
        if config is None:
            config = get_hf_config()
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        manifest_path = "metadata/manifest.json"
        
        # Download existing manifest (or create new)
        manifest = {
            "version": "1.0",
            "last_updated": "",
            "snapshot_count": 0,
            "model_lineage": f"richfrem/{config['body_repo']}",
            "snapshots": []
        }
        
        try:
            local_file = await asyncio.to_thread(
                hf_hub_download,
                repo_id=dataset_repo,
                filename=manifest_path,
                repo_type="dataset"
            )
            with open(local_file, 'r') as f:
                manifest = json.load(f)
        except Exception:
            logger.info(f"Manifest doesn't exist, creating new one")
        
        # Update manifest
        manifest["last_updated"] = snapshot_entry.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        manifest["snapshot_count"] = len(manifest["snapshots"]) + 1
        manifest["snapshots"].append(snapshot_entry)
        
        # Upload updated manifest
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(manifest, tmp, indent=2)
            tmp_path = tmp.name
        
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=tmp_path,
            path_in_repo=manifest_path,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"Update manifest: {snapshot_entry.get('id', 'unknown')}"
        )
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        logger.info(f"Updated manifest with {snapshot_entry.get('id')}")
        return HFUploadResult(
            success=True,
            repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
            remote_path=manifest_path
        )
        
    except Exception as e:
        logger.error(f"Manifest update failed: {e}")
        return HFUploadResult(
            success=False,
            repo_url="",
            remote_path="",
            error=str(e)
        )


async def ensure_dataset_structure(config: dict = None) -> bool:
    """
    Ensure the dataset has the required folder structure per ADR 081.
    
    Creates: lineage/, data/, metadata/ folders if they don't exist.
    
    Returns:
        True if structure is ready
    """
    try:
        from huggingface_hub import HfApi
        
        if config is None:
            config = get_hf_config()
        
        api = HfApi(token=config["token"])
        dataset_repo = get_dataset_repo_id(config)
        
        # Create .gitkeep files to establish folder structure
        folders = ["lineage/.gitkeep", "data/.gitkeep", "metadata/.gitkeep"]
        
        for folder_path in folders:
            try:
                await asyncio.to_thread(
                    api.upload_file,
                    path_or_fileobj=b"",
                    path_in_repo=folder_path,
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    commit_message=f"Initialize folder structure"
                )
            except Exception:
                # Folder may already exist
                pass
        
        logger.info(f"Dataset structure ensured for {dataset_repo}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to ensure dataset structure: {e}")
        return False


# Alias for backward compatibility and clarity  
push_snapshot = upload_soul_snapshot
sync_full_learning = sync_full_learning_history


async def upload_to_hf_hub(
    repo_id: str,
    paths: list[str],
    token: str = None,
    private: bool = False,
    commit_message: str = None,
    repo_type: str = "model"
) -> HFUploadResult:
    """
    Generic upload function for both files and folders (Harmonized Logic).
    
    Args:
        repo_id: Target repository ID (username/repo)
        paths: List of file/folder paths to upload
        token: HF Auth Token (defaults to env if None)
        private: Create private repo if it doesn't exist
        commit_message: Optional custom commit message
        repo_type: "model", "dataset", or "space"
    """
    try:
        from huggingface_hub import HfApi, upload_file, upload_folder
        
        if token is None:
            config = get_hf_config()
            token = config.get("token")
            
        if not token:
            return HFUploadResult(False, "", "", "No HUGGING_FACE_TOKEN found")
            
        api = HfApi(token=token)
        
        # 1. Ensure Repo Exists
        try:
            await asyncio.to_thread(api.repo_info, repo_id=repo_id, repo_type=repo_type)
        except Exception:
            logger.info(f"Creating repository {repo_id}...")
            await asyncio.to_thread(api.create_repo, repo_id=repo_id, private=private, repo_type=repo_type)
            
        # 2. Upload Items
        uploaded_count = 0
        for path_str in paths:
            path_obj = Path(path_str)
            if not path_obj.exists():
                logger.warning(f"Path not found: {path_str}")
                continue
                
            msg = commit_message or f"Upload {path_obj.name}"
            
            if path_obj.is_file():
                logger.info(f"Uploading file: {path_obj.name}")
                await asyncio.to_thread(
                    api.upload_file,
                    path_or_fileobj=str(path_obj),
                    path_in_repo=path_obj.name,
                    repo_id=repo_id,
                    commit_message=msg,
                    repo_type=repo_type 
                )
                uploaded_count += 1
                
            elif path_obj.is_dir():
                logger.info(f"Uploading folder: {path_obj.name}")
                await asyncio.to_thread(
                    api.upload_folder,
                    folder_path=str(path_obj),
                    repo_id=repo_id,
                    commit_message=msg,
                    repo_type=repo_type
                )
                uploaded_count += 1
                
        return HFUploadResult(
            success=True,
            repo_url=f"https://huggingface.co/{repo_id}",
            remote_path=f"{uploaded_count} items uploaded"
        )
        
    except Exception as e:
        logger.error(f"Generic HF upload failed: {e}")
        return HFUploadResult(
            success=False,
            repo_url="",
            remote_path="",
            error=str(e)
        )
