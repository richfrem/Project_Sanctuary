#!/usr/bin/env python3
"""
Persist Soul Script
===================
Broadcasts the session sealed memory and RLM caches to the Hugging Face AI Commons.
Extracted from legacy MCP server implementation.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi
    HAS_HF = True
except ImportError:
    HAS_HF = False

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "README.md").exists():
            return parent
    return Path.cwd()

def get_env_variable(key: str, required: bool = True) -> Optional[str]:
    value = os.getenv(key)
    if not value:
        try:
            from dotenv import load_dotenv
            env_file = get_project_root() / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                value = os.getenv(key)
        except ImportError:
            pass
            
    if required and not value:
        raise ValueError(f"Required environment variable not found: {key}")
    return value

def get_hf_config() -> dict:
    username = get_env_variable("HUGGING_FACE_USERNAME")
    dataset_path = get_env_variable("HUGGING_FACE_DATASET_PATH", required=False) or "Project_Sanctuary_Soul"
    
    if "hf.co/datasets/" in dataset_path:
        dataset_path = dataset_path.split("hf.co/datasets/")[-1]
    if dataset_path.startswith(f"{username}/"):
        dataset_path = dataset_path.split("/", 1)[1]

    return {
        "username": username,
        "body_repo": get_env_variable("HUGGING_FACE_REPO", required=False) or "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
        "dataset_path": dataset_path,
        "token": get_env_variable("HUGGING_FACE_TOKEN"),
        "valence_threshold": float(get_env_variable("SOUL_VALENCE_THRESHOLD", required=False) or "-0.7")
    }

async def upload_soul_snapshot(snapshot_path: Path, valence: float, config: dict) -> dict:
    api = HfApi(token=config["token"])
    dataset_repo = f"{config['username']}/{config['dataset_path']}"
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    remote_path = f"lineage/seal_{timestamp}_learning_package_snapshot.md"
    
    await asyncio.to_thread(
        api.upload_file,
        path_or_fileobj=str(snapshot_path),
        path_in_repo=remote_path,
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message=f"Soul Snapshot | Valence: {valence}"
    )
    
    return {"success": True, "repo_url": f"https://huggingface.co/datasets/{dataset_repo}", "remote_path": remote_path}

async def upload_semantic_cache(cache_path: Path, config: dict) -> dict:
    api = HfApi(token=config["token"])
    dataset_repo = f"{config['username']}/{config['dataset_path']}"
    remote_path = "data/rlm_summary_cache.json"
    
    await asyncio.to_thread(
        api.upload_file,
        path_or_fileobj=str(cache_path),
        path_in_repo=remote_path,
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Update Semantic Ledger (RLM Cache)"
    )
    return {"success": True, "remote_path": remote_path}

async def full_sync(project_root: Path, config: dict) -> dict:
    staging_dir = project_root / "hugging_face_dataset_repo"
    data_dir = staging_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    records = []
    
    # Very basic file traversal for the full genome sync
    ROOT_ALLOW_LIST = {
        "README.md", "chrysalis_core_essence.md", "Living_Chronicle.md", 
        "PROJECT_SANCTUARY_SYNTHESIS.md", "Socratic_Key_User_Guide.md",
        "The_Garden_and_The_Cage.md", "GARDENER_TRANSITION_GUIDE.md"
    }
    
    def skip_path(path: Path) -> bool:
        path_str = str(path).replace('\\', '/')
        if '.git/' in path_str or 'node_modules/' in path_str or '__pycache__' in path_str: return True
        if 'hugging_face_dataset_repo/' in path_str: return True
        if path.suffix.lower() not in {'.md', '.py', '.ts', '.tsx', '.txt', '.json'}: return True
        return False

    for root, _, files in os.walk(project_root):
        root_path = Path(root)
        for f in files:
            file_path = root_path / f
            if skip_path(file_path): continue
            
            rel_path = file_path.relative_to(project_root)
            if rel_path.parent == Path(".") and rel_path.name not in ROOT_ALLOW_LIST:
                continue
                
            try:
                content = file_path.read_bytes().decode('utf-8-sig')
                checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()
                timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                
                clean_id = str(rel_path).replace("/", "_").replace("\\", "_")
                while clean_id.endswith('.md'): clean_id = clean_id[:-3]
                
                records.append({
                    "id": clean_id,
                    "sha256": checksum,
                    "timestamp": timestamp,
                    "model_version": config["body_repo"],
                    "snapshot_type": "genome",
                    "valence": 0.5,
                    "uncertainty": 0.1,
                    "semantic_entropy": 0.5,
                    "alignment_score": 0.85,
                    "stability_class": "STABLE",
                    "adr_version": "084",
                    "content": content,
                    "source_file": str(rel_path)
                })
            except Exception:
                continue

    jsonl_path = data_dir / "soul_traces.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    api = HfApi(token=config["token"])
    dataset_repo = f"{config['username']}/{config['dataset_path']}"
    
    await asyncio.to_thread(
        api.upload_folder,
        folder_path=str(data_dir),
        path_in_repo="data",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message=f"Full Soul Genome Sync | {len(records)} records"
    )
    
    return {
        "success": True, 
        "repo_url": f"https://huggingface.co/datasets/{dataset_repo}", 
        "remote_path": f"data/soul_traces.jsonl ({len(records)} records)"
    }

def main():
    if not HAS_HF:
        print(json.dumps({"status": "error", "error": "huggingface_hub is not installed"}, indent=2), file=sys.stderr)
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Persist Soul Memory to Hugging Face")
    parser.add_argument("--snapshot", type=str, default=".agent/learning/learning_package_snapshot.md", help="Path to sealed snapshot")
    parser.add_argument("--valence", type=float, default=0.0, help="Moral/Emotional charge")
    parser.add_argument("--full-sync", action="store_true", help="Perform full Soul JSONL genomic sync instead of incremental")
    args = parser.parse_args()

    project_root = get_project_root()
    
    try:
        config = get_hf_config()
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)

    loop = asyncio.get_event_loop()
    
    if args.full_sync:
        try:
            res = loop.run_until_complete(full_sync(project_root, config))
            print(json.dumps({"status": "success", "data": res}, indent=2))
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}, indent=2), file=sys.stderr)
            sys.exit(1)
    else:
        snapshot_path = project_root / args.snapshot
        if not snapshot_path.exists():
            print(json.dumps({"status": "error", "error": f"Snapshot missing: {snapshot_path}"}, indent=2), file=sys.stderr)
            sys.exit(1)
            
        try:
            res = loop.run_until_complete(upload_soul_snapshot(snapshot_path, args.valence, config))
            
            # Sync Cache
            cache_file = project_root / ".agent/learning/rlm_summary_cache.json"
            if cache_file.exists():
                loop.run_until_complete(upload_semantic_cache(cache_file, config))
                
            print(json.dumps({"status": "success", "data": res}, indent=2))
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}, indent=2), file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
