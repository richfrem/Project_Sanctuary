#!/usr/bin/env python3
"""
sync_soul_dataset.py — Hugging Face Hub Dataset Synchronizer CLI
==============================================================
Purpose:
    Coordinates synchronization of the local huggingface/ dataset workspace
    to the remote Hugging Face repository (richfrem/Project_Sanctuary_Soul).
Layer: Hub Transport / Remote Persistence (delegates to managing-huggingface-assets)

Key Inputs:
    Files: huggingface/data/soul_traces.jsonl, huggingface/metadata/manifest.json, README.md
    Credentials: HUGGING_FACE_TOKEN (~/.zshrc), HUGGING_FACE_USERNAME=richfrem
    Skill Dependency: managing-huggingface-assets (global APM skill)

Key Outputs:
    Hub Dataset: https://huggingface.co/datasets/richfrem/Project_Sanctuary_Soul
    Immutable Commit: Pushed training split, dataset card, and cryptographic seal

Usage Examples:
    python3 scripts/sync_soul_dataset.py --check-readiness  # Validate local files & HF skill
    python3 scripts/sync_soul_dataset.py --dry-run          # Preview upload without remote write
    python3 scripts/sync_soul_dataset.py --message "..."    # Execute upload to Hugging Face Hub

Key Functions:
    find_managing_hf_skill(), validate_local_dataset(), execute_sync(), main()
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Global APM Skill Discovery
# ---------------------------------------------------------------------------

# Discover the installed location of the global managing-huggingface-assets skill.
def find_managing_hf_skill() -> Optional[Path]:
    """Locate the global APM skill managing-huggingface-assets."""
    env_path = os.getenv("MANAGING_HF_SKILL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Standard global APM skill directory
    default_path = Path.home() / ".gemini" / "antigravity-cli" / "skills" / "managing-huggingface-assets" / "scripts"
    if default_path.exists():
        return default_path

    # Fallback search in user home .gemini skills
    candidate = Path.home() / ".gemini" / "skills" / "managing-huggingface-assets" / "scripts"
    if candidate.exists():
        return candidate

    return None


# ---------------------------------------------------------------------------
# Local Dataset Validation & Upload Execution
# ---------------------------------------------------------------------------

# Verify that the local dataset workspace contains valid data files and manifests.
def validate_local_dataset(dataset_dir: Path) -> Tuple[bool, str]:
    """Check that the local dataset workspace contains valid artifacts."""
    if not dataset_dir.exists():
        return False, f"Dataset directory {dataset_dir} does not exist"

    data_file = dataset_dir / "data" / "soul_traces.jsonl"
    manifest_file = dataset_dir / "metadata" / "manifest.json"
    readme_file = dataset_dir / "README.md"

    if not data_file.exists() or data_file.stat().st_size == 0:
        return False, f"Missing or empty dataset file: {data_file}"

    if not manifest_file.exists():
        return False, f"Missing manifest file: {manifest_file}"

    if not readme_file.exists():
        return False, f"Missing dataset card README: {readme_file}"

    return True, "Dataset structure valid"


# Delegate the folder upload to managing-huggingface-assets with exponential backoff.
async def execute_sync(dataset_dir: Path, hf_scripts_dir: Path, commit_msg: str, dry_run: bool = False) -> bool:
    """Execute upload via managing-huggingface-assets."""
    sys.path.insert(0, str(hf_scripts_dir))
    try:
        from hf_config import get_hf_config
        from hf_upload import upload_folder, ensure_dataset_structure, ensure_dataset_card
    except ImportError as e:
        sys.stderr.write(f"Failed to import from managing-huggingface-assets: {e}\n")
        return False

    config = get_hf_config()
    target_repo = config.dataset_repo_id

    print(f"Target Hugging Face Dataset: https://huggingface.co/datasets/{target_repo}")
    print(f"Local Dataset Root:         {dataset_dir}")
    print(f"Commit Message:             {commit_msg}")

    if dry_run:
        print("\n[DRY-RUN] Dataset is validated. Sync would delegate to managing-huggingface-assets upload_folder().")
        print(f"[DRY-RUN] Destination: repo_id='{target_repo}', folder='{dataset_dir}'")
        return True

    # Real upload execution (requires explicit human approval per Human Gate Law)
    print("\nInitializing dataset card and folder structure on remote...")
    await ensure_dataset_structure(config)
    await ensure_dataset_card(config)

    print(f"Uploading {dataset_dir.name} to Hugging Face...")
    result = await upload_folder(
        folder=dataset_dir,
        remote_prefix="",
        config=config,
        commit_msg=commit_msg
    )

    if result.success:
        print(f"\nSUCCESS: Soul traces persisted to {result.repo_url}")
        return True
    else:
        print(f"\nERROR: Upload failed: {result.error}")
        return False


# ---------------------------------------------------------------------------
# CLI Command Entrypoint
# ---------------------------------------------------------------------------

# Command-line entrypoint: parse flags, validate readiness, and execute Hub sync.
def main():
    """CLI entrypoint for soul trace dataset readiness and synchronization."""
    parser = argparse.ArgumentParser(description="Sanctuary Soul Trace Dataset Sync")
    parser.add_argument(
        "--dataset-dir",
        default="huggingface",
        help="Path to local Hugging Face dataset folder"
    )
    parser.add_argument(
        "--message",
        default="Persist sovereign soul traces and learning snapshot",
        help="Commit message for Hugging Face upload"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset and preview sync without uploading"
    )
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        help="Validate local dataset and managing-huggingface-assets availability"
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    # 1. Validate local dataset
    valid, msg = validate_local_dataset(dataset_dir)
    if not valid:
        sys.stderr.write(f"Dataset Validation Error: {msg}\n")
        sys.exit(1)

    print(f"Dataset Check: OK ({dataset_dir.name})")

    # 2. Check managing-huggingface-assets integration
    hf_scripts = find_managing_hf_skill()
    if not hf_scripts:
        sys.stderr.write(
            "ERROR: Global APM skill 'managing-huggingface-assets' not found.\n"
            "Install it via APM or set MANAGING_HF_SKILL_PATH.\n"
        )
        sys.exit(1)

    print(f"HF Skill Integration: OK ({hf_scripts})")

    if args.check_readiness:
        print("Readiness check passed. Skill is ready to persist soul traces.")
        sys.exit(0)

    # 3. Execute sync
    success = asyncio.run(execute_sync(
        dataset_dir=dataset_dir,
        hf_scripts_dir=hf_scripts,
        commit_msg=args.message,
        dry_run=args.dry_run
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
