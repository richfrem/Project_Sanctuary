#!/usr/bin/env python3
"""persist_soul.py - Soul Traces & RLM Caches Transport to Hugging Face Hub.
Purpose: Broadcasts a sealed learning snapshot (and the RLM summary cache, if present) to the Hugging Face dataset by
    delegating to the global APM skill `managing-huggingface-assets` (ADR 081 layout).
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 3 remote transport).
Key Input Dependencies:
    - Global APM skill `managing-huggingface-assets` (found via MANAGING_HF_SKILL_PATH or the standard skill directories)
    - Environment: HUGGING_FACE_TOKEN, HUGGING_FACE_USERNAME, HUGGING_FACE_REPO, HUGGING_FACE_DATASET_PATH
    - A sealed snapshot file (default .agent/learning/learning_package_snapshot.md)
Key Outputs: JSON receipts on stdout (upload results); JSON errors on stderr with exit code 1.
Usage:
    python3 persist_soul.py                      # upload snapshot (+ RLM cache if present)
    python3 persist_soul.py --dry-run            # resolve skill, credentials and snapshot; upload nothing
    python3 persist_soul.py --snapshot PATH --valence 0.2
    python3 persist_soul.py --full-sync          # prints the commands for a full dataset sync
Key Functions: find_managing_hf_skill(), _find_project_root(), _full_sync_commands(), main().
History: replaced a dependency on the removed `plugins/huggingface-utils/` plugin (it failed with
    "No module named hf_config"); fixed 2026-11-03.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

SKILL_NAME = "managing-huggingface-assets"
REQUIRED_ENV = ["HUGGING_FACE_TOKEN", "HUGGING_FACE_USERNAME", "HUGGING_FACE_REPO", "HUGGING_FACE_DATASET_PATH"]


def find_managing_hf_skill() -> Optional[Path]:
    """Locate the scripts directory of the global APM skill managing-huggingface-assets.

    Order: MANAGING_HF_SKILL_PATH override, then the standard global skill directories for Claude,
    Copilot/Codex and Antigravity/Gemini.
    """
    override = os.getenv("MANAGING_HF_SKILL_PATH")
    if override and (Path(override) / "hf_config.py").exists():
        return Path(override)
    home = Path.home()
    for base in (home / ".claude" / "skills", home / ".agents" / "skills",
                 home / ".gemini" / "antigravity-cli" / "skills", home / ".gemini" / "skills"):
        scripts = base / SKILL_NAME / "scripts"
        if (scripts / "hf_config.py").exists():
            return scripts
    return None


def _find_project_root(start: Path) -> Path:
    """Walk up from `start` to the directory containing .git (a directory, or a file in a worktree)."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return start


def _full_sync_commands(project_root: Path) -> List[str]:
    """Commands that perform a full dataset sync (extract, verify, publish)."""
    skills = project_root / "plugins" / "sanctuary-core" / ".apm" / "skills" / "preserving-soul-traces" / "scripts"
    return [
        f"python3 {skills}/soul_coverage.py closure",
        f"python3 {skills}/trace_extractor.py --verify",
        f"python3 {skills}/sync_soul_dataset.py --dry-run",
        f"python3 {skills}/sync_soul_dataset.py --message \"feat(soul): sync soul traces\"",
    ]


def _fail(payload: Dict[str, object]) -> None:
    """Print a JSON error to stderr and exit 1."""
    print(json.dumps(payload, indent=2), file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist Soul Memory to Hugging Face")
    parser.add_argument("--snapshot", type=str, default=".agent/learning/learning_package_snapshot.md",
                        help="Path to sealed snapshot (relative to the project root)")
    parser.add_argument("--valence", type=float, default=0.0, help="Moral/Emotional charge")
    parser.add_argument("--full-sync", action="store_true", help="Print the commands for a full dataset sync")
    parser.add_argument("--dry-run", action="store_true", help="Validate everything, upload nothing")
    args = parser.parse_args()

    project_root = _find_project_root(Path(__file__).resolve().parent)

    if args.full_sync:
        print(json.dumps({"status": "redirect",
                          "message": "A full sync is extract, verify, then sync; run these commands.",
                          "commands": _full_sync_commands(project_root)}, indent=2))
        return

    scripts = find_managing_hf_skill()
    if scripts is None:
        _fail({"status": "error", "error": f"Global APM skill '{SKILL_NAME}' not found.",
               "fix": f"Install it with APM (apm install ... --global) or set MANAGING_HF_SKILL_PATH to its scripts directory."})
    sys.path.insert(0, str(scripts))
    try:
        from hf_config import get_hf_config
        from hf_upload import upload_semantic_cache, upload_soul_snapshot
    except ImportError as exc:
        _fail({"status": "error", "error": f"Could not import from {scripts}: {exc}"})

    try:
        config = get_hf_config()
    except ValueError as exc:
        _fail({"status": "error", "error": str(exc), "fix": f"Set {', '.join(REQUIRED_ENV)} in the shell environment."})

    snapshot_path = project_root / args.snapshot
    if not snapshot_path.exists():
        _fail({"status": "error", "error": f"Snapshot missing: {snapshot_path}"})
    cache_file = project_root / ".agent/learning/rlm_summary_cache.json"

    if args.dry_run:
        print(json.dumps({"status": "dry-run", "skill_scripts": str(scripts), "dataset": config.dataset_repo_id,
                          "snapshot": str(snapshot_path), "rlm_cache": str(cache_file) if cache_file.exists() else None}, indent=2))
        return

    try:
        result = asyncio.run(upload_soul_snapshot(snapshot_path, args.valence, config))
        print(json.dumps({"status": "success" if result.success else "error", "snapshot": result.__dict__}, indent=2))
        if not result.success:
            sys.exit(1)
        if cache_file.exists():
            cache_result = asyncio.run(upload_semantic_cache(cache_file, config))
            print(json.dumps({"cache_sync": cache_result.__dict__}, indent=2))
    except Exception as exc:  # network or API failure
        _fail({"status": "error", "error": str(exc)})


if __name__ == "__main__":
    main()
