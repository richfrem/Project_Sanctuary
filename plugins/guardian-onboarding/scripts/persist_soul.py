#!/usr/bin/env python3
"""
Persist Soul Script (Thin Wrapper)
===================================
Broadcasts the session sealed memory and RLM caches to the Hugging Face AI Commons.

This script is now a thin wrapper that delegates all HF operations to the
standalone `plugins/huggingface-utils/` plugin. The Guardian plugin consumes
the shared HF utilities rather than maintaining its own inline implementations.

Migration:
    - Config resolution   → plugins/huggingface-utils/scripts/hf_config.py
    - Upload primitives   → plugins/huggingface-utils/skills/hf-upload/scripts/hf_upload.py
    - Init & env setup    → plugins/huggingface-utils/skills/hf-init/scripts/hf_init.py
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve imports from huggingface-utils plugin
# ---------------------------------------------------------------------------
def _resolve_hf_plugin():
    """Find and add the huggingface-utils plugin to sys.path."""
    script_dir = Path(__file__).resolve().parent
    # Walk up to project root
    project_root = script_dir
    for parent in script_dir.parents:
        if (parent / ".git").exists():
            project_root = parent
            break

    hf_config_path = project_root / "plugins" / "huggingface-utils" / "scripts"
    hf_upload_path = project_root / "plugins" / "huggingface-utils" / "skills" / "hf-upload" / "scripts"

    for p in [hf_config_path, hf_upload_path]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    return project_root

PROJECT_ROOT = _resolve_hf_plugin()

try:
    from hf_config import get_hf_config
    from hf_upload import (
        upload_soul_snapshot,
        upload_semantic_cache,
        upload_folder,
    )
    HAS_HF_PLUGIN = True
except ImportError as e:
    HAS_HF_PLUGIN = False
    _import_error = str(e)


def main():
    if not HAS_HF_PLUGIN:
        print(json.dumps({
            "status": "error",
            "error": f"huggingface-utils plugin not found: {_import_error}",
            "fix": "Ensure plugins/huggingface-utils/ exists. Run: python plugins/huggingface-utils/skills/hf-init/scripts/hf_init.py"
        }, indent=2), file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Persist Soul Memory to Hugging Face")
    parser.add_argument("--snapshot", type=str,
                        default=".agent/learning/learning_package_snapshot.md",
                        help="Path to sealed snapshot")
    parser.add_argument("--valence", type=float, default=0.0,
                        help="Moral/Emotional charge")
    parser.add_argument("--full-sync", action="store_true",
                        help="Perform full Soul JSONL genomic sync")
    args = parser.parse_args()

    try:
        config = get_hf_config()
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)

    if args.full_sync:
        # Delegate to forge-soul-exporter for full sync
        print(json.dumps({
            "status": "redirect",
            "message": "Full sync should use the forge-soul-exporter skill.",
            "command": f"python {PROJECT_ROOT}/plugins/obsidian-integration/skills/forge-soul-exporter/scripts/forge_soul.py --vault-root {PROJECT_ROOT} --full-sync"
        }, indent=2))
        return

    # Incremental: Upload snapshot + cache
    snapshot_path = PROJECT_ROOT / args.snapshot
    if not snapshot_path.exists():
        print(json.dumps({"status": "error", "error": f"Snapshot missing: {snapshot_path}"}, indent=2), file=sys.stderr)
        sys.exit(1)

    try:
        # Upload snapshot
        result = asyncio.run(upload_soul_snapshot(snapshot_path, args.valence, config))
        print(json.dumps({"status": "success", "snapshot": result.__dict__}, indent=2))

        # Sync RLM cache if it exists
        cache_file = PROJECT_ROOT / ".agent/learning/rlm_summary_cache.json"
        if cache_file.exists():
            cache_result = asyncio.run(upload_semantic_cache(cache_file, config))
            print(json.dumps({"cache_sync": cache_result.__dict__}, indent=2))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
