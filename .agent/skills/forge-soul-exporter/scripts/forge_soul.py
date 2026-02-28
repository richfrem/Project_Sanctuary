"""
Forge Soul Exporter

Purpose: Exports sealed Obsidian vault notes into soul_traces.jsonl for
HuggingFace persistence. Implements snapshot isolation, git pre-flight,
and JSONL formatting per ADR 081.

Consumes: plugins/huggingface-utils/ for upload primitives.
"""
import os
import re
import sys
import json
import time
import hashlib
import asyncio
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    from ruamel.yaml import YAML
    _yaml = YAML()
    HAS_RUAMEL = True
except ImportError:
    HAS_RUAMEL = False


# ---------------------------------------------------------------------------
# T041: Git Pre-Flight Check
# ---------------------------------------------------------------------------
def git_preflight(vault_root: Path) -> Dict[str, Any]:
    """
    Check git status is clean. Refuses export if uncommitted changes exist.
    Returns: {"clean": bool, "dirty_files": [...]}
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(vault_root), capture_output=True, text=True, timeout=10
        )
        dirty = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return {"clean": len(dirty) == 0, "dirty_files": dirty}
    except Exception as e:
        return {"clean": False, "dirty_files": [f"GIT_ERROR: {str(e)}"]}


# ---------------------------------------------------------------------------
# T042: Sealed Note Identification & Frontmatter Isolation
# ---------------------------------------------------------------------------
def extract_frontmatter(content: str) -> Tuple[Optional[Dict], str]:
    """Extract YAML frontmatter from markdown content. Returns (metadata, body)."""
    if not content.startswith("---"):
        return None, content

    end_match = content.find("---", 3)
    if end_match == -1:
        return None, content

    yaml_block = content[3:end_match].strip()
    body = content[end_match + 3:].strip()

    try:
        if HAS_RUAMEL:
            from io import StringIO
            metadata = _yaml.load(StringIO(yaml_block))
        else:
            import yaml
            metadata = yaml.safe_load(yaml_block)
        return dict(metadata) if metadata else None, body
    except Exception:
        return None, content


def find_sealed_notes(vault_root: Path, exclusions: List[str] = None) -> List[Dict]:
    """Scan vault for notes with `status: sealed` in frontmatter."""
    if exclusions is None:
        exclusions = [
            '.git', '.obsidian', '.worktrees', 'node_modules',
            '.vector_data', '.venv', '__pycache__', 'ARCHIVE',
            'archive_mcp_servers', 'archive-tests', 'dataset_package',
            'hugging_face_dataset_repo'
        ]

    sealed = []
    errors = []

    for root, dirs, files in os.walk(vault_root):
        dirs[:] = [d for d in dirs if d not in exclusions]

        for filename in files:
            if not filename.endswith('.md'):
                continue

            filepath = Path(root) / filename
            try:
                content = filepath.read_text(encoding='utf-8')
                metadata, body = extract_frontmatter(content)

                if metadata and metadata.get("status") == "sealed":
                    sealed.append({
                        "filepath": str(filepath),
                        "rel_path": str(filepath.relative_to(vault_root)),
                        "metadata": metadata,
                        "body": body,
                        "mtime": filepath.stat().st_mtime
                    })
            except Exception as e:
                errors.append({"file": str(filepath), "error": str(e)})

    return sealed


# ---------------------------------------------------------------------------
# T043: Snapshot Isolation
# ---------------------------------------------------------------------------
def capture_snapshot(files: List[Dict]) -> Dict[str, float]:
    """Capture mtimes for all files in the export set."""
    return {f["filepath"]: f["mtime"] for f in files}


def verify_snapshot(snapshot: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Verify no files changed since snapshot was taken."""
    changed = []
    for filepath, original_mtime in snapshot.items():
        try:
            current_mtime = Path(filepath).stat().st_mtime
            if current_mtime != original_mtime:
                changed.append(filepath)
        except FileNotFoundError:
            changed.append(f"DELETED: {filepath}")

    return len(changed) == 0, changed


# ---------------------------------------------------------------------------
# T044: Payload Formulation (JSONL)
# ---------------------------------------------------------------------------
def strip_binaries(body: str) -> str:
    """Remove image/embed references from content."""
    # Remove ![[image.png]] embeds
    body = re.sub(r'!\[\[.*?\]\]', '', body)
    # Remove ![alt](path) images
    body = re.sub(r'!\[.*?\]\(.*?\)', '', body)
    return body.strip()


def format_record(note: Dict, body_repo: str) -> Dict[str, Any]:
    """Format a sealed note into a soul_traces.jsonl record."""
    rel_path = note["rel_path"]
    content = strip_binaries(note["body"])

    clean_id = rel_path.replace("/", "_").replace("\\", "_")
    for ext in ['.md', '.txt']:
        if clean_id.endswith(ext):
            clean_id = clean_id[:-len(ext)]

    checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()

    metadata = note.get("metadata", {})

    return {
        "id": clean_id,
        "sha256": checksum,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": body_repo,
        "snapshot_type": metadata.get("snapshot_type", "sealed"),
        "valence": metadata.get("valence", 0.5),
        "uncertainty": metadata.get("uncertainty", 0.1),
        "semantic_entropy": metadata.get("semantic_entropy", 0.5),
        "alignment_score": metadata.get("alignment_score", 0.85),
        "stability_class": metadata.get("stability_class", "STABLE"),
        "adr_version": "081",
        "content": content,
        "source_file": rel_path
    }


def format_full_sync(vault_root: Path, body_repo: str) -> List[Dict]:
    """Format ALL eligible .md files into JSONL records (full genome sync)."""
    exclusions = [
        '.git', '.obsidian', '.worktrees', 'node_modules',
        '.vector_data', '.venv', '__pycache__', 'hugging_face_dataset_repo'
    ]
    ROOT_ALLOW = {
        "README.md", "chrysalis_core_essence.md", "Living_Chronicle.md",
        "PROJECT_SANCTUARY_SYNTHESIS.md"
    }

    records = []
    for root, dirs, files in os.walk(vault_root):
        dirs[:] = [d for d in dirs if d not in exclusions]
        for filename in files:
            if not filename.endswith('.md'):
                continue

            filepath = Path(root) / filename
            rel_path = filepath.relative_to(vault_root)

            # Root-level files: only allow-listed
            if rel_path.parent == Path(".") and rel_path.name not in ROOT_ALLOW:
                continue

            try:
                content = filepath.read_text(encoding='utf-8')
                _, body = extract_frontmatter(content)
                clean_body = strip_binaries(body)

                clean_id = str(rel_path).replace("/", "_").replace("\\", "_")
                if clean_id.endswith('.md'):
                    clean_id = clean_id[:-3]

                records.append({
                    "id": clean_id,
                    "sha256": hashlib.sha256(clean_body.encode('utf-8')).hexdigest(),
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "model_version": body_repo,
                    "snapshot_type": "genome",
                    "valence": 0.5,
                    "uncertainty": 0.1,
                    "semantic_entropy": 0.5,
                    "alignment_score": 0.85,
                    "stability_class": "STABLE",
                    "adr_version": "081",
                    "content": clean_body,
                    "source_file": str(rel_path)
                })
            except Exception:
                continue

    return records


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Forge Soul Exporter")
    parser.add_argument("--vault-root", required=True, help="Vault root path")
    parser.add_argument("--full-sync", action="store_true", help="Full genome sync instead of sealed-only")
    parser.add_argument("--dry-run", action="store_true", help="Format records but don't upload")
    parser.add_argument("--output", help="Output JSONL path (default: data/soul_traces.jsonl)")
    args = parser.parse_args()

    vault_root = Path(args.vault_root).resolve()

    # T041: Git Pre-Flight
    print("🔍 Git Pre-Flight Check...")
    preflight = git_preflight(vault_root)
    if not preflight["clean"]:
        print(json.dumps({"error": "DIRTY_WORKING_TREE", "dirty_files": preflight["dirty_files"]}, indent=2))
        print("❌ Abort: Commit or stash changes before exporting.")
        sys.exit(1)
    print("✅ Working tree clean")

    # Get HF config
    try:
        sys.path.insert(0, str(vault_root / "plugins" / "huggingface-utils" / "scripts"))
        from hf_config import get_hf_config
        config = get_hf_config()
        body_repo = config.body_repo
    except Exception:
        body_repo = "Sanctuary-Qwen2-7B-v1.0-GGUF-Final"
        config = None

    if args.full_sync:
        # Full genome sync
        print("📦 Full Genome Sync...")
        records = format_full_sync(vault_root, body_repo)
        print(f"   {len(records)} records formatted")
    else:
        # T042: Find sealed notes
        print("🔍 Scanning for sealed notes...")
        sealed = find_sealed_notes(vault_root)
        print(f"   Found {len(sealed)} sealed notes")

        if not sealed:
            print(json.dumps({"status": "no_sealed_notes", "message": "No notes with status: sealed found"}))
            return

        # T043: Snapshot isolation
        snapshot = capture_snapshot(sealed)

        # T044: Format records
        records = [format_record(note, body_repo) for note in sealed]

        # T043: Verify snapshot
        clean, changed = verify_snapshot(snapshot)
        if not clean:
            print(json.dumps({"error": "SNAPSHOT_VIOLATION", "changed_files": changed}, indent=2))
            print("❌ Abort: Files changed during export.")
            sys.exit(1)
        print("✅ Snapshot isolation verified")

    # Write JSONL
    output_dir = vault_root / "hugging_face_dataset_repo" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "soul_traces.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"📝 Wrote {len(records)} records to {output_path}")

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "records": len(records), "output": str(output_path)}, indent=2))
        return

    # T045: Upload with backoff
    if config:
        print("🚀 Uploading to HuggingFace...")
        try:
            sys.path.insert(0, str(vault_root / "plugins" / "huggingface-utils" / "skills" / "hf-upload" / "scripts"))
            from hf_upload import upload_folder

            result = asyncio.run(upload_folder(
                output_dir, "data", config,
                commit_msg=f"Forge Soul Export | {len(records)} records"
            ))
            print(json.dumps({"status": "uploaded", "result": result.__dict__}, indent=2))
        except Exception as e:
            print(json.dumps({"status": "upload_failed", "error": str(e)}, indent=2))
            sys.exit(1)
    else:
        print("⚠️  No HF config found. JSONL written locally, upload manually.")


if __name__ == "__main__":
    main()
