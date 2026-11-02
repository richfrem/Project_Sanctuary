#!/usr/bin/env python3
"""
Trace Extractor (`trace_extractor.py`)
======================================
Extracts, formats, and seals Project Sanctuary markdown documents and session
traces into the canonical JSONL dataset (`soul_traces.jsonl`) per ADR 081 & 084.

Features:
  - Frontmatter parsing & metadata normalization
  - Content sanitization (binary/image embed stripping)
  - SHA-256 integrity digest generation
  - Deduplication and upsert handling
  - Manifest generation/updates (`metadata/manifest.json`)
  - Schema verification mode (`--verify`)
"""

import os
import re
import sys
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

CANONICAL_SCHEMA_FIELDS = {
    "id": str,
    "sha256": str,
    "timestamp": str,
    "model_version": str,
    "snapshot_type": str,
    "valence": (int, float),
    "uncertainty": (int, float),
    "semantic_entropy": (int, float),
    "alignment_score": (int, float),
    "stability_class": str,
    "adr_version": str,
    "content": str,
    "source_file": str,
}

EXCLUSIONS = {
    ".git", ".obsidian", ".worktrees", "node_modules",
    ".vector_data", ".venv", "__pycache__", "ARCHIVE",
    "archive_mcp_servers", "archive-tests", "dataset_package",
    "hugging_face_dataset_repo", "build", "dist"
}

ROOT_ALLOW = {
    "README.md", "AGENTS.md", "CLAUDE.md", "GEMINI.md"
}


def extract_frontmatter(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract YAML frontmatter if present without third-party dependencies."""
    if not content.startswith("---"):
        return None, content

    end_idx = content.find("---", 3)
    if end_idx == -1:
        return None, content

    yaml_block = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()

    metadata: Dict[str, Any] = {}
    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            # Parse numbers / booleans if applicable
            if val.lower() == "true":
                metadata[key] = True
            elif val.lower() == "false":
                metadata[key] = False
            else:
                try:
                    if "." in val:
                        metadata[key] = float(val)
                    else:
                        metadata[key] = int(val)
                except ValueError:
                    metadata[key] = val

    return metadata if metadata else None, body


def sanitize_markdown_content(body: str) -> str:
    """Strip binary/image embeds to keep dataset lean and text-focused."""
    # Strip ![[image.png]]
    body = re.sub(r"!\[\[.*?\]\]", "", body)
    # Strip ![alt](path)
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    return body.strip()


def sanitize_record_id(rel_path: str) -> str:
    """Convert relative filepath to a standardized alphanumeric record ID."""
    clean_id = rel_path.replace("/", "_").replace("\\", "_")
    for ext in [".md", ".txt"]:
        if clean_id.endswith(ext):
            clean_id = clean_id[:-len(ext)]
    return clean_id


def build_soul_trace_record(
    filepath: Path,
    repo_root: Path,
    model_version: str = "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
    snapshot_type: str = "genome"
) -> Optional[Dict[str, Any]]:
    """Build a validated soul trace record from a given markdown file."""
    try:
        raw_text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        sys.stderr.write(f"Error reading {filepath}: {e}\n")
        return None

    meta, body = extract_frontmatter(raw_text)
    clean_content = sanitize_markdown_content(body)

    if not clean_content:
        return None

    try:
        rel_path = str(filepath.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        rel_path = filepath.name

    record_id = sanitize_record_id(rel_path)
    content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    record_meta = meta or {}
    record = {
        "id": record_id,
        "sha256": content_hash,
        "timestamp": now_iso,
        "model_version": model_version,
        "snapshot_type": record_meta.get("snapshot_type", snapshot_type),
        "valence": float(record_meta.get("valence", 0.5)),
        "uncertainty": float(record_meta.get("uncertainty", 0.1)),
        "semantic_entropy": float(record_meta.get("semantic_entropy", 0.5)),
        "alignment_score": float(record_meta.get("alignment_score", 0.85)),
        "stability_class": str(record_meta.get("stability_class", "STABLE")),
        "adr_version": str(record_meta.get("adr_version", "084")),
        "content": clean_content,
        "source_file": rel_path
    }
    return record


def verify_dataset(dataset_path: Path, repair: bool = False) -> Tuple[bool, List[str]]:
    """Verify an existing JSONL dataset for schema compliance and hash validity."""
    if not dataset_path.exists():
        return False, [f"Dataset not found at {dataset_path}"]

    errors = []
    line_count = 0
    records = []
    repaired_count = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                errors.append(f"Line {idx}: JSON decode error: {e}")
                continue

            # Field presence and type check
            for field, expected_type in CANONICAL_SCHEMA_FIELDS.items():
                if field not in rec:
                    errors.append(f"Line {idx} (id={rec.get('id', '?')}): Missing field '{field}'")
                elif not isinstance(rec[field], expected_type):
                    errors.append(
                        f"Line {idx} (id={rec.get('id', '?')}): Field '{field}' expected {expected_type}, got {type(rec[field])}"
                    )

            # Checksum verification
            if "content" in rec and "sha256" in rec:
                calc_hash = hashlib.sha256(rec["content"].encode("utf-8")).hexdigest()
                if calc_hash != rec["sha256"]:
                    if repair:
                        rec["sha256"] = calc_hash
                        repaired_count += 1
                    else:
                        errors.append(
                            f"Line {idx} (id={rec.get('id', '?')}): SHA-256 mismatch (expected {rec['sha256']}, calculated {calc_hash})"
                        )

            records.append(rec)

    if line_count == 0:
        errors.append("Dataset file is empty")

    if repair and repaired_count > 0 and len(errors) == 0:
        with open(dataset_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Repaired {repaired_count} SHA-256 digests in {dataset_path.name}")
        return True, []

    return len(errors) == 0, errors


def update_manifest(manifest_path: Path, dataset_path: Path, model_version: str) -> None:
    """Update dataset manifest.json with updated record count and hashes."""
    count = 0
    hasher = hashlib.sha256()
    if dataset_path.exists():
        with open(dataset_path, "rb") as f:
            for line in f:
                if line.strip():
                    count += 1
                    hasher.update(line)

    dataset_digest = hasher.hexdigest()
    manifest_data = {
        "version": "1.0",
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot_count": count,
        "dataset_sha256": dataset_digest,
        "model_lineage": model_version,
        "schema_compliance": "ADR_084_CANONICAL"
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Sanctuary Soul Trace Extractor")
    parser.add_argument("--repo-root", default=".", help="Project repository root")
    parser.add_argument("--source-dir", help="Specific directory to extract (e.g. chronicle/entries, wiki, protocols)")
    parser.add_argument("--target-file", help="Specific file to extract")
    parser.add_argument("--output", default="hugging_face_dataset_repo/data/soul_traces.jsonl", help="Output JSONL path")
    parser.add_argument("--manifest", default="hugging_face_dataset_repo/metadata/manifest.json", help="Manifest path")
    parser.add_argument("--model-version", default="Sanctuary-Qwen2-7B-v1.0-GGUF-Final", help="Model lineage identifier")
    parser.add_argument("--snapshot-type", default="genome", help="Default snapshot type")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records extracted (0 = all)")
    parser.add_argument("--append", action="store_true", help="Upsert into existing JSONL instead of overwriting")
    parser.add_argument("--dry-run", action="store_true", help="Preview extraction without writing")
    parser.add_argument("--verify", action="store_true", help="Verify dataset integrity and exit")
    parser.add_argument("--repair", action="store_true", help="When verifying, repair invalid SHA-256 digests in-place")

    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()

    if args.verify or args.repair:
        passed, errors = verify_dataset(output_path, repair=args.repair)
        if passed:
            print(f"VERIFICATION SUCCESS: {output_path} conforms to ADR 084 canonical schema.")
            sys.exit(0)
        else:
            print(f"VERIFICATION FAILED: {len(errors)} errors found:")
            for e in errors[:10]:
                print(f"  - {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more.")
            sys.exit(1)

    # Collect source files
    target_files: List[Path] = []
    if args.target_file:
        tf = Path(args.target_file).resolve()
        if tf.exists() and tf.is_file():
            target_files.append(tf)
    elif args.source_dir:
        sd = Path(args.source_dir).resolve()
        for root, dirs, files in os.walk(sd):
            dirs[:] = [d for d in dirs if d not in EXCLUSIONS]
            for f in files:
                if f.endswith(".md"):
                    target_files.append(Path(root) / f)
    else:
        # Default scan across canonical domains
        canonical_dirs = ["chronicle/entries", "protocols", "wiki", "context"]
        for cdir in canonical_dirs:
            sd = repo_root / cdir
            if sd.exists():
                for root, dirs, files in os.walk(sd):
                    dirs[:] = [d for d in dirs if d not in EXCLUSIONS]
                    for f in files:
                        if f.endswith(".md"):
                            target_files.append(Path(root) / f)
        # Add root allowlist
        for rf in ROOT_ALLOW:
            rp = repo_root / rf
            if rp.exists():
                target_files.append(rp)

    target_files.sort()
    if args.limit > 0:
        target_files = target_files[:args.limit]

    print(f"Identified {len(target_files)} eligible markdown files for extraction.")

    # Extract records
    records_by_id: Dict[str, Dict[str, Any]] = {}

    # If appending, load existing records first
    if args.append and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if "id" in rec:
                            records_by_id[rec["id"]] = rec
                    except Exception:
                        pass
        print(f"Loaded {len(records_by_id)} existing records from {output_path.name}.")

    extracted_count = 0
    for tf in target_files:
        rec = build_soul_trace_record(tf, repo_root, args.model_version, args.snapshot_type)
        if rec:
            records_by_id[rec["id"]] = rec
            extracted_count += 1

    print(f"Extracted {extracted_count} records. Total dataset size: {len(records_by_id)} records.")

    if args.dry_run:
        print("[DRY-RUN] Sample record:")
        if records_by_id:
            first_key = next(iter(records_by_id))
            sample = records_by_id[first_key].copy()
            sample["content"] = sample["content"][:120] + "... [TRUNCATED]"
            print(json.dumps(sample, indent=2))
        return

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records_by_id.values():
            f.write(json.dumps(rec) + "\n")

    print(f"Saved dataset to {output_path}")

    # Update manifest
    update_manifest(manifest_path, output_path, args.model_version)
    print(f"Updated dataset manifest at {manifest_path}")


if __name__ == "__main__":
    main()
