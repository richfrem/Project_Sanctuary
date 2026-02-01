#!/usr/bin/env python3
"""
tools/retrieve/bundler/validate.py
=====================================

Purpose:
    Validate manifest files against the context bundler schema.
    Part of Red Team P0 fix - ensures manifests are valid before bundling.

Layer: Retrieve / Bundler

Usage:
    python tools/retrieve/bundler/validate.py manifest.json
    python tools/retrieve/bundler/validate.py --all-base
    python tools/retrieve/bundler/validate.py --check-index

Key Functions:
    - validate_manifest(): Validate single manifest against schema
    - validate_all_base(): Validate all registered base manifests
    - validate_index(): Validate base-manifests-index.json integrity
"""

import json
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Resolve paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.resolve()
BUNDLER_DIR = PROJECT_ROOT / "tools" / "standalone" / "context-bundler"
BASE_MANIFESTS_DIR = BUNDLER_DIR / "base-manifests"
SCHEMA_PATH = BUNDLER_DIR / "file-manifest-schema.json"
INDEX_PATH = BASE_MANIFESTS_DIR.parent / "base-manifests-index.json"


def load_schema() -> Dict[str, Any]:
    """Load the manifest schema."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest_path: Path, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate a manifest file against the schema.
    
    Args:
        manifest_path: Path to the manifest JSON file.
        verbose: Print validation results.
    
    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []
    
    # Check file exists
    if not manifest_path.exists():
        errors.append(f"File not found: {manifest_path}")
        return False, errors
    
    # Check valid JSON
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors
    
    # Check required fields
    if "title" not in data:
        errors.append("Missing required field: 'title'")
    elif not isinstance(data["title"], str) or not data["title"].strip():
        errors.append("Field 'title' must be a non-empty string")
    
    if "files" not in data:
        errors.append("Missing required field: 'files'")
    elif not isinstance(data["files"], list):
        errors.append("Field 'files' must be an array")
    elif len(data["files"]) == 0:
        errors.append("Field 'files' must contain at least one entry (first should be prompt)")
    else:
        # Validate each file entry
        for i, entry in enumerate(data["files"]):
            if isinstance(entry, str):
                # Legacy format - warn but allow
                if verbose:
                    print(f"  ⚠️  files[{i}] uses legacy string format, should be {{path, note}}")
            elif isinstance(entry, dict):
                if "path" not in entry:
                    errors.append(f"files[{i}] missing required 'path' field")
                elif not isinstance(entry["path"], str) or not entry["path"].strip():
                    errors.append(f"files[{i}].path must be a non-empty string")
                # Path traversal check
                path_str = entry.get("path", "")
                if ".." in path_str:
                    errors.append(f"files[{i}].path contains path traversal: {path_str}")
            else:
                errors.append(f"files[{i}] must be string or object, got {type(entry).__name__}")
    
    # Optional field validation
    if "description" in data and not isinstance(data.get("description"), (str, type(None))):
        errors.append("Field 'description' must be a string or null")
    
    is_valid = len(errors) == 0
    
    if verbose:
        if is_valid:
            print(f"✅ {manifest_path.name} is valid")
        else:
            print(f"❌ {manifest_path.name} has {len(errors)} error(s):")
            for err in errors:
                print(f"   - {err}")
    
    return is_valid, errors


def validate_index(verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate the base-manifests-index.json integrity.
    
    Checks:
    - Index file exists and is valid JSON
    - All referenced base manifest files exist
    - All referenced files are valid manifests
    """
    errors = []
    
    if not INDEX_PATH.exists():
        errors.append(f"Index file not found: {INDEX_PATH}")
        return False, errors
    
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in index: {e}")
        return False, errors
    
    if not isinstance(index, dict):
        errors.append("Index must be a JSON object")
        return False, errors
    
    for type_name, filename in index.items():
        if not isinstance(filename, str):
            errors.append(f"Index['{type_name}'] must be a string filename")
            continue
        
        manifest_path = BASE_MANIFESTS_DIR / filename
        if not manifest_path.exists():
            errors.append(f"Index['{type_name}'] references missing file: {filename}")
        else:
            is_valid, manifest_errors = validate_manifest(manifest_path, verbose=False)
            if not is_valid:
                errors.append(f"Index['{type_name}'] references invalid manifest: {filename}")
                errors.extend([f"  - {e}" for e in manifest_errors])
    
    is_valid = len(errors) == 0
    
    if verbose:
        if is_valid:
            print(f"✅ Index is valid ({len(index)} types registered)")
        else:
            print(f"❌ Index has {len(errors)} error(s):")
            for err in errors:
                print(f"   {err}")
    
    return is_valid, errors


def validate_all_base(verbose: bool = True) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate all base manifests in the base-manifests directory.
    
    Returns:
        Tuple of (all_valid, dict of filename -> errors).
    """
    results = {}
    all_valid = True
    
    if not BASE_MANIFESTS_DIR.exists():
        print(f"❌ Base manifests directory not found: {BASE_MANIFESTS_DIR}")
        return False, {"_directory": ["Directory not found"]}
    
    for manifest_path in BASE_MANIFESTS_DIR.glob("*.json"):
        is_valid, errors = validate_manifest(manifest_path, verbose=verbose)
        results[manifest_path.name] = errors
        if not is_valid:
            all_valid = False
    
    if verbose:
        print(f"\n{'✅' if all_valid else '❌'} Validated {len(results)} base manifests")
    
    return all_valid, results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate context bundler manifests")
    parser.add_argument("manifest", nargs="?", help="Path to manifest JSON file")
    parser.add_argument("--all-base", action="store_true", help="Validate all base manifests")
    parser.add_argument("--check-index", action="store_true", help="Validate base-manifests-index.json")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output, exit code only")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.check_index:
        is_valid, _ = validate_index(verbose=verbose)
        sys.exit(0 if is_valid else 1)
    
    if args.all_base:
        all_valid, _ = validate_all_base(verbose=verbose)
        sys.exit(0 if all_valid else 1)
    
    if args.manifest:
        is_valid, _ = validate_manifest(Path(args.manifest), verbose=verbose)
        sys.exit(0 if is_valid else 1)
    
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
