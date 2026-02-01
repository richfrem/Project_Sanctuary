#!/usr/bin/env python3
"""
cleanup.py (CLI)
=====================================

Purpose:
    Vector Cleanup: Consistency check to remove stale chunks from DB.

Layer: Curate / Vector

Usage Examples:
    python tools/curate/vector/cleanup.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    --apply         : Perform the deletion
    --prune-orphans : Remove entries not matching manifest
    --v             : Verbose mode

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - get_chroma_path(): No description.
    - load_manifest_globs(): Load include/exclude patterns from manifest.
    - matches_any(): Check if path matches any glob pattern or is inside a listed directory.
    - main(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import os
import sys
import argparse
import json
import fnmatch
from pathlib import Path

# ChromaDB
import chromadb
from chromadb.config import Settings

# Helper to get the Chroma DB path
def get_chroma_path():
    env_path = os.getenv("VECTOR_DB_PATH")
    if env_path:
        return Path(os.path.expanduser(env_path)).resolve()
    home = Path(os.path.expanduser("~"))
    return home / ".agent" / "learning" / "chroma_db"

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VECTOR_DB_PATH = get_chroma_path()
COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION", "child_chunks_v5")
MANIFEST_PATH = SCRIPT_DIR / "ingest_manifest.json"

def load_manifest_globs():
    """Load include/exclude patterns from manifest."""
    if not MANIFEST_PATH.exists():
        print("Manifest not found, skipping manifest checks.")
        return [], []
    
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    includes = manifest.get("include", [])
    excludes = manifest.get("exclude", [])
    return includes, excludes

def matches_any(path_str, patterns):
    """Check if path matches any glob pattern or is inside a listed directory."""
    for p in patterns:
        # Standard glob match
        if fnmatch.fnmatch(path_str, p):
            return True
        
        # Directory prefix match (e.g. "foo/bar" matches "foo/bar/baz.txt")
        # Ensure we match full directory names by checking for trailing slash or separator
        clean_p = p.rstrip('/')
        if path_str.startswith(clean_p + '/'):
            return True
        if path_str == clean_p:
            return True
            
    return False

def main():
    parser = argparse.ArgumentParser(description="Clean up Vector DB stale entries.")
    parser.add_argument("--apply", action="store_true", help="Perform the deletion")
    parser.add_argument("--prune-orphans", action="store_true", help="Remove entries not matching manifest")
    parser.add_argument("--v", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    print(f"Checking Vector DB at: {VECTOR_DB_PATH}")
    
    if not VECTOR_DB_PATH.exists():
        print("❌ Vector DB not found.")
        print("   Run: python tools/codify/vector/ingest.py --full")
        return

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Collection '{COLLECTION_NAME}' not found: {e}")
        return

    # Get all documents with their metadata
    print(f"📊 Collection: {COLLECTION_NAME}")
    total_chunks = collection.count()
    print(f"   Total chunks: {total_chunks}")

    if total_chunks == 0:
        print("   Collection is empty. Nothing to clean.")
        return

    # Fetch all documents (paginated if large)
    print("🔍 Scanning entries...")
    
    # Load manifest if pruning orphans
    includes, excludes = load_manifest_globs() if args.prune_orphans else ([], [])
    if args.prune_orphans:
        print(f"   Loaded manifest: {len(includes)} includes, {len(excludes)} excludes")

    # Get unique source files from metadata
    all_data = collection.get(include=["metadatas"])
    sources = set()
    id_to_source = {}
    
    for i, meta in enumerate(all_data['metadatas']):
        source = meta.get('source', '')
        if source:
            sources.add(source)
            # Track IDs for this source
            doc_id = all_data['ids'][i]
            if source not in id_to_source:
                id_to_source[source] = []
            id_to_source[source].append(doc_id)

    print(f"   Unique source files: {len(sources)}")

    # Check which files exist or match manifest
    entries_to_remove = []
    
    for rel_path in sources:
        full_path = PROJECT_ROOT / rel_path
        
        # 1. Check existence (Stale)
        if not full_path.exists():
            entries_to_remove.append(rel_path)
            if args.v:
                print(f"   [MISSING] {rel_path}")
            continue
            
        # 2. Check manifest (Orphan)
        if args.prune_orphans:
            is_included = matches_any(rel_path, includes)
            is_excluded = matches_any(rel_path, excludes)
            
            if not is_included or is_excluded:
                entries_to_remove.append(rel_path)
                if args.v:
                    reason = "EXCLUDED" if is_excluded else "NOT_INCLUDED"
                    print(f"   [ORPHAN-{reason}] {rel_path}")
                continue

        if args.v:
            print(f"   [OK] {rel_path}")

    remove_count = len(entries_to_remove)
    print(f"\nEntries to remove: {remove_count}")

    if remove_count == 0:
        print("✅ Vector DB is clean. No action needed.")
        return

    # Count chunks to remove
    ids_to_remove = []
    for source in entries_to_remove:
        ids_to_remove.extend(id_to_source.get(source, []))
    
    print(f"   Chunks to remove: {len(ids_to_remove)}")

    if args.apply:
        print(f"🗑️  Removing {len(ids_to_remove)} chunks...")
        
        # ChromaDB delete in batches
        batch_size = 5000
        for i in range(0, len(ids_to_remove), batch_size):
            batch = ids_to_remove[i:i + batch_size]
            collection.delete(ids=batch)
            print(f"   Deleted batch {i//batch_size + 1}: {len(batch)} chunks")
        
        print("✅ Vector DB cleaned successfully.")
        print(f"   New total: {collection.count()} chunks")
    else:
        print("\n⚠️  DRY RUN COMPLETE.")
        print("   To actually remove these entries, run:")
        if args.prune_orphans:
            print("   python tools/curate/vector/cleanup.py --apply --prune-orphans")
        else:
            print("   python tools/curate/vector/cleanup.py --apply")


if __name__ == "__main__":
    main()
