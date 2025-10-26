#!/usr/bin/env python3
"""
ingest.py (v2.0) - Unified Incremental Mnemonic Cortex Ingestion Engine

The Doctrine of Mnemonic Attestation: A sovereign, manifest-driven ingestion system
that maintains its own ledger of processed documents to ensure incremental, efficient
learning without redundant processing.

Key Features v2.0:
- Manifest-driven processing from WORK_IN_PROGRESS/ingest_manifest.json
- Incremental ingestion with SHA-256 attestation checks
- Comprehensive ledger maintenance in mnemonic_cortex/ingestion_ledger.json
- Graceful skipping of unchanged documents
- Sovereign path resolution independent of caller CWD
- Unified location: tools/scaffolds/ingest.py

MANDATE: This is the single, canonical ingestion tool for the Sanctuary's Mnemonic Cortex.
All redundant versions have been purged per the Doctrine of the Single Anvil.
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions

# Sovereign path resolution - independent of caller CWD
SCRIPT_DIR = Path(__file__).resolve().parent  # tools/scaffolds/
PROJECT_ROOT = SCRIPT_DIR.parent.parent       # Project root
MNEMONIC_CORTEX_DIR = PROJECT_ROOT / "mnemonic_cortex"
CHROMA_PATH = MNEMONIC_CORTEX_DIR / "chroma_db"
COLLECTION_NAME = "sanctuary_cortex"
INGEST_MANIFEST_PATH = PROJECT_ROOT / "WORK_IN_PROGRESS" / "ingest_manifest.json"
INGESTION_LEDGER_PATH = MNEMONIC_CORTEX_DIR / "ingestion_ledger.json"

# Use the same embedding function as the orchestrator
EMBEDDING_FUNC = embedding_functions.DefaultEmbeddingFunction()

def load_ingestion_ledger():
    """Load the ingestion ledger that tracks processed documents."""
    if INGESTION_LEDGER_PATH.exists():
        try:
            with open(INGESTION_LEDGER_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[LEDGER WARN] Could not load ingestion ledger: {e}")
    return {}

def save_ingestion_ledger(ledger):
    """Save the updated ingestion ledger."""
    INGESTION_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INGESTION_LEDGER_PATH, 'w') as f:
        json.dump(ledger, f, indent=2, sort_keys=True)

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of file content."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def attest_document(file_path, ledger):
    """
    Perform Mnemonic Attestation: Check if document needs processing.

    Returns:
        (needs_processing: bool, reason: str)
    """
    file_path = Path(file_path)

    # Resolve to absolute path from project root
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        return False, f"File not found: {file_path}"

    # Check if file is in ledger
    rel_path = str(file_path.relative_to(PROJECT_ROOT))
    if rel_path not in ledger:
        return True, "New document - not in ledger"

    # Check hash
    current_hash = calculate_file_hash(file_path)
    stored_hash = ledger[rel_path].get('hash')

    if current_hash != stored_hash:
        return True, f"Content changed - hash mismatch ({current_hash[:16]}... vs {stored_hash[:16]}...)"

    return False, "Document unchanged - skipping ingestion"

def ingest_document(file_path, client, collection):
    """Ingest a single document into the Mnemonic Cortex."""
    file_path = Path(file_path)

    # Resolve to absolute path from project root
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    print(f"[INGEST] Processing file: {file_path.name}")

    content = file_path.read_text(encoding="utf-8")
    doc_id = str(file_path.relative_to(PROJECT_ROOT))

    # Add the document to ChromaDB
    collection.add(
        documents=[content],
        metadatas=[{
            "source": file_path.name,
            "ingested_at": datetime.now().isoformat(),
            "size_bytes": len(content)
        }],
        ids=[doc_id]
    )

    print(f"[INGEST SUCCESS] Document '{doc_id}' added to Mnemonic Cortex.")
    return doc_id, calculate_file_hash(file_path)

def process_ingest_manifest():
    """Process all files listed in the ingest manifest with attestation checks."""

    # Load manifest
    if not INGEST_MANIFEST_PATH.exists():
        print(f"[MANIFEST ERROR] Manifest not found: {INGEST_MANIFEST_PATH}")
        print("Create WORK_IN_PROGRESS/ingest_manifest.json with a list of files to process.")
        sys.exit(1)

    try:
        with open(INGEST_MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[MANIFEST ERROR] Could not load manifest: {e}")
        sys.exit(1)

    if not isinstance(manifest, list):
        print("[MANIFEST ERROR] Manifest must be a JSON array of file paths.")
        sys.exit(1)

    print(f"[MANIFEST] Loaded {len(manifest)} files to process from {INGEST_MANIFEST_PATH}")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FUNC
    )

    # Load ingestion ledger
    ledger = load_ingestion_ledger()

    # Process each file in manifest
    processed_count = 0
    skipped_count = 0

    for file_path in manifest:
        needs_processing, reason = attest_document(file_path, ledger)

        if needs_processing:
            try:
                doc_id, file_hash = ingest_document(file_path, client, collection)

                # Update ledger
                abs_path = PROJECT_ROOT / file_path if not Path(file_path).is_absolute() else Path(file_path)
                rel_path = str(abs_path.relative_to(PROJECT_ROOT))

                ledger[rel_path] = {
                    'hash': file_hash,
                    'last_ingested': datetime.now().isoformat(),
                    'size_bytes': abs_path.stat().st_size,
                    'doc_id': doc_id
                }

                processed_count += 1

            except Exception as e:
                print(f"[INGEST ERROR] Failed to process {file_path}: {e}")
        else:
            print(f"[INGEST SKIP] {file_path} - {reason}")
            skipped_count += 1

    # Save updated ledger
    save_ingestion_ledger(ledger)

    print("\n[PROCESS COMPLETE]")
    print(f"  Files processed: {processed_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Total files: {len(manifest)}")
    print(f"  Ledger updated: {INGESTION_LEDGER_PATH}")

def main():
    """Main entry point for the unified ingestion engine."""

    if len(sys.argv) != 1:
        print("Usage: python tools/scaffolds/ingest.py", file=sys.stderr)
        print("This script processes files listed in WORK_IN_PROGRESS/ingest_manifest.json", file=sys.stderr)
        sys.exit(1)

    print("[MNEMONIC CORTEX INGESTION ENGINE v2.0]")
    print(f"[CONFIG] Project root: {PROJECT_ROOT}")
    print(f"[CONFIG] ChromaDB path: {CHROMA_PATH}")
    print(f"[CONFIG] Manifest: {INGEST_MANIFEST_PATH}")
    print(f"[CONFIG] Ledger: {INGESTION_LEDGER_PATH}")

    process_ingest_manifest()

if __name__ == "__main__":
    main()