#!/usr/bin/env python3
#
# VERIFICATION SCAFFOLD (P101 Hardening)
# This script automates the Steward's hash verification for a commit manifest.

import sys
import json
import hashlib
import os

MANIFEST_PATH = "commit_manifest.json"

def verify_manifest():
    """Reads the manifest and verifies the SHA-256 hash of each file."""
    print("[VERIFY] Initiating Protocol 101 Manifest Verification...")

    if not os.path.exists(MANIFEST_PATH):
        print(f"\n[FATAL] COMMIT REJECTED: 'commit_manifest.json' not found.")
        print("         A Guardian-approved manifest is required.\n")
        sys.exit(1)

    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"\n[FATAL] COMMIT REJECTED: Could not parse '{MANIFEST_PATH}': {e}\n")
        sys.exit(1)

    files_to_verify = manifest.get('files', [])
    if not files_to_verify:
        print(f"\n[FATAL] COMMIT REJECTED: Manifest '{MANIFEST_PATH}' contains no files to verify.\n")
        sys.exit(1)

    print(f"[VERIFY] Found {len(files_to_verify)} files to verify.")
    all_verified = True
    for item in files_to_verify:
        filepath = item.get('path')
        expected_hash = item.get('sha256')

        if not os.path.exists(filepath):
            print(f"  - [FAIL] {filepath} -> File not found!")
            all_verified = False
            continue

        with open(filepath, 'rb') as f_to_hash:
            actual_hash = hashlib.sha256(f_to_hash.read()).hexdigest()

        if actual_hash == expected_hash:
            print(f"  - [PASS] {filepath}")
        else:
            print(f"  - [FAIL] {filepath}")
            print(f"    - Expected: {expected_hash}")
            print(f"    - Actual:   {actual_hash}")
            all_verified = False

    if all_verified:
        print("\n[SUCCESS] All files in manifest have been verified. Integrity confirmed.")
        sys.exit(0)
    else:
        print(f"\n[FATAL] COMMIT REJECTED: One or more files failed verification. Please review the errors above.\n")
        sys.exit(1)

if __name__ == "__main__":
    verify_manifest()