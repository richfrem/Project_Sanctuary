import json
import os
from pathlib import Path

manifest_path = ".agent/learning/learning_manifest.json"
project_root = Path(".")

print(f"Validating {manifest_path}...")

try:
    with open(manifest_path, 'r') as f:
        files = json.load(f)
except Exception as e:
    print(f"Error loading manifest: {e}")
    exit(1)

missing = []
for file_path in files:
    # Handle directory references (some might end in /)
    clean_path = file_path.rstrip('/')
    path = project_root / clean_path
    
    if not path.exists():
        print(f"❌ MISSING: {file_path}")
        missing.append(file_path)
    else:
        print(f"✅ FOUND: {file_path}")

if missing:
    print(f"\nFound {len(missing)} missing files.")
    exit(1)
else:
    print("\nAll files verified successfully.")
    exit(0)
