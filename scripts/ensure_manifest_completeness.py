import json
import os
import re
from pathlib import Path

MANIFEST_PATH = ".agent/learning/learning_manifest.json"
PROJECT_ROOT = Path(".")

# Load Manifest
with open(MANIFEST_PATH, 'r') as f:
    manifest_entries = json.load(f)

print(f"Loaded {len(manifest_entries)} entries from {MANIFEST_PATH}")

# 1. Build Effective File List (Expand Directories)
effective_files = set()
explicit_entries = set(manifest_entries)

# Helper to check if a file is already "covered" by the manifest (either explicitly or via a dir)
def is_covered(file_rel_path_str):
    if file_rel_path_str in explicit_entries:
        return True
    # Check parent directories
    p = Path(file_rel_path_str)
    for parent in p.parents:
        # manifest entries for dirs might have trailing slash or not depending on convention
        # usually snapshot_utils expands dirs.
        # Let's check string matches
        parent_str = str(parent)
        parent_str_slash = parent_str + "/"
        if parent_str in explicit_entries or parent_str_slash in explicit_entries:
            return True
    return False

# Build list of files to SCAN for links
# We only assume we need to scan things that are currently in the manifest.
files_to_scan = []

for entry in manifest_entries:
    path = PROJECT_ROOT / entry.rstrip('/')
    if path.is_file():
        files_to_scan.append(path)
    elif path.is_dir():
        # Recursive glob
        for f in path.rglob("*"):
            if f.is_file():
                files_to_scan.append(f)

print(f"Scanning {len(files_to_scan)} files for .mmd references...")

# 2. Scan content for .mmd links
# Matches: ](path/to/file.mmd) or "path/to/file.mmd"
# Regex: (?:\]\(|'|")([^)'"]+\.mmd)(?:\)|'|")
mmd_regex = re.compile(r'(?:\]\(|[\'"])([^)\'"]+\.mmd)(?:\)|[\'"])')

missing_mmd_files = set()

for file_path in files_to_scan:
    if file_path.suffix not in ['.md', '.txt', '.py', '.json']:
        continue
        
    try:
        content = file_path.read_text(errors='ignore')
    except Exception as e:
        print(f"Warning: could not read {file_path}: {e}")
        continue
        
    matches = mmd_regex.findall(content)
    for match in matches:
        # Resolve path
        # Match might be relative to the file, or relative to root (if starts with / or is just a path)
        # But usually in our system, links are relative.
        
        # 1. Try relative to file
        resolved_path = None
        
        # Cleaning match (remove ../ etc)
        # If absolute path (unlikely in repo)
        if match.startswith('/'):
            # Assume relative to project root actually if it starts with / in a markdown link usually it means root?
            # Or just absolute path on FS.
            # Let's try treating as relative to root stripped of /
            candidate = PROJECT_ROOT / match.lstrip('/')
            if candidate.exists():
                resolved_path = candidate
        else:
            # Try relative to current file's dir
            candidate_rel = file_path.parent / match
            if candidate_rel.resolve().exists():
                resolved_path = candidate_rel.resolve().relative_to(PROJECT_ROOT.resolve())
            else:
                # Try relative to project root directly (sometimes people put paths that way)
                candidate_root = PROJECT_ROOT / match
                if candidate_root.exists():
                    resolved_path = candidate_root
        
        if resolved_path:
            # Convert to string relative to root
            try:
                rel_path_str = str(resolved_path) # already made relative above if possible
                # double check relative status
                if resolved_path.is_absolute():
                     rel_path_str = str(resolved_path.relative_to(PROJECT_ROOT.resolve()))
            except ValueError:
                 # path not in tree?
                 continue

            # Check if covered
            if not is_covered(rel_path_str):
                # Avoid duplicates
                if rel_path_str not in missing_mmd_files:
                    print(f"Found missing dependency: {rel_path_str} (referenced in {file_path.relative_to(PROJECT_ROOT)})")
                    missing_mmd_files.add(rel_path_str)

# 3. Update Manifest
if missing_mmd_files:
    print(f"\nAdding {len(missing_mmd_files)} missing diagrams to manifest...")
    manifest_entries.extend(list(missing_mmd_files))
    
    # Optional: Sort or clean up?
    # Keeping original order + appended for diff clarity might be better, or just sorting.
    # JSON usually loaded as list.
    
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest_entries, f, indent=4)
    print("Manifest updated.")
else:
    print("\nNo missing .mmd dependencies found.")
