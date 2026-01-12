#!/usr/bin/env python3
"""
=============================================================================
 Map Repository Files - Project Sanctuary Link Checker
=============================================================================
 Purpose: Index all files in the repository for smart link fixing.
 Usage: python scripts/link-checker/map_repository_files.py
=============================================================================
"""
import os
import json
import sys
from pathlib import Path

# Add mcp_servers to path for shared config
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp_servers"))

# Import shared exclusion config from mcp_servers/lib
from lib.exclusion_config import EXCLUDE_DIR_NAMES, ALWAYS_EXCLUDE_FILES


def find_project_root() -> Path:
    """Find the Project Sanctuary root by looking for markers."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "IDENTITY").exists() or (current / "ADRs").exists():
            return current
        current = current.parent
    return Path.cwd()


def should_skip_file(filename: str) -> bool:
    """Check if file should be skipped based on exclusion config."""
    for excl in ALWAYS_EXCLUDE_FILES:
        if isinstance(excl, str):
            if filename == excl or filename.endswith(f'/{excl}'):
                return True
        elif hasattr(excl, 'match'):  # Compiled regex
            if excl.match(filename):
                return True
    return False


def generate_file_map(root_dir: Path) -> dict:
    """Generate a map of filename -> list of relative paths."""
    file_map = {}
    
    print(f"Mapping files in {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        # Use shared exclusion config
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        
        for filename in files:
            if should_skip_file(filename):
                continue
            # Skip binary files
            if filename.endswith(('.pyc', '.pyo', '.gguf', '.bin', '.safetensors')):
                continue
            if filename.startswith('.'):
                continue
                
            if filename not in file_map:
                file_map[filename] = []
            
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(root_dir)
            file_map[filename].append(str(rel_path))
            
    return file_map


def main():
    root_dir = find_project_root()
    file_map = generate_file_map(root_dir)
    
    output_file = Path(__file__).parent / 'file_inventory.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(file_map, f, indent=2, sort_keys=True)
        
    print(f"Inventory saved to {output_file}")
    print(f"Mapped {len(file_map)} unique filenames.")


if __name__ == "__main__":
    main()