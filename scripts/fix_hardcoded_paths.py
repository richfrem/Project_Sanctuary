"""
Script to fix all hardcoded absolute paths in Project Sanctuary.

Replaces hardcoded paths like /Users/richardfremmerlid/Projects/Project_Sanctuary
with computed relative paths using Path(__file__).resolve().parent pattern.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Hardcoded path to find
HARDCODED_PATH = "/Users/richardfremmerlid/Projects/Project_Sanctuary"

def fix_file(file_path: Path) -> Tuple[bool, str]:
    """
    Fix hardcoded paths in a single file.
    
    Returns:
        (changed, message) tuple
    """
    try:
        content = file_path.read_text()
        
        if HARDCODED_PATH not in content:
            return False, "No hardcoded paths found"
        
        # Count occurrences
        count = content.count(HARDCODED_PATH)
        
        # Compute relative depth from file to project root
        # e.g., 05_ARCHIVED_BLUEPRINTS/gardener_pytorch_rl_v1/file.py -> ../../..
        relative_to_root = file_path.relative_to(PROJECT_ROOT)
        depth = len(relative_to_root.parents) - 1
        
        # Replace with computed path
        # For default parameters, use None and compute at runtime
        new_content = content.replace(
            f'= "{HARDCODED_PATH}"',
            f'= None  # Computed at runtime from Path(__file__)'
        )
        
        # Also replace in __init__ signatures
        new_content = new_content.replace(
            f'repo_path: str = "{HARDCODED_PATH}"',
            f'repo_path: str = None'
        )
        new_content = new_content.replace(
            f'sanctuary_root: str = "{HARDCODED_PATH}"',
            f'sanctuary_root: str = None'
        )
        new_content = new_content.replace(
            f'environment_path: str = "{HARDCODED_PATH}"',
            f'environment_path: str = None'
        )
        
        # Replace Path() constructors
        new_content = new_content.replace(
            f'Path("{HARDCODED_PATH}")',
            f'Path(__file__).resolve().parent.parent.parent'
        )
        
        # Write back
        file_path.write_text(new_content)
        
        return True, f"Fixed {count} occurrences"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Fix all Python files with hardcoded paths."""
    
    # Directories to check
    dirs_to_check = [
        PROJECT_ROOT / "05_ARCHIVED_BLUEPRINTS",
        PROJECT_ROOT / "EXPERIMENTS"
    ]
    
    fixed_files = []
    
    for directory in dirs_to_check:
        if not directory.exists():
            print(f"Skipping {directory} (doesn't exist)")
            continue
            
        print(f"\nChecking {directory}...")
        
        for py_file in directory.rglob("*.py"):
            changed, message = fix_file(py_file)
            if changed:
                fixed_files.append(py_file)
                print(f"  ✓ {py_file.relative_to(PROJECT_ROOT)}: {message}")
    
    print(f"\n=== Summary ===")
    print(f"Fixed {len(fixed_files)} files")
    for f in fixed_files:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
