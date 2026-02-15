#!/usr/bin/env python3
"""
find_json_duplicates.py (CLI)
=====================================

Purpose:
    Finds duplicate entries across JSON inventory files.

Layer: Curate / Utilities

Usage Examples:
    python tools/curate/hygiene/find_json_duplicates.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    (None detected)

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - find_duplicates(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import re
import collections
import os

FILE_PATH = r'legacy-system\reference-data\ai_analysis_tracking.json'

def find_duplicates():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex to capture keys that start an object definition like "KEY": {
    # This captures the top-level keys for forms/reports
    keys = re.findall(r'\"([A-Z0-9]+)\"\s*:\s*\{', content)
    
    counts = collections.Counter(keys)
    duplicates = [k for k, v in counts.items() if v > 1]
    
    if duplicates:
        print(", ".join(sorted(duplicates)))
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    find_duplicates()
