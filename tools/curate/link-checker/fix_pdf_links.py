#!/usr/bin/env python3
"""
fix_pdf_links.py (CLI)
=====================================

Purpose:
    Scans markdown files and fixes broken PDF links by URL-encoding spaces.

Layer: Curate / Utilities

Usage Examples:
    python tools/curate/link-checker/fix_pdf_links.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    (None detected)

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - fix_pdf_links_in_file(): Fix PDF links with spaces in a markdown file.
    - main(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import os
import re
from pathlib import Path

# Base directory for overview files
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "legacy-system" / "oracle-forms-overviews"


def fix_pdf_links_in_file(file_path: Path) -> bool:
    """
    Fix PDF links with spaces in a markdown file.
    
    Returns True if file was modified, False otherwise.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Pattern matches markdown links to PDF files with spaces
    # e.g., [Courts Functional Document V1.0](../sourcedocuments/Courts Functional Document V1.0.pdf)
    def encode_spaces_in_url(match):
        text = match.group(1)
        url = match.group(2)
        # Only encode spaces in URLs that contain spaces and end with .pdf
        if ' ' in url and url.lower().endswith('.pdf'):
            encoded_url = url.replace(' ', '%20')
            return f'[{text}]({encoded_url})'
        return match.group(0)
    
    # Match markdown links: [text](url)
    content = re.sub(r'\[([^\]]+)\]\(([^)]+\.pdf)\)', encode_spaces_in_url, content, flags=re.IGNORECASE)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def main():
    print("Fixing PDF links with spaces...")
    
    modified_count = 0
    
    # Process all subdirectories
    for subdir in ['forms', 'reports', 'archived']:
        dir_path = BASE_DIR / subdir
        if not dir_path.exists():
            continue
        
        for md_file in dir_path.glob('*-Overview.md'):
            if fix_pdf_links_in_file(md_file):
                print(f"Fixed: {md_file.name}")
                modified_count += 1
    
    print(f"\nComplete. Modified {modified_count} files.")


if __name__ == "__main__":
    main()
