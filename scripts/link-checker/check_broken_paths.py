#!/usr/bin/env python3
"""
=============================================================================
 Check Broken Paths - Project Sanctuary Link Checker
=============================================================================
 Purpose: Find broken relative links in markdown files.
 Usage: python scripts/link-checker/check_broken_paths.py
 
 Note: For a more comprehensive check including manifests, use:
       python scripts/link-checker/verify_links.py
=============================================================================
"""
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

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
    # Check static exclusions
    for excl in ALWAYS_EXCLUDE_FILES:
        if isinstance(excl, str):
            if filename == excl or filename.endswith(f'/{excl}'):
                return True
        elif hasattr(excl, 'match'):  # Compiled regex
            if excl.match(filename):
                return True
    return False


def find_files(root_dir: Path, extensions: list) -> list:
    """Recursively find files with specific extensions."""
    matches = []
    for root, dirs, files in os.walk(root_dir):
        # Use shared exclusion config for dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        
        for filename in files:
            if should_skip_file(filename):
                continue
            if any(filename.endswith(ext) for ext in extensions):
                matches.append(Path(root) / filename)
    return matches


def check_broken_links(file_path: Path, root_dir: Path) -> list:
    """Check for broken relative links in a file."""
    broken_links = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return []

    # Regex for Markdown links: [text](path)
    link_pattern = re.compile(r'\[.*?\]\((.*?)\)')
    
    # Also look for HTML-style links
    src_pattern = re.compile(r'(?:src|href)=[\'\"](.*?)[\'\"]')

    links = link_pattern.findall(content)
    links.extend(src_pattern.findall(content))

    for link in links:
        # Ignore web links, anchors, file:// URIs
        if link.startswith(('http', 'mailto:', '#', 'file://')):
            continue
            
        # Clean anchor tags
        clean_link = link.split('#')[0]
        if not clean_link:
            continue
            
        clean_link = unquote(clean_link)
        
        # Construct absolute path for verification
        target_path = file_path.parent / clean_link
        
        if not target_path.exists():
            # Fallback: check relative to root for paths starting with known dirs
            for known_dir in ['ADRs/', '01_PROTOCOLS/', 'LEARNING/', 'docs/', 'scripts/']:
                if clean_link.startswith(known_dir):
                    root_target = root_dir / clean_link
                    if root_target.exists():
                        break
            else:
                broken_links.append(link)

    return broken_links


def main():
    root_dir = find_project_root()
    log_file = Path(__file__).parent / "broken_links.log"
    
    print(f"Scanning for broken links in {root_dir}...\n")
    
    extensions = ['.md', '.markdown', '.mmd']
    files = find_files(root_dir, extensions)
    
    total_broken = 0
    
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"# Broken Links Report - Project Sanctuary\n\n")
        
        for file_path in files:
            broken = check_broken_links(file_path, root_dir)
            if broken:
                rel_path = file_path.relative_to(root_dir)
                log.write(f"## {rel_path}\n")
                print(f"FILE: {rel_path}")
                for link in broken:
                    log.write(f"- [ ] `{link}`\n")
                log.write("\n")
                total_broken += len(broken)
                
        log.write(f"\n---\nTotal: {total_broken} broken references.\n")
            
    print(f"\nScan complete. Found {total_broken} broken references.")
    print(f"Report: {log_file}")


if __name__ == "__main__":
    main()