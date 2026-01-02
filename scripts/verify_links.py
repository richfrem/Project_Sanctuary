#!/usr/bin/env python3
import os
import re
import json
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urlparse, unquote

# Regex for Markdown links: [text](link) or [text]: link
# Refined to avoid picking up arbitrary colons in sentences.
# The second part for [id]: link now expects a path-like (with / or .) or URL-like string.
MD_LINK_PATTERN = re.compile(r'\[.*?\]\((.*?)\)|^\s*\[(.*?)\]:\s*([./\w\-_]*[./][./\w\-_]*|https?://\S+)', re.MULTILINE)
# Pattern for URL detection (http/https)
URL_PATTERN = re.compile(r'https?://[^\s)\]]+')

def is_external(link: str) -> bool:
    return link.startswith(('http://', 'https://'))

def resolve_relative_path(current_file: Path, link: str, project_root: Path) -> Path:
    # Remove fragments and query params
    clean_link = link.split('#')[0].split('?')[0]
    if not clean_link:
        return current_file # Self-reference anchor

    # Unquote URL-encoded characters (like %20 for space)
    clean_link = unquote(clean_link)

    # Handle file:// schema
    if clean_link.startswith('file://'):
        clean_link = clean_link.replace('file://', '')
        # Absolute from project root
        return project_root / clean_link.lstrip('/')
    else:
        # Try relative to current file first
        rel_path = (current_file.parent / clean_link).resolve()
        if not rel_path.exists():
            # Fallback: Check if it's root-relative despite lacking a leading slash
            # This is common in some markdown environments.
            root_rel_path = (project_root / clean_link).resolve()
            if root_rel_path.exists():
                return root_rel_path
        return rel_path

def check_external_link(url: str, timeout: int = 5) -> Tuple[bool, str]:
    try:
        # Some sites block generic user agents, so we add one
        headers = {'User-Agent': 'Mozilla/5.0 (SanctuaryLinkChecker/1.0)'}
        response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        if response.status_code >= 400:
            # Try GET if HEAD fails (some servers block HEAD)
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        
        if response.status_code < 400:
            return True, "OK"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def scan_md_file(file_path: Path, project_root: Path, check_external: bool = False) -> List[Dict]:
    invalid_links = []
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        in_code_block = False
        
        for line_num, line_content in enumerate(lines, 1):
            stripped_line = line_content.strip()
            
            # Simple state machine for fenced code blocks
            if stripped_line.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                continue

            matches = MD_LINK_PATTERN.findall(line_content)
            for match in matches:
                # match[0] is from [text](link)
                # match[2] is from [id]: link
                link = match[0] or match[2]
                if not link: continue
                link = link.strip()
                
                if is_external(link):
                    if check_external:
                        valid, err = check_external_link(link)
                        if not valid:
                            invalid_links.append({
                                "link": link,
                                "line": line_num,
                                "type": "external",
                                "error": err
                            })
                else:
                    # Handle mailto: and other protocols
                    if link.startswith(('mailto:', 'tel:', 'ssh:', 'irc:')):
                        continue
                    
                    try:
                        target_path = resolve_relative_path(file_path, link, project_root)
                        if not target_path.exists():
                            invalid_links.append({
                                "link": link,
                                "line": line_num,
                                "type": "internal",
                                "error": "File not found",
                                "resolved_path": str(target_path.relative_to(project_root)) if project_root in target_path.parents else str(target_path)
                            })
                    except OSError as e:
                        invalid_links.append({
                            "link": link,
                            "line": line_num,
                            "type": "internal",
                            "error": f"Invalid path (OSError): {str(e)}"
                        })
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
    return invalid_links

def scan_json_manifest(file_path: Path, project_root: Path, check_external: bool = False) -> List[Dict]:
    invalid_links = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        def find_links_in_obj(obj):
            if isinstance(obj, str):
                if obj.startswith(('/', '.', 'http')):
                    if is_external(obj):
                        if check_external:
                            valid, err = check_external_link(obj)
                            if not valid:
                                invalid_links.append({"link": obj, "type": "external", "error": err})
                    else:
                        if obj.startswith(('mailto:', 'tel:')): return
                        target_path = resolve_relative_path(file_path, obj, project_root)
                        if not target_path.exists():
                            invalid_links.append({
                                "link": obj,
                                "type": "internal",
                                "error": "File not found",
                                "resolved_path": str(target_path.relative_to(project_root)) if project_root in target_path.parents else str(target_path)
                            })
            elif isinstance(obj, dict):
                for v in obj.values(): find_links_in_obj(v)
            elif isinstance(obj, list):
                for item in obj: find_links_in_obj(item)
                    
        find_links_in_obj(data)
    except Exception as e:
        invalid_links.append({"error": f"JSON parse error: {str(e)}"})
    return invalid_links

def main():
    parser = argparse.ArgumentParser(description="Sanctuary Link Verifier")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--check-external", action="store_true", help="Also check external HTTP/HTTPS links")
    parser.add_argument("--output", default="invalid_links_report.json", help="Output report file")
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    report = {}
    print(f"Scanning project root: {project_root}")

    md_files = list(project_root.glob("**/*.md"))
    manifest_files = list(project_root.glob("**/*manifest.json"))
    
    # Exclude .venv and node_modules
    md_files = [
        f for f in md_files 
        if ".venv" not in str(f) 
        and "node_modules" not in str(f) 
        and ".agent/learning" not in str(f) 
        and "dataset_package" not in str(f)
        and "learning_audit_packet.md" not in str(f)
        and "red_team_audit_packet.md" not in str(f)
        and "learning_debrief.md" not in str(f)
        and "learning_package_snapshot.md" not in str(f)
        and "ARCHIVE" not in str(f)
    ]
    # User requested to ignore all manifest files for now
    manifest_files = [] 

    total_files = len(md_files) + len(manifest_files)
    print(f"Found {len(md_files)} .md files and {len(manifest_files)} manifest files (excluding .venv/node_modules).")

    processed = 0
    for file_path in md_files:
        errors = scan_md_file(file_path, project_root, args.check_external)
        if errors:
            report[str(file_path.relative_to(project_root))] = errors
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{total_files} files...")

    for file_path in manifest_files:
        errors = scan_json_manifest(file_path, project_root, args.check_external)
        if errors:
            report[str(file_path.relative_to(project_root))] = errors
        processed += 1

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nScan complete. Found issues in {len(report)} files.")
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
