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
    """Scan manifest JSON files for broken file references.
    
    Handles two formats:
    1. Array of file paths: ["file1.md", "file2.md"]
    2. Object with nested paths that start with / or . or http
    """
    invalid_links = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def check_file_path(path_str: str) -> None:
            """Check if a path string points to an existing file."""
            if not path_str or not isinstance(path_str, str):
                return
            # Skip external URLs, mail, empty strings
            if is_external(path_str) or path_str.startswith(('mailto:', 'tel:')):
                return
            
            # Skip strings that don't look like file paths:
            # - Pure file extensions (start with . and no / or further .)
            if path_str.startswith('.') and '/' not in path_str and path_str.count('.') == 1 and len(path_str) < 15:
                return
            # - Regex patterns (contain regex special characters like ^, $, *, |, etc.)
            if any(c in path_str for c in ['^', '$', '*', '|', '\\', '(', ')']):
                return
            # - Descriptions (contain spaces but not paths)
            if ' ' in path_str and not path_str.endswith(('.md', '.py', '.json', '.txt', '.yaml', '.yml')):
                return
            # - Version strings like "1.2" or "1.0"
            if re.match(r'^\d+\.\d+$', path_str):
                return
            # - Short strings that are just extensions or metadata keys
            if len(path_str) < 3:
                return
            # - Skip if it doesn't contain a / or look like a file path with extension
            if '/' not in path_str and '.' not in path_str:
                return
            # - Skip template strings (contain {{ or }})
            if '{{' in path_str or '}}' in path_str:
                return
                
            # For manifest files, paths are typically relative to project root
            if path_str.startswith('/'):
                target_path = project_root / path_str.lstrip('/')
            else:
                target_path = project_root / path_str
            
            # Handle directory patterns (ending with /)
            if path_str.endswith('/'):
                if not target_path.exists() or not target_path.is_dir():
                    invalid_links.append({
                        "path": path_str,
                        "type": "manifest_directory",
                        "error": "Directory not found",
                        "resolved_path": str(target_path)
                    })
            else:
                if not target_path.exists():
                    invalid_links.append({
                        "path": path_str,
                        "type": "manifest_file",
                        "error": "File not found",
                        "resolved_path": str(target_path)
                    })
            
        def find_links_in_obj(obj, skip_keys=None):
            """Recursively find and check file paths in JSON structure.
            
            Args:
                skip_keys: Set of dict keys to skip (e.g., 'output' for generated files)
            """
            if skip_keys is None:
                skip_keys = {'output', 'command', 'description'}  # Skip non-path metadata
            
            if isinstance(obj, str):
                check_file_path(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if k not in skip_keys:
                        find_links_in_obj(v, skip_keys)
            elif isinstance(obj, list):
                for item in obj: find_links_in_obj(item, skip_keys)
                    
        find_links_in_obj(data)
    except Exception as e:
        invalid_links.append({"error": f"JSON parse error: {str(e)}"})
    return invalid_links

def main():
    parser = argparse.ArgumentParser(description="Sanctuary Link Verifier")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--check-external", action="store_true", help="Also check external HTTP/HTTPS links")
    # Default output to script's directory
    script_dir = Path(__file__).parent
    parser.add_argument("--output", default=str(script_dir / "invalid_links_report.json"), help="Output report file")
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
    
    # Load manifests from the manifest registry
    manifest_registry_path = project_root / ".agent/learning/manifest_registry.json"
    manifest_files = []
    if manifest_registry_path.exists():
        try:
            with open(manifest_registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            for manifest_path in registry.get("manifests", {}).keys():
                # Skip manifests that contain patterns/exclusions, not actual file references
                if any(skip in manifest_path for skip in ["exclusion_manifest", "gguf_model_manifest"]):
                    continue
                full_path = project_root / manifest_path
                if full_path.exists():
                    manifest_files.append(full_path)
                else:
                    print(f"Warning: Registry manifest not found: {manifest_path}")
        except Exception as e:
            print(f"Error loading manifest registry: {e}")
    else:
        print("Warning: Manifest registry not found at .agent/learning/manifest_registry.json")

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
