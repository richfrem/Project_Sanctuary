#!/usr/bin/env python3
"""
bundle.py (CLI)
=====================================

Purpose:
    Bundles multiple source files into a single Markdown 'Context Bundle' based on a JSON manifest.

Layer: Curate / Bundler

Usage Examples:
    python tools/retrieve/bundler/bundle.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    manifest        : Path to file-manifest.json
    -o              : Output markdown file path

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - write_file_content(): Helper to write a single file's content to the markdown output.
    - bundle_files(): Bundles files specified in a JSON manifest into a single Markdown file.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import json
import os
import argparse
import datetime
import sys
from pathlib import Path
from typing import Optional

# Ensure we can import the path resolver from project root
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print(f"DEBUG: Bundler initialized. Project Root: {project_root}")

try:
    from tools.investigate.utils.path_resolver import resolve_path
except ImportError:
    # Fallback to local logic if running standalone without project context
    def resolve_path(path_str: str) -> Path:
        p = Path(path_str)
        if p.exists(): return p.resolve()
        # Try project root relative
        p_root = project_root / path_str
        if p_root.exists(): return p_root.resolve()
        # Try relative to cwd
        return Path(os.path.abspath(path_str))

def write_file_content(out, path: Path, rel_path: str, note: str = ""):
    """Helper to write a single file's content to the markdown output."""
    out.write(f"\n---\n\n")
    out.write(f"## File: {rel_path}\n")
    out.write(f"**Path:** `{rel_path}`\n")
    if note:
        out.write(f"**Note:** {note}\n")
    out.write("\n")

    try:
        ext = path.suffix.lower().replace('.', '')
        # Map common extensions to markdown languages
        lang_map = {
            'js': 'javascript', 'ts': 'typescript', 'py': 'python', 
            'md': 'markdown', 'json': 'json', 'yml': 'yaml', 'html': 'html',
            'mmd': 'mermaid', 'css': 'css', 'sql': 'sql', 'xml': 'xml',
            'txt': 'text', 'ps1': 'powershell', 'sh': 'bash',
            'pks': 'sql', 'pkb': 'sql', 'pkg': 'sql', 'in': 'text'
        }
        lang = lang_map.get(ext, '')

        # Define textual extensions that can be read as utf-8
        text_extensions = set(lang_map.keys())
        
        if ext in text_extensions or not ext:
            with open(path, 'r', encoding='utf-8', errors='replace') as source_file:
                content = source_file.read()
                
            out.write(f"```{lang}\n")
            out.write(content)
            out.write("\n```\n")
        else:
            out.write(f"> ⚠️ Binary or unknown file type ({ext}). Content skipped.\n")
    except Exception as e:
        out.write(f"> ⚠️ Error reading file: {e}\n")

def bundle_files(manifest_path: str, output_path: str) -> None:
    """
    Bundles files specified in a JSON manifest into a single Markdown file.

    Args:
        manifest_path (str): Path to the input JSON manifest.
        output_path (str): Path to write the output markdown bundle.
    
    Raises:
        FileNotFoundError: If manifest doesn't exist.
        json.JSONDecodeError: If manifest is invalid.
    """
    manifest_abs_path = os.path.abspath(manifest_path)
    base_dir = os.path.dirname(manifest_abs_path)
    
    try:
        with open(manifest_abs_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return

    # Extract metadata
    # Prefer 'title', fall back to 'name' or 'tool_name' or Default
    title = manifest.get('title') or manifest.get('name') or manifest.get('tool_name') or 'Context Bundle'
    description = manifest.get('description', '')
    files = manifest.get('files', [])

    print(f"📦 Bundling '{title}'...")

    with open(output_path, 'w', encoding='utf-8') as out:
        # Header
        out.write(f"# {title}\n")
        out.write(f"**Generated:** {datetime.datetime.now().isoformat()}\n")
        if description:
            out.write(f"\n{description}\n")
        out.write("\n---\n\n")

        # Table of Contents
        out.write("## 📑 Table of Contents\n")
        
        # We need to collect valid items first to generate TOC correctly if we expand dirs
        # But expansion happens during processing. 
        # For simplicity in this version, TOC will list the Manifest Entries, mentioning recursion if applicable.
        for i, item in enumerate(files, 1):
            path_str = item.get('path', 'Unknown')
            note = item.get('note', '')
            out.write(f"{i}. [{path_str}](#entry-{i})\n")
        out.write("\n---\n\n")

        # Content Loop
        for i, item in enumerate(files, 1):
            rel_path = item.get('path')
            note = item.get('note', '')
            
            out.write(f"<a id='entry-{i}'></a>\n")
            
            # Resolve path
            found_path = None
            
            # Try PathResolver
            try:
                candidate_str = resolve_path(rel_path)
                candidate = Path(candidate_str)
                if candidate.exists():
                    found_path = candidate
            except Exception:
                pass

            # Try Relative to Manifest
            if not found_path:
                candidate = Path(base_dir) / rel_path
                if candidate.exists():
                    found_path = candidate
            
            # Use relative path if found (or keep original string)
            display_path = str(found_path.relative_to(project_root)).replace('\\', '/') if found_path else rel_path

            if found_path and found_path.exists():
                if found_path.is_dir():
                    # RECURSIVE DIRECTORY PROCESSING
                    out.write(f"### Directory: {display_path}\n")
                    if note:
                        out.write(f"**Note:** {note}\n")
                    out.write(f"> 📂 Expanding contents of `{display_path}`...\n")
                    
                    # Walk directory
                    for root, dirs, filenames in os.walk(found_path):
                        # Filter hidden dirs (like .git, __pycache__, node_modules)
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != '__pycache__']
                        
                        for filename in filenames:
                            file_full_path = Path(root) / filename
                            # Calculate relative path from project root for display
                            try:
                                file_rel_path = str(file_full_path.relative_to(project_root)).replace('\\', '/')
                            except ValueError:
                                file_rel_path = str(file_full_path)
                                
                            write_file_content(out, file_full_path, file_rel_path, note="(Expanded from directory)")
                else:
                    # SINGLE FILE PROCESSING
                    write_file_content(out, found_path, display_path, note)
            else:
                out.write(f"## {i}. {rel_path} (MISSING)\n")
                out.write(f"> ❌ File not found: {rel_path}\n")
                # Debug info
                try:
                    debug_resolve = resolve_path(rel_path)
                    out.write(f"> Debug: ResolvePath tried: {debug_resolve}\n")
                except:
                    pass
                try:
                    out.write(f"> Debug: BaseDir tried: {Path(base_dir) / rel_path}\n")
                except:
                    pass

    print(f"✅ Bundle created at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Context Bundler')
    parser.add_argument('manifest', help='Path to file-manifest.json')
    parser.add_argument('-o', '--output', help='Output markdown file path', default='bundle.md')
    
    args = parser.parse_args()
    bundle_files(args.manifest, args.output)
