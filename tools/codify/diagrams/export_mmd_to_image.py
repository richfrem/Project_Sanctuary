#!/usr/bin/env python3
"""
export_mmd_to_image.py (CLI)
=====================================

Purpose:
    Core Utility for converting Mermaid logic definitions (.mmd) into
    visual assets (.png/.svg) for documentation and reports.
    Enforces ADR 085 (Content Hygiene) by ensuring diagrams are rendered
    to static artifacts rather than relying on dynamic GitHub rendering.

Layer: Codify / Diagrams

Usage Examples:
    python3 tools/codify/diagrams/export_mmd_to_image.py --input docs/
    python3 tools/codify/diagrams/export_mmd_to_image.py --check
    python3 tools/codify/diagrams/export_mmd_to_image.py my_diagram.mmd

CLI Arguments:
    --input, -i     : Input MMD file or directory (optional)
    --output, -o    : Output file path or directory (optional)
    --check         : Dry-run check for outdated images
    --svg           : Render as SVG instead of PNG (default: PNG)
    targets         : Positional arguments for partial filename matching (Legacy mode)

Input Files:
    - .mmd files in target directory or specific file

Output:
    - .png/.svg image files

Key Functions:
    - render_diagram_explicit(): Calls headless mermaid-cli.
    - main(): Argument parsing and batch processing logic.

Script Dependencies:
    - npm package: @mermaid-js/mermaid-cli

Consumed by:
    - scripts/render_diagrams.py (wrapper - deprecated)
    - CI/CD Pipelines
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "architecture_diagrams"
OUTPUT_FORMAT = "png"  # or "svg"


def get_npx_cmd():
    return "npx.cmd" if sys.platform == "win32" else "npx"


def check_mmdc():
    """Check if mermaid-cli is available."""
    try:
        result = subprocess.run(
            [get_npx_cmd(), "-y", "@mermaid-js/mermaid-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"✅ mermaid-cli available: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ mermaid-cli check failed: {e}")
    return False


def render_diagram(mmd_path: Path, output_format: str = "png") -> bool:
    """Render a single .mmd file to image."""
    output_path = mmd_path.with_suffix(f".{output_format}")
    
    try:
        result = subprocess.run(
            [
                "npx", "-y", "@mermaid-js/mermaid-cli",
                "-i", str(mmd_path),
                "-o", str(output_path),
                "-b", "transparent",
                "-t", "default"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   Timeout rendering {mmd_path.name}")
        return False
    except Exception as e:
        print(f"   Exception: {e}")
        return False


def check_outdated(mmd_path: Path, output_format: str = "png") -> bool:
    """Check if image is older than source or missing."""
    output_path = mmd_path.with_suffix(f".{output_format}")
    
    if not output_path.exists():
        return True
    
    return mmd_path.stat().st_mtime > output_path.stat().st_mtime


import argparse

def main():
    parser = argparse.ArgumentParser(description="Mermaid Diagram Renderer")
    parser.add_argument("--input", "-i", type=str, help="Input MMD file or directory", required=False)
    parser.add_argument("--output", "-o", type=str, help="Output file path or directory", required=False)
    parser.add_argument("--svg", action="store_true", help="Render as SVG instead of PNG")
    parser.add_argument("--check", action="store_true", help="Check for outdated images only")
    # For backward compatibility / positional args
    parser.add_argument("targets", nargs="*", help="Partial filenames to match in default dir")
    
    args = parser.parse_args()
    
    output_format = "svg" if args.svg else "png"
    
    print("🎨 Mermaid Diagram Renderer")
    print("=" * 60)
    
    mmd_files_map = {} # Map input_path -> output_path
    
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Input path not found: {input_path}")
            sys.exit(1)
            
        if input_path.is_file():
            # Single file
            if args.output:
                out_path = Path(args.output)
                if out_path.is_dir():
                    out_path = out_path / input_path.with_suffix(f".{output_format}").name
            else:
                out_path = input_path.with_suffix(f".{output_format}")
            mmd_files_map[input_path] = out_path
            
        elif input_path.is_dir():
            # Directory
            search_files = sorted(input_path.rglob("*.mmd"))
            out_root = Path(args.output) if args.output else input_path
            
            for m in search_files:
                # Calculate relative path if output is different dir
                if args.output:
                    rel = m.relative_to(input_path)
                    dest = out_root / rel.with_suffix(f".{output_format}")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                else:
                    dest = m.with_suffix(f".{output_format}")
                mmd_files_map[m] = dest
    else:
        # Legacy/Default behavior (docs/architecture_diagrams)
        # Use targets to filter
        print(f"   Source: {DIAGRAMS_DIR}")
        all_files = sorted(DIAGRAMS_DIR.rglob("*.mmd"))
        
        selected = []
        if args.targets:
             for t in args.targets:
                 matches = [f for f in all_files if t in str(f)]
                 selected.extend(matches)
        else:
             selected = all_files
        
        selected = sorted(list(set(selected)))
        for m in selected:
            mmd_files_map[m] = m.with_suffix(f".{output_format}")

    if not mmd_files_map:
        print("❌ No MMD files found to process.")
        sys.exit(0)

    print(f"   Format: {output_format.upper()}")
    print(f"   Mode: {'Check only' if args.check else 'Render'}")
    print(f"\n📂 Processing {len(mmd_files_map)} diagrams...")

    if args.check:
        outdated = []
        for inp, outp in mmd_files_map.items():
            if not outp.exists() or inp.stat().st_mtime > outp.stat().st_mtime:
                outdated.append(inp)
        print(f"\n⚠️  {len(outdated)} diagrams need rendering:")
        for f in outdated:
            print(f"   - {f}")
        return

    # Check mermaid-cli
    print("\n🔧 Checking mermaid-cli...")
    if not check_mmdc():
        print("❌ Please install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
        sys.exit(1)
        
    print(f"\n🖼️  Rendering...")
    success = 0
    failed = 0
    
    for inp, outp in mmd_files_map.items():
        print(f"   {inp.name} -> {outp.name}...", end=" ", flush=True)
        
        # We need to adapt render_diagram to accept explicit output path
        # Assuming render_diagram modifications below or inline adjustment
        # Current render_diagram calculates output path from input.
        # We must refactor render_diagram or pass output arg.
        
        # Calling modified render_diagram
        if render_diagram_explicit(inp, outp):
             print("✅")
             success += 1
        else:
             print("❌")
             failed += 1

    print("\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print(f"   Rendered: {success}")
    print(f"   Failed: {failed}")
    print("=" * 60)

def render_diagram_explicit(mmd_path: Path, output_path: Path) -> bool:
    """Render .mmd to specific output path."""
    try:
        result = subprocess.run(
            [
                get_npx_cmd(), "-y", "@mermaid-js/mermaid-cli",
                "-i", str(mmd_path),
                "-o", str(output_path),
                "-b", "transparent",
                "-t", "default"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            print(f"   Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   Timeout rendering {mmd_path.name}")
        return False
    except Exception as e:
        print(f"   Exception: {e}")
        return False

# Re-point main execution
if __name__ == "__main__":
    main()