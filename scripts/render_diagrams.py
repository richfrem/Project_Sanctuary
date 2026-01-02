#!/usr/bin/env python3
"""
Mermaid Diagram Renderer (Task #154 - Phase 3)

Renders all .mmd files in docs/architecture_diagrams/ to PNG images.
Run this script whenever diagrams are updated to regenerate images.

Usage:
    python3 scripts/render_diagrams.py                 # Render all
    python3 scripts/render_diagrams.py my_diagram.mmd  # Render specific file(s)
    python3 scripts/render_diagrams.py --svg           # Render as SVG instead
    python3 scripts/render_diagrams.py --check         # Check for outdated images
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "architecture_diagrams"
OUTPUT_FORMAT = "png"  # or "svg"


def check_mmdc():
    """Check if mermaid-cli is available."""
    try:
        result = subprocess.run(
            ["npx", "-y", "@mermaid-js/mermaid-cli", "--version"],
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


def main():
    output_format = OUTPUT_FORMAT
    check_only = False
    target_files: List[str] = []

    # Parse args
    for arg in sys.argv[1:]:
        if arg == "--svg":
            output_format = "svg"
        elif arg == "--check":
            check_only = True
        elif not arg.startswith("--"):
            target_files.append(arg)
    
    print("🎨 Mermaid Diagram Renderer")
    print("=" * 60)
    print(f"   Source: {DIAGRAMS_DIR}")
    print(f"   Format: {output_format.upper()}")
    print(f"   Mode: {'Check only' if check_only else 'Render'}")
    
    # Find all .mmd files
    all_mmd_files = sorted(DIAGRAMS_DIR.rglob("*.mmd"))
    
    # Filter if targets provided
    if target_files:
        mmd_files = []
        for target in target_files:
            # Check for exact matches or partial matches
            matches = [f for f in all_mmd_files if target in str(f)]
            mmd_files.extend(matches)
        # Remove duplicates
        mmd_files = sorted(list(set(mmd_files)))
        
        if not mmd_files:
            print(f"\n❌ No diagrams matched targets: {target_files}")
            return
    else:
        mmd_files = all_mmd_files

    print(f"\n📂 Found {len(mmd_files)} diagram files to process")
    
    if check_only:
        outdated = [f for f in mmd_files if check_outdated(f, output_format)]
        print(f"\n⚠️  {len(outdated)} diagrams need rendering:")
        for f in outdated:
            print(f"   - {f.relative_to(DIAGRAMS_DIR)}")
        return
    
    # Check mermaid-cli
    print("\n🔧 Checking mermaid-cli...")
    if not check_mmdc():
        print("❌ Please install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
        sys.exit(1)
    
    # Render all
    print(f"\n🖼️  Rendering {len(mmd_files)} diagrams...")
    success = 0
    failed = 0
    
    for mmd_path in mmd_files:
        try:
            rel_path = mmd_path.relative_to(DIAGRAMS_DIR)
        except ValueError:
             rel_path = mmd_path
             
        print(f"   {rel_path}...", end=" ", flush=True)
        
        if render_diagram(mmd_path, output_format):
            print("✅")
            success += 1
        else:
            print("❌")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print(f"   Rendered: {success}")
    print(f"   Failed: {failed}")
    print(f"   Output: {DIAGRAMS_DIR}/**/*.{output_format}")
    print("=" * 60)


if __name__ == "__main__":
    main()
