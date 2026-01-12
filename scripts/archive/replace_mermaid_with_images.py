#!/usr/bin/env python3
"""
Replace Inline Mermaid Blocks with Image References (Task #154 - Phase 3)

Scans source files for ```mermaid blocks and replaces them with
image references pointing to the canonical diagrams.
"""

import hashlib
import re
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "architecture_diagrams"

# Directories to skip (generated files, or the diagrams dir itself)
SKIP_DIRS = {'.git', '.venv', 'node_modules', '__pycache__', '.agent'}
SKIP_PATHS = {'docs/architecture_diagrams'}


def compute_hash(content: str) -> str:
    """Compute hash of normalized content."""
    normalized = '\n'.join(line.strip() for line in content.strip().split('\n'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def normalize_for_comparison(content: str) -> str:
    """Normalize content for similarity comparison."""
    content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)
    content = re.sub(r'%%.*$', '', content, flags=re.MULTILINE)
    content = '\n'.join(line.strip() for line in content.strip().split('\n') if line.strip())
    return content.lower()


def compute_similarity(content1: str, content2: str) -> float:
    """Compute similarity ratio."""
    norm1 = normalize_for_comparison(content1)
    norm2 = normalize_for_comparison(content2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def load_canonical_diagrams() -> dict:
    """Load all canonical diagrams with their hashes."""
    diagrams = {}
    for mmd_path in DIAGRAMS_DIR.rglob("*.mmd"):
        content = mmd_path.read_text()
        h = compute_hash(content)
        rel_path = mmd_path.relative_to(PROJECT_ROOT)
        png_path = mmd_path.with_suffix('.png').relative_to(PROJECT_ROOT)
        diagrams[h] = {
            'mmd_path': str(rel_path),
            'png_path': str(png_path),
            'content': content,
            'name': mmd_path.stem
        }
    return diagrams


def find_best_match(block_content: str, diagrams: dict, threshold: float = 0.7) -> dict:
    """Find the best matching canonical diagram for an inline block."""
    block_hash = compute_hash(block_content)
    
    # Exact match
    if block_hash in diagrams:
        return diagrams[block_hash]
    
    # Similarity match
    best_match = None
    best_score = 0
    
    for h, info in diagrams.items():
        score = compute_similarity(block_content, info['content'])
        if score > best_score:
            best_score = score
            best_match = info
    
    if best_score >= threshold:
        return best_match
    
    return None


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    for skip in SKIP_PATHS:
        if skip in path_str:
            return True
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def replace_mermaid_blocks(file_path: Path, diagrams: dict) -> tuple:
    """Replace mermaid blocks in a file. Returns (replacements_made, content_changed)."""
    content = file_path.read_text()
    original_content = content
    
    # Find all ```mermaid ... ``` blocks
    pattern = r'```mermaid\s*\n(.*?)```'
    
    replacements = []
    
    def replace_block(match):
        block_content = match.group(1)
        best = find_best_match(block_content, diagrams)
        
        if best:
            # Create image reference with link to source
            img_ref = f"![{best['name']}]({best['png_path']})\n\n*[Source: {best['name']}.mmd]({best['mmd_path']})*"
            replacements.append(best['name'])
            return img_ref
        else:
            # No match found, keep original
            return match.group(0)
    
    new_content = re.sub(pattern, replace_block, content, flags=re.DOTALL)
    
    if new_content != original_content:
        file_path.write_text(new_content)
        return len(replacements), True
    
    return 0, False


def main():
    print("🔄 Replacing Inline Mermaid Blocks with Image References")
    print("=" * 60)
    
    # Load canonical diagrams
    print("\n📂 Loading canonical diagrams...")
    diagrams = load_canonical_diagrams()
    print(f"   Loaded {len(diagrams)} canonical diagrams")
    
    # Find all markdown files
    print("\n📄 Scanning source files...")
    md_files = []
    for md_path in PROJECT_ROOT.rglob("*.md"):
        if not should_skip(md_path):
            md_files.append(md_path)
    
    print(f"   Found {len(md_files)} markdown files to process")
    
    # Process each file
    print("\n🔄 Processing files...")
    total_replacements = 0
    files_modified = 0
    
    for md_path in sorted(md_files):
        rel_path = md_path.relative_to(PROJECT_ROOT)
        
        # Quick check if file has mermaid blocks
        content = md_path.read_text()
        if '```mermaid' not in content:
            continue
        
        count, modified = replace_mermaid_blocks(md_path, diagrams)
        
        if modified:
            print(f"   ✅ {rel_path}: {count} replacements")
            total_replacements += count
            files_modified += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Files modified: {files_modified}")
    print(f"   Total replacements: {total_replacements}")
    print("=" * 60)


if __name__ == "__main__":
    main()
