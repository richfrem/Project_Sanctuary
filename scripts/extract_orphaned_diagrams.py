#!/usr/bin/env python3
"""
Extract Orphaned Mermaid Diagrams (Task #154 - Phase 2)

Reads the inventory_mermaid.json and extracts all orphaned diagrams
(those without .mmd files) into the new centralized directory structure.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "architecture_diagrams"

# Mapping of diagram types/content patterns to subdirectories
def categorize_diagram(content: str, preview: str) -> str:
    """Determine which subdirectory a diagram belongs in."""
    content_lower = content.lower()
    preview_lower = preview.lower()
    
    # Check for workflow/protocol patterns
    if any(kw in content_lower for kw in ['protocol', 'learning', 'scout', 'seal', 'audit', 'pipeline', 'flow']):
        return 'workflows'
    
    # Check for sequence diagrams
    if content_lower.startswith('sequencediagram'):
        return 'sequences'
    
    # Check for class diagrams
    if content_lower.startswith('classdiagram'):
        return 'class'
    
    # Check for RAG-related
    if any(kw in content_lower for kw in ['rag', 'embedding', 'vector', 'chunking']):
        return 'rag'
    
    # Check for transport/SSE/stdio
    if any(kw in content_lower for kw in ['transport', 'sse', 'stdio', 'bridge']):
        return 'transport'
    
    # Default to system for architecture diagrams
    return 'system'


def sanitize_filename(preview: str, hash_val: str) -> str:
    """Create a reasonable filename from preview text."""
    # Extract meaningful words from preview
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', preview[:80])
    
    if len(words) >= 3:
        # Use first few meaningful words
        name = '_'.join(words[:4]).lower()
    else:
        # Fall back to hash-based name
        name = f"diagram_{hash_val[:8]}"
    
    # Ensure it's a valid filename
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    
    return name[:50]  # Limit length


def extract_content_by_hash(hash_val: str, inline_blocks: list) -> str:
    """Extract the full content for a given hash from inline blocks."""
    for block in inline_blocks:
        if block['content_hash'] == hash_val:
            # Read the file and extract the block
            try:
                file_path = PROJECT_ROOT / block['file_path']
                content = file_path.read_text(encoding='utf-8')
                pattern = r'```mermaid\s*\n(.*?)```'
                for match in re.finditer(pattern, content, re.DOTALL):
                    block_content = match.group(1)
                    # Compute hash to verify
                    from hashlib import sha256
                    normalized = '\n'.join(line.strip() for line in block_content.strip().split('\n'))
                    h = sha256(normalized.encode('utf-8')).hexdigest()[:16]
                    if h == hash_val:
                        return block_content.strip()
            except Exception as e:
                print(f"   Warning: Could not read {block['file_path']}: {e}")
    
    return None


def main():
    print("📦 Extracting Orphaned Mermaid Diagrams")
    print("=" * 60)
    
    # Load inventory
    inventory_path = PROJECT_ROOT / "inventory_mermaid.json"
    if not inventory_path.exists():
        print("❌ inventory_mermaid.json not found. Run mermaid_inventory.py first.")
        return
    
    with open(inventory_path, 'r') as f:
        inventory = json.load(f)
    
    canonical_diagrams = inventory.get('canonical_diagrams', [])
    inline_blocks = inventory.get('inline_blocks', [])
    
    # Filter to orphaned (no canonical_mmd)
    orphaned = [d for d in canonical_diagrams if not d.get('canonical_mmd') and d.get('total_occurrences', 0) > 0]
    
    print(f"\n📊 Found {len(orphaned)} orphaned canonical diagrams")
    print(f"   Target directory: {DIAGRAMS_DIR}")
    
    # Track created files
    created_files = []
    skipped = []
    category_counts = defaultdict(int)
    
    for diag in orphaned:
        hash_val = diag['canonical_hash']
        preview = diag.get('preview', '')
        occurrences = diag.get('total_occurrences', 0)
        
        print(f"\n   Processing {hash_val[:12]}... ({occurrences} occurrences)")
        
        # Extract full content
        content = extract_content_by_hash(hash_val, inline_blocks)
        if not content:
            print(f"      ⚠️  Could not extract content, skipping")
            skipped.append(hash_val)
            continue
        
        # Categorize
        category = categorize_diagram(content, preview)
        category_counts[category] += 1
        
        # Generate filename
        filename = sanitize_filename(preview, hash_val)
        target_dir = DIAGRAMS_DIR / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing file with same name
        target_path = target_dir / f"{filename}.mmd"
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{filename}_{counter}.mmd"
            counter += 1
        
        # Write the file
        target_path.write_text(content, encoding='utf-8')
        created_files.append({
            'path': str(target_path.relative_to(PROJECT_ROOT)),
            'hash': hash_val,
            'category': category,
            'occurrences': occurrences
        })
        print(f"      ✅ Created: {target_path.relative_to(PROJECT_ROOT)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"   Total orphaned diagrams:  {len(orphaned)}")
    print(f"   Successfully created:     {len(created_files)}")
    print(f"   Skipped (no content):     {len(skipped)}")
    print("\n   By category:")
    for cat, count in sorted(category_counts.items()):
        print(f"      {cat}: {count}")
    
    # Write manifest
    manifest_path = PROJECT_ROOT / "docs" / "architecture_diagrams" / "MANIFEST.json"
    manifest = {
        "description": "Centralized Mermaid diagram repository (Task #154)",
        "created": "2025-12-29",
        "total_diagrams": len(created_files),
        "diagrams": created_files
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n   ✅ Manifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("=" * 60)
    print("   1. Review created .mmd files for naming/categorization")
    print("   2. Run similarity check to find any remaining duplicates")
    print("   3. Render .mmd → .png with mermaid-cli")
    print("   4. Update all .md files to reference images")
    print("=" * 60)


if __name__ == "__main__":
    main()
