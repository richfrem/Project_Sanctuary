#!/usr/bin/env python3
"""
Mermaid Inventory Script (Task #154 - Phase 1)

Scans the project for all Mermaid diagram content:
1. Finds all ```mermaid blocks in .md files
2. Finds all .mmd files
3. Generates SHA-256 hashes for deduplication
4. Produces inventory_mermaid.json mapping unique diagrams to their locations
"""

import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from difflib import SequenceMatcher
from collections import defaultdict


# Directories to exclude from scanning
EXCLUDED_DIRS = {
    '.git', '.venv', 'node_modules', '__pycache__', 
    '.pytest_cache', '.mypy_cache', 'dist', 'build'
}

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class MermaidBlock:
    """Represents a single Mermaid diagram occurrence."""
    file_path: str
    line_start: int
    line_end: int
    content_hash: str
    diagram_type: str  # e.g., 'flowchart', 'sequenceDiagram', 'classDiagram'
    content_preview: str  # First 100 chars for identification


@dataclass
class MermaidFile:
    """Represents a dedicated .mmd file."""
    file_path: str
    content_hash: str
    diagram_type: str
    line_count: int


@dataclass 
class UniqueDigram:
    """A unique diagram identified by hash."""
    content_hash: str
    diagram_type: str
    content_preview: str
    occurrences: list  # List of file paths where this appears
    is_canonical: bool  # True if stored in .mmd file
    canonical_path: Optional[str]  # Path to .mmd if exists


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of normalized content."""
    # Normalize whitespace for consistent hashing
    normalized = '\n'.join(line.strip() for line in content.strip().split('\n'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def detect_diagram_type(content: str) -> str:
    """Detect the Mermaid diagram type from content."""
    content_lower = content.strip().lower()
    
    type_patterns = [
        ('flowchart', r'^(flowchart|graph)\s'),
        ('sequenceDiagram', r'^sequencediagram'),
        ('classDiagram', r'^classdiagram'),
        ('stateDiagram', r'^statediagram'),
        ('erDiagram', r'^erdiagram'),
        ('journey', r'^journey'),
        ('gantt', r'^gantt'),
        ('pie', r'^pie'),
        ('quadrantChart', r'^quadrantchart'),
        ('requirementDiagram', r'^requirementdiagram'),
        ('gitGraph', r'^gitgraph'),
        ('mindmap', r'^mindmap'),
        ('timeline', r'^timeline'),
        ('zenuml', r'^zenuml'),
        ('sankey', r'^sankey'),
        ('xychart', r'^xychart'),
        ('block', r'^block-beta'),
    ]
    
    for dtype, pattern in type_patterns:
        if re.match(pattern, content_lower):
            return dtype
    
    return 'unknown'


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded from scanning."""
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True
    return False


def extract_mermaid_blocks(file_path: Path) -> list[MermaidBlock]:
    """Extract all ```mermaid blocks from a markdown file."""
    blocks = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return blocks
    
    # Pattern to match ```mermaid ... ``` blocks
    pattern = r'```mermaid\s*\n(.*?)```'
    
    lines = content.split('\n')
    line_positions = {}
    pos = 0
    for i, line in enumerate(lines):
        line_positions[pos] = i + 1  # 1-indexed
        pos += len(line) + 1  # +1 for newline
    
    for match in re.finditer(pattern, content, re.DOTALL):
        block_content = match.group(1)
        start_pos = match.start()
        end_pos = match.end()
        
        # Find line numbers
        start_line = 1
        end_line = 1
        for pos, line_num in line_positions.items():
            if pos <= start_pos:
                start_line = line_num
            if pos <= end_pos:
                end_line = line_num
        
        blocks.append(MermaidBlock(
            file_path=str(file_path.relative_to(PROJECT_ROOT)),
            line_start=start_line,
            line_end=end_line,
            content_hash=compute_hash(block_content),
            diagram_type=detect_diagram_type(block_content),
            content_preview=block_content.strip()[:100].replace('\n', ' ')
        ))
    
    return blocks


def scan_mmd_files(root: Path) -> list[MermaidFile]:
    """Scan for all .mmd files."""
    mmd_files = []
    
    for mmd_path in root.rglob('*.mmd'):
        if should_exclude(mmd_path):
            continue
        
        try:
            content = mmd_path.read_text(encoding='utf-8')
            line_count = len(content.split('\n'))
            
            mmd_files.append(MermaidFile(
                file_path=str(mmd_path.relative_to(PROJECT_ROOT)),
                content_hash=compute_hash(content),
                diagram_type=detect_diagram_type(content),
                line_count=line_count
            ))
        except (UnicodeDecodeError, PermissionError):
            continue
    
    return mmd_files


def scan_markdown_files(root: Path) -> list[MermaidBlock]:
    """Scan all markdown files for mermaid blocks."""
    all_blocks = []
    
    for md_path in root.rglob('*.md'):
        if should_exclude(md_path):
            continue
        
        blocks = extract_mermaid_blocks(md_path)
        all_blocks.extend(blocks)
    
    return all_blocks


def build_inventory(blocks: list[MermaidBlock], mmd_files: list[MermaidFile]) -> dict:
    """Build the complete inventory with deduplication analysis."""
    
    # Build hash -> occurrences mapping
    hash_to_blocks: dict[str, list[MermaidBlock]] = {}
    for block in blocks:
        if block.content_hash not in hash_to_blocks:
            hash_to_blocks[block.content_hash] = []
        hash_to_blocks[block.content_hash].append(block)
    
    # Map hash -> canonical .mmd file
    hash_to_mmd: dict[str, MermaidFile] = {}
    for mmd in mmd_files:
        hash_to_mmd[mmd.content_hash] = mmd
    
    # Build unique diagrams
    all_hashes = set(hash_to_blocks.keys()) | set(hash_to_mmd.keys())
    unique_diagrams = []
    
    for h in sorted(all_hashes):
        blocks_list = hash_to_blocks.get(h, [])
        mmd = hash_to_mmd.get(h)
        
        # Determine preview and type
        if blocks_list:
            preview = blocks_list[0].content_preview
            dtype = blocks_list[0].diagram_type
        elif mmd:
            preview = f"[See {mmd.file_path}]"
            dtype = mmd.diagram_type
        else:
            preview = ""
            dtype = "unknown"
        
        unique_diagrams.append({
            "content_hash": h,
            "diagram_type": dtype,
            "content_preview": preview,
            "occurrence_count": len(blocks_list),
            "occurrences": [
                {
                    "file": b.file_path,
                    "lines": f"{b.line_start}-{b.line_end}"
                }
                for b in blocks_list
            ],
            "is_canonical": mmd is not None,
            "canonical_path": mmd.file_path if mmd else None
        })
    
    # Sort by occurrence count (most duplicated first)
    unique_diagrams.sort(key=lambda x: -x["occurrence_count"])
    
    # Calculate summary statistics
    total_inline_blocks = len(blocks)
    total_mmd_files = len(mmd_files)
    total_unique = len(unique_diagrams)
    duplicated = [d for d in unique_diagrams if d["occurrence_count"] > 1]
    orphaned_inline = [d for d in unique_diagrams if d["occurrence_count"] > 0 and not d["is_canonical"]]
    
    return {
        "summary": {
            "total_inline_mermaid_blocks": total_inline_blocks,
            "total_mmd_files": total_mmd_files,
            "unique_diagrams": total_unique,
            "duplicated_diagrams": len(duplicated),
            "orphaned_inline_diagrams": len(orphaned_inline),
            "potential_token_savings": f"~{total_inline_blocks * 50} tokens (est.)"
        },
        "duplicated_diagrams": duplicated,
        "all_unique_diagrams": unique_diagrams,
        "mmd_files": [asdict(m) for m in mmd_files],
        "inline_blocks": [asdict(b) for b in blocks]
    }


# Store full content for similarity analysis
_hash_to_content: dict[str, str] = {}


def normalize_for_similarity(content: str) -> str:
    """Normalize mermaid content for similarity comparison."""
    # Remove config blocks (--- ... ---)
    content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)
    # Normalize whitespace
    content = '\n'.join(line.strip() for line in content.strip().split('\n') if line.strip())
    # Remove comments
    content = re.sub(r'%%.*$', '', content, flags=re.MULTILINE)
    return content.lower()


def compute_similarity(content1: str, content2: str) -> float:
    """Compute similarity ratio between two diagram contents."""
    norm1 = normalize_for_similarity(content1)
    norm2 = normalize_for_similarity(content2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def cluster_similar_diagrams(blocks: list[MermaidBlock], mmd_files: list[MermaidFile], 
                              similarity_threshold: float = 0.85) -> dict:
    """
    Cluster diagrams by similarity, not just exact hash match.
    Returns clusters of similar diagrams that could be unified.
    """
    # Collect all unique contents with their full text
    hash_to_content: dict[str, str] = {}
    
    # Re-extract full content for each unique hash
    for block in blocks:
        if block.content_hash not in hash_to_content:
            # Need to re-read the file to get full content
            try:
                file_path = PROJECT_ROOT / block.file_path
                content = file_path.read_text(encoding='utf-8')
                pattern = r'```mermaid\s*\n(.*?)```'
                for match in re.finditer(pattern, content, re.DOTALL):
                    block_content = match.group(1)
                    h = compute_hash(block_content)
                    if h not in hash_to_content:
                        hash_to_content[h] = block_content
            except Exception:
                pass
    
    for mmd in mmd_files:
        if mmd.content_hash not in hash_to_content:
            try:
                content = (PROJECT_ROOT / mmd.file_path).read_text(encoding='utf-8')
                hash_to_content[mmd.content_hash] = content
            except Exception:
                pass
    
    # Store for later use
    global _hash_to_content
    _hash_to_content = hash_to_content
    
    # Build similarity clusters
    hashes = list(hash_to_content.keys())
    clustered = set()
    clusters = []
    
    for i, h1 in enumerate(hashes):
        if h1 in clustered:
            continue
        
        cluster = [h1]
        clustered.add(h1)
        
        for h2 in hashes[i+1:]:
            if h2 in clustered:
                continue
            
            sim = compute_similarity(hash_to_content[h1], hash_to_content[h2])
            if sim >= similarity_threshold:
                cluster.append(h2)
                clustered.add(h2)
        
        if len(cluster) > 1:
            clusters.append({
                "cluster_id": len(clusters) + 1,
                "member_hashes": cluster,
                "member_count": len(cluster),
                "similarity_threshold": similarity_threshold
            })
    
    return clusters


def build_canonical_map(unique_diagrams: list, similarity_clusters: list, 
                        mmd_files: list[MermaidFile]) -> dict:
    """
    Build a map of truly canonical diagrams that should be retained.
    Considers both exact duplicates and similar diagrams.
    """
    # Map hash to mmd file
    hash_to_mmd = {m.content_hash: m.file_path for m in mmd_files}
    
    # Build hash to diagram info
    hash_to_info = {d["content_hash"]: d for d in unique_diagrams}
    
    # For each cluster, pick the canonical representative
    canonical_diagrams = []
    clustered_hashes = set()
    
    for cluster in similarity_clusters:
        # Find the best representative (prefer ones with .mmd files)
        members = cluster["member_hashes"]
        
        # Priority: has .mmd > most occurrences > first
        best = None
        best_score = -1
        
        for h in members:
            info = hash_to_info.get(h, {})
            has_mmd = h in hash_to_mmd
            occurrences = info.get("occurrence_count", 0)
            score = (1000 if has_mmd else 0) + occurrences
            
            if score > best_score:
                best_score = score
                best = h
        
        if best:
            canonical_diagrams.append({
                "canonical_hash": best,
                "canonical_mmd": hash_to_mmd.get(best),
                "similar_variants": [h for h in members if h != best],
                "total_occurrences": sum(
                    hash_to_info.get(h, {}).get("occurrence_count", 0) 
                    for h in members
                ),
                "preview": hash_to_info.get(best, {}).get("content_preview", "")[:80]
            })
            clustered_hashes.update(members)
    
    # Add unclustered diagrams (truly unique)
    for d in unique_diagrams:
        h = d["content_hash"]
        if h not in clustered_hashes:
            canonical_diagrams.append({
                "canonical_hash": h,
                "canonical_mmd": hash_to_mmd.get(h),
                "similar_variants": [],
                "total_occurrences": d["occurrence_count"],
                "preview": d["content_preview"][:80]
            })
    
    # Sort by total occurrences
    canonical_diagrams.sort(key=lambda x: -x["total_occurrences"])
    
    return canonical_diagrams


def main():
    """Main entry point."""
    print("🔍 Mermaid Inventory Scanner (Task #154 - Phase 1)")
    print("=" * 60)
    
    print("\n📂 Scanning for .mmd files...")
    mmd_files = scan_mmd_files(PROJECT_ROOT)
    print(f"   Found {len(mmd_files)} .mmd files")
    
    print("\n📄 Scanning markdown files for ```mermaid blocks...")
    blocks = scan_markdown_files(PROJECT_ROOT)
    print(f"   Found {len(blocks)} inline mermaid blocks")
    
    print("\n🧮 Building inventory with deduplication analysis...")
    inventory = build_inventory(blocks, mmd_files)
    
    print("\n🔬 Running similarity clustering (85% threshold)...")
    similarity_clusters = cluster_similar_diagrams(blocks, mmd_files, similarity_threshold=0.85)
    print(f"   Found {len(similarity_clusters)} clusters of similar diagrams")
    
    print("\n📐 Building canonical diagram map...")
    canonical_map = build_canonical_map(
        inventory["all_unique_diagrams"], 
        similarity_clusters, 
        mmd_files
    )
    
    # Count truly unique after clustering
    truly_unique = len(canonical_map)
    with_mmd = sum(1 for c in canonical_map if c["canonical_mmd"])
    without_mmd = truly_unique - with_mmd
    
    # Update inventory with similarity analysis
    inventory["similarity_clusters"] = similarity_clusters
    inventory["canonical_diagrams"] = canonical_map
    inventory["summary"]["truly_unique_after_similarity"] = truly_unique
    inventory["summary"]["canonical_with_mmd"] = with_mmd
    inventory["summary"]["canonical_without_mmd"] = without_mmd
    
    # Write output
    output_path = PROJECT_ROOT / "inventory_mermaid.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\n✅ Inventory written to: {output_path}")
    
    # Print summary
    summary = inventory["summary"]
    print("\n" + "=" * 60)
    print("📊 INVENTORY SUMMARY")
    print("=" * 60)
    print(f"   Total inline ```mermaid blocks: {summary['total_inline_mermaid_blocks']}")
    print(f"   Total .mmd files:               {summary['total_mmd_files']}")
    print(f"   Unique diagrams (by hash):      {summary['unique_diagrams']}")
    print(f"   Similarity clusters found:      {len(similarity_clusters)}")
    print(f"   ─────────────────────────────────────────────────────")
    print(f"   TRULY UNIQUE (after clustering): {truly_unique}")
    print(f"      ├─ With canonical .mmd:       {with_mmd}")
    print(f"      └─ Missing .mmd (orphaned):   {without_mmd}")
    print(f"   ─────────────────────────────────────────────────────")
    print(f"   Potential token savings:        {summary['potential_token_savings']}")
    
    # Show similarity clusters
    if similarity_clusters:
        print("\n" + "-" * 60)
        print("🔗 SIMILARITY CLUSTERS (variants that could be unified):")
        print("-" * 60)
        for cluster in similarity_clusters[:5]:
            print(f"\n   Cluster #{cluster['cluster_id']}: {cluster['member_count']} similar diagrams")
            print(f"   Hashes: {', '.join(h[:8] + '...' for h in cluster['member_hashes'])}")
    
    # Show canonical diagrams that need .mmd files
    orphaned = [c for c in canonical_map if not c["canonical_mmd"] and c["total_occurrences"] > 0]
    if orphaned:
        print("\n" + "-" * 60)
        print("⚠️  TOP CANONICAL DIAGRAMS NEEDING .mmd FILES:")
        print("-" * 60)
        for c in orphaned[:10]:
            print(f"\n   Hash: {c['canonical_hash']}")
            print(f"   Occurrences: {c['total_occurrences']}")
            print(f"   Variants: {len(c['similar_variants'])}")
            print(f"   Preview: {c['preview'][:60]}...")
    
    # Show diagrams that already have .mmd files
    with_mmd_list = [c for c in canonical_map if c["canonical_mmd"]]
    if with_mmd_list:
        print("\n" + "-" * 60)
        print("✅ CANONICAL DIAGRAMS WITH .mmd FILES:")
        print("-" * 60)
        for c in with_mmd_list[:15]:
            print(f"   {c['canonical_mmd']}")
            print(f"      └─ Occurrences: {c['total_occurrences']}, Variants absorbed: {len(c['similar_variants'])}")
    
    print("\n" + "=" * 60)
    print("📋 PHASE 2 RECOMMENDATIONS:")
    print("=" * 60)
    print(f"   1. Review the {without_mmd} orphaned canonical diagrams")
    print(f"   2. Create .mmd files in docs/mcp_servers/architecture/diagrams/")
    print(f"   3. Render .mmd → .png using mermaid-cli")
    print(f"   4. Replace all {summary['total_inline_mermaid_blocks']} inline blocks with image refs")
    print("=" * 60)


if __name__ == "__main__":
    main()

