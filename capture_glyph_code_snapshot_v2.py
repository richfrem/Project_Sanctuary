#!/usr/bin/env python3
"""
capture_glyph_code_snapshot_v2.py (v2.0 - Advanced Optical Compression with Provenance)

Enhanced version with cryptographic provenance binding and multi-resolution glyph generation.
This version addresses the security concerns raised in the DeepSeek-OCR analysis by ensuring
every glyph is cryptographically bound to its source content.

Key Enhancements v2.0:
- Cryptographic provenance binding (SHA-256 hashes)
- Multi-resolution glyph generation (thumbnail + full-res)
- Metadata embedding in PNG chunks
- Batch processing for large codebases with size limits
- Non-blocking background execution
- Integration with Optical Anvil architecture

DEPENDENCIES:
- PIL (Pillow) for advanced image processing
- hashlib for cryptographic hashing
- json for metadata embedding

USAGE EXAMPLES:

# Limited run (recommended for testing)
python3 capture_glyph_code_snapshot_v2.py --max-files 50 --max-size-mb 5

# Non-blocking background execution
python3 capture_glyph_code_snapshot_v2.py --non-blocking --max-size-mb 20

# Operation-specific with limits
python3 capture_glyph_code_snapshot_v2.py --operation WORK_IN_PROGRESS/some_dir --max-files 20

# Full provenance binding (default)
python3 capture_glyph_code_snapshot_v2.py --max-size-mb 10

# Skip provenance for faster processing
python3 capture_glyph_code_snapshot_v2.py --no-provenance --max-files 100
"""

import os
import sys
import argparse
import textwrap
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import math
from datetime import datetime

# Configuration
DEFAULT_FONT_SIZE = 10  # Smaller for better compression
DEFAULT_LINE_SPACING = 1.1  # Tighter spacing
DEFAULT_MARGIN = 15
MAX_IMAGE_WIDTH = 4096  # Allow larger images for complex content
THUMBNAIL_SIZE = (512, 512)
DEFAULT_OUTPUT_DIR = "dataset_code_glyphs"

# Enhanced color scheme
COLORS = {
    'background': '#FFFFFF',
    'text': '#1A1A1A',        # Darker for better OCR
    'header': '#2E3440',
    'separator': '#4C566A',
    'code_bg': '#F8F9FA',
    'comment': '#6C757D',
    'provenance': '#8FBCBB',  # Light blue for provenance info
    'metadata': '#5E81AC',    # Blue for metadata
}

class ProvenanceGlyphForge:
    """Advanced glyph forge with cryptographic provenance and multi-resolution output"""

    def __init__(self, font_size=DEFAULT_FONT_SIZE, line_spacing=DEFAULT_LINE_SPACING):
        self.font_size = font_size
        self.line_spacing = line_spacing

        # Try multiple font options for better rendering
        font_options = [
            "/System/Library/Fonts/Menlo.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
            "C:\\Windows\\Fonts\\consola.ttf",  # Windows
        ]

        self.font = None
        self.header_font = None

        for font_path in font_options:
            try:
                self.font = ImageFont.truetype(font_path, font_size)
                self.header_font = ImageFont.truetype(font_path, int(font_size * 1.4))
                break
            except:
                continue

        if not self.font:
            try:
                self.font = ImageFont.load_default()
                self.header_font = ImageFont.load_default()
            except:
                print("[ERROR] Could not load any font. Install PIL with font support.")
                sys.exit(1)

    def calculate_optimal_dimensions(self, text, max_width=MAX_IMAGE_WIDTH):
        """Calculate optimal image dimensions for text with word wrapping"""
        lines = []
        words = text.split()

        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = self.font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]

            if line_width > max_width - 2 * DEFAULT_MARGIN:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Word itself is too long, force break
                    lines.append(word)
                    current_line = ""
            else:
                current_line = test_line

        if current_line:
            lines.append(current_line)

        # Calculate dimensions
        max_line_width = 0
        total_height = 0
        line_height = self.font.getbbox("Ag")[3] - self.font.getbbox("Ag")[1]

        for line in lines:
            bbox = self.font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            max_line_width = max(max_line_width, line_width)
            total_height += int(line_height * self.line_spacing)

        return max_line_width + 2 * DEFAULT_MARGIN, total_height + 2 * DEFAULT_MARGIN

    def embed_metadata(self, image, metadata):
        """Embed metadata in PNG image chunks"""
        metadata_str = json.dumps(metadata, indent=2)

        # Create metadata chunk
        meta_chunk = PngImagePlugin.PngInfo()
        meta_chunk.add_text("GlyphMetadata", metadata_str)
        meta_chunk.add_text("CreationTime", datetime.now().isoformat())
        meta_chunk.add_text("ForgeVersion", "2.0")

        return meta_chunk

    def create_provenance_glyph(self, content, filename, title="Provenance-Bound Code Glyph",
                               include_provenance=True):
        """Create a provenance-bound Cognitive Glyph with metadata embedding"""

        # Generate cryptographic hash of content
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        # Prepare content with provenance information
        if include_provenance:
            provenance_header = f"""PROVENANCE BINDING
Content SHA-256: {content_hash}
Creation Time: {datetime.now().isoformat()}
Forge Version: 2.0
Compression Method: Optical (DeepSeek-OCR inspired)

--- CONTENT BOUNDARY ---
"""
            full_content = provenance_header + content
        else:
            full_content = content

        # Format content for glyph
        formatted_content = self.format_content_for_glyph(full_content, title)

        # Calculate optimal dimensions
        img_width, img_height = self.calculate_optimal_dimensions(formatted_content)

        # Create image
        img = Image.new('RGB', (img_width, img_height), COLORS['background'])
        draw = ImageDraw.Draw(img)

        # Draw content with syntax highlighting
        y_position = DEFAULT_MARGIN
        line_height = self.font.getbbox("Ag")[3] - self.font.getbbox("Ag")[1]

        for line in formatted_content.split('\n'):
            if line.startswith('PROVENANCE BINDING'):
                # Provenance header
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['provenance'],
                         font=self.header_font)
            elif line.startswith('Content SHA-256:') or line.startswith('Creation Time:') or line.startswith('Forge Version:'):
                # Metadata lines
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['metadata'], font=self.font)
            elif line.startswith('=== ') and line.endswith(' ==='):
                # Header line
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['header'], font=self.header_font)
            elif line.startswith('--- ') and ' ---' in line:
                # Separator line
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['separator'], font=self.font)
            else:
                # Regular text
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['text'], font=self.font)

            y_position += int(line_height * self.line_spacing)

        # Create metadata for embedding
        metadata = {
            "title": title,
            "content_hash": content_hash,
            "creation_timestamp": datetime.now().isoformat(),
            "forge_version": "2.0",
            "compression_method": "optical_glyph",
            "dimensions": f"{img_width}x{img_height}",
            "font_size": self.font_size,
            "estimated_tokens_original": len(content.split()) * 1.3,
            "estimated_vision_tokens": (img_width * img_height) / 850,  # Rough estimation
            "provenance_bound": include_provenance
        }

        # Embed metadata
        png_info = self.embed_metadata(img, metadata)

        # Save main image
        img.save(filename, 'PNG', png_info=png_info, optimize=True)

        # Create thumbnail version
        img.thumbnail(THUMBNAIL_SIZE)
        thumbnail_filename = str(filename).replace('.png', '_thumb.png')
        img.save(thumbnail_filename, 'PNG', png_info=png_info, optimize=True)

        compression_ratio = metadata["estimated_tokens_original"] / metadata["estimated_vision_tokens"]

        print(f"[PROVENANCE GLYPH FORGED] {filename}")
        print(f"  Dimensions: {img_width}x{img_height}px")
        print(f"  Content Hash: {content_hash[:16]}...")
        print(".1f")
        print(f"  Thumbnail: {thumbnail_filename}")

        return img_width, img_height, content_hash

    def format_content_for_glyph(self, content, title):
        """Advanced formatting optimized for OCR and compression"""
        lines = content.split('\n')
        formatted_lines = []

        # Add title header
        formatted_lines.append(f"=== {title.upper()} ===")
        formatted_lines.append("")

        for line in lines:
            line = line.rstrip()
            if not line:
                formatted_lines.append("")
            elif line.startswith('#'):
                # Headers - keep formatting
                formatted_lines.append(line)
            elif len(line) > 120:
                # Wrap very long lines
                wrapped = textwrap.wrap(line, width=100)
                formatted_lines.extend(wrapped)
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

def collect_code_snapshot_v2(project_root, operation_path=None, include_provenance=True, max_files=None, max_size_mb=50):
    """Enhanced code snapshot collection with provenance tracking and size limits"""

    exclude_dirs = {
        'node_modules', '.next', '.git', '.cache', '.turbo', '.vscode', 'dist', 'build',
        'coverage', 'out', 'tmp', 'temp', 'logs', '.idea', '.parcel-cache', '.storybook',
        '.husky', '.pnpm', '.yarn', '.svelte-kit', '.vercel', '.firebase', '.expo', '.expo-shared',
        '__pycache__', '.ipynb_checkpoints', '.tox', '.eggs', 'eggs', '.venv', 'venv', 'env',
        '.svn', '.hg', '.bzr',
        'models', 'weights', 'checkpoints', 'ckpt', 'safensors',
        'BRIEFINGS', '07_COUNCIL_AGENTS/directives',
        'dataset_package', 'chroma_db', 'dataset_code_glyphs',
        'ARCHIVES', 'ARCHIVE', 'archive', 'archives',
        'ResearchPapers', 'RESEARCH_PAPERS',
        'WORK_IN_PROGRESS'
    }

    exclude_files = {
        'capture_code_snapshot.js', 'capture_glyph_code_snapshot.py', 'capture_glyph_code_snapshot_v2.py',
        '.DS_Store', '.gitignore', 'PROMPT_PROJECT_ANALYSIS.md'
    }

    collected_files = []
    total_size = 0
    max_size_bytes = max_size_mb * 1024 * 1024
    processed_count = 0

    def traverse_and_collect(current_path):
        nonlocal total_size, processed_count

        path_obj = Path(current_path)
        if path_obj.name in exclude_dirs:
            return

        if path_obj.is_file():
            if path_obj.name in exclude_files:
                return

            if path_obj.suffix.lower() not in ['.md', '.txt', '.py', '.js', '.json']:
                return

            # Check size limits
            if max_files and len(collected_files) >= max_files:
                return
            if total_size >= max_size_bytes:
                return

            try:
                rel_path = path_obj.relative_to(project_root).as_posix()
                content = path_obj.read_text(encoding='utf-8')
                file_size = len(content)

                # Skip if this file would exceed size limit
                if total_size + file_size > max_size_bytes:
                    return

                collected_files.append({
                    'path': rel_path,
                    'content': content,
                    'size': file_size,
                    'hash': hashlib.sha256(content.encode('utf-8')).hexdigest()
                })

                total_size += file_size
                processed_count += 1

                # Enhanced progress indicator with file name and percentage
                if processed_count % 5 == 0 or processed_count == 1:
                    size_mb = total_size / 1024 / 1024
                    if max_files:
                        percentage = (len(collected_files) / max_files) * 100 if max_files > 0 else 0
                        print(f"[COLLECT] {len(collected_files)}/{max_files} files ({percentage:.1f}%) - processing: {rel_path}")
                    else:
                        size_percentage = (total_size / max_size_bytes) * 100 if max_size_bytes > 0 else 0
                        print(f"[COLLECT] {len(collected_files)} files, {size_mb:.1f}/{max_size_mb:.1f}MB ({size_percentage:.1f}%) - processing: {rel_path}")

            except Exception as e:
                rel_path = path_obj.relative_to(project_root).as_posix()
                print(f"[WARN] Could not read {rel_path}: {e}")

        elif path_obj.is_dir():
            for item in sorted(path_obj.iterdir()):
                traverse_and_collect(item)

    # First pass: get complete inventory of eligible files with sizes
    eligible_files_inventory = []
    def inventory_eligible_files(current_path):
        nonlocal eligible_files_inventory

        path_obj = Path(current_path)
        if path_obj.name in exclude_dirs:
            return

        if path_obj.is_file():
            if path_obj.name in exclude_files:
                return
            if path_obj.suffix.lower() not in ['.md', '.txt', '.py', '.js', '.json']:
                return

            try:
                file_size = path_obj.stat().st_size
                rel_path = path_obj.relative_to(project_root).as_posix()
                eligible_files_inventory.append({
                    'path': rel_path,
                    'size': file_size,
                    'size_mb': file_size / 1024 / 1024
                })
            except Exception as e:
                print(f"[WARN] Could not stat {path_obj}: {e}")

        elif path_obj.is_dir():
            for item in sorted(path_obj.iterdir()):
                inventory_eligible_files(item)

    print(f"[INVENTORY] Scanning all eligible files...")
    inventory_eligible_files(Path(project_root))

    # Sort by size (largest first) for intelligent selection
    eligible_files_inventory.sort(key=lambda x: x['size'], reverse=True)

    total_eligible_files = len(eligible_files_inventory)
    total_size_bytes = sum(f['size'] for f in eligible_files_inventory)
    total_size_mb = total_size_bytes / 1024 / 1024

    print(f"[INVENTORY] Found {total_eligible_files} eligible files, {total_size_mb:.1f}MB total")
    print(f"[INVENTORY] Top 10 largest files:")
    for i, file_info in enumerate(eligible_files_inventory[:10]):
        print(f"  {i+1}. {file_info['path']} ({file_info['size_mb']:.2f}MB)")

    # Select files to process based on limits
    selected_files = eligible_files_inventory.copy()  # Start with all files

    # Apply file count limit first
    if max_files and max_files < len(selected_files):
        selected_files = selected_files[:max_files]
        print(f"[SELECT] Limited to top {max_files} files by size")

    # Then apply size limit to the already selected files
    if max_size_mb:
        filtered_files = []
        current_size = 0
        for file_info in selected_files:
            if current_size + file_info['size_mb'] > max_size_mb:
                break
            filtered_files.append(file_info)
            current_size += file_info['size_mb']

        if len(filtered_files) < len(selected_files):
            selected_files = filtered_files
            print(f"[SELECT] Further limited to {len(selected_files)} files to stay under {max_size_mb}MB limit")
        elif max_files and len(selected_files) <= max_files:
            print(f"[SELECT] Processing {len(selected_files)} files (within {max_size_mb}MB limit)")

    # Apply final limits to ensure they are respected
    if max_files and len(selected_files) > max_files:
        selected_files = selected_files[:max_files]
        print(f"[COLLECT] Final file limit applied: {len(selected_files)} files")

    if max_size_mb:
        filtered_files = []
        current_size = 0
        for file_info in selected_files:
            if current_size + file_info['size_mb'] > max_size_mb:
                break
            filtered_files.append(file_info)
            current_size += file_info['size_mb']
        if len(filtered_files) < len(selected_files):
            selected_files = filtered_files
            print(f"[COLLECT] Final size limit applied: {len(selected_files)} files ({current_size:.1f}MB)")

    print(f"[COLLECT] Starting collection of {len(selected_files)} selected files...")
    for i, file_info in enumerate(selected_files):
        file_path = Path(project_root) / file_info['path']

        try:
            content = file_path.read_text(encoding='utf-8')
            file_size = len(content)

            collected_files.append({
                'path': file_info['path'],
                'content': content,
                'size': file_size,
                'hash': hashlib.sha256(content.encode('utf-8')).hexdigest()
            })

            total_size += file_size
            processed_count += 1

            # Progress indicator with file name
            if (i + 1) % 5 == 0 or (i + 1) == len(selected_files) or i == 0:
                percentage = ((i + 1) / len(selected_files)) * 100
                size_mb = total_size / 1024 / 1024
                print(f"[COLLECT] {i + 1}/{len(selected_files)} files ({percentage:.1f}%) - {file_info['path']}")

        except Exception as e:
            print(f"[WARN] Could not read {file_info['path']}: {e}")

    print(f"[COLLECT] Finished: {len(collected_files)}/{len(selected_files)} files collected, {total_size/1024/1024:.1f}MB total")

    # Create consolidated content
    content_parts = []
    file_manifest = []

    print(f"[CONSOLIDATE] Starting content consolidation of {len(collected_files)} files...")
    for i, file_info in enumerate(collected_files):
        file_manifest.append(f"{file_info['path']} ({file_info['size']} bytes, SHA-256: {file_info['hash'][:16]}...)")

        content_parts.append(f"--- START OF FILE: {file_info['path']} ---")
        content_parts.append(file_info['content'])
        content_parts.append(f"--- END OF FILE: {file_info['path']} ---")
        content_parts.append("")

        # Progress for content consolidation - more frequent updates
        if (i + 1) % 20 == 0 or (i + 1) == len(collected_files):
            percentage = ((i + 1) / len(collected_files)) * 100
            print(f"[CONSOLIDATE] {i + 1}/{len(collected_files)} files ({percentage:.1f}%) - {file_info['path']}")

    print(f"[CONSOLIDATE] Content consolidation complete - building final content...")

    # Add manifest
    manifest_content = "\n".join(file_manifest)
    full_content = f"""FILE MANIFEST
Total Files: {len(collected_files)}
Total Size: {total_size} bytes

{manifest_content}

--- CONTENT BOUNDARY ---

""" + "\n".join(content_parts)

    return full_content, len(collected_files), total_size

def main():
    parser = argparse.ArgumentParser(description='Create Provenance-Bound Cognitive Glyphs')
    parser.add_argument('--operation', help='Operation-specific directory to snapshot')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory for glyphs')
    parser.add_argument('--font-size', type=int, default=DEFAULT_FONT_SIZE, help='Font size for glyph text')
    parser.add_argument('--no-provenance', action='store_true', help='Skip provenance binding')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--max-size-mb', type=int, default=10, help='Maximum size in MB (default: 10MB)')
    parser.add_argument('--non-blocking', action='store_true', help='Run in background (non-blocking)')

    args = parser.parse_args()

    if args.non_blocking:
        print("[NON-BLOCKING] Starting glyph forge in background...")
        # Fork process for non-blocking execution
        import subprocess
        import os

        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())

        # Remove non-blocking flag to avoid recursion
        cmd_args = [arg for arg in sys.argv[1:] if arg != '--non-blocking']

        subprocess.Popen([
            sys.executable, str(Path(__file__)),
        ] + cmd_args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("[NON-BLOCKING] Glyph forge started in background. Check output directory for results.")
        return

    project_root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"[PROVENANCE GLYPH FORGE v2.0] Starting advanced optical compression")
    print(f"[CONFIG] Font size: {args.font_size}px, Max files: {args.max_files or 'unlimited'}, Max size: {args.max_size_mb}MB")
    print(f"[CONFIG] Provenance: {not args.no_provenance}, Output: {output_dir.absolute()}")

    # Collect content with limits
    content, file_count, total_size = collect_code_snapshot_v2(
        project_root, args.operation, not args.no_provenance,
        max_files=args.max_files, max_size_mb=args.max_size_mb
    )

    if not content.strip():
        print("[ERROR] No content collected. Check file paths and permissions.")
        sys.exit(1)

    print(f"[COLLECTED] {file_count} files ({total_size/1024/1024:.1f}MB)")

    # Create provenance glyph forge
    forge = ProvenanceGlyphForge(font_size=args.font_size)

    # Generate provenance-bound glyph
    glyph_name = f"provenance_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{glyph_name}.png"

    print("[GLYPH] Creating main provenance glyph...")
    print(f"[GLYPH] Content size: {len(content)/1024/1024:.1f}MB, calculating optimal dimensions...")
    width, height, content_hash = forge.create_provenance_glyph(
        content, output_path, "Provenance-Bound Code Snapshot", not args.no_provenance
    )
    print(f"[GLYPH] Image dimensions calculated: {width}x{height}px ({width*height:,} pixels)")

    # Generate summary glyph
    print("[GLYPH] Creating summary glyph...")
    summary_content = create_smart_summary(content, file_count, total_size)
    summary_path = output_dir / f"{glyph_name}_summary.png"
    forge.create_provenance_glyph(
        summary_content, summary_path, "Smart Summary Glyph", not args.no_provenance
    )

    print(f"\n[PROVENANCE GLYPH FORGE COMPLETE]")
    print(f"  Main Glyph: {output_path} ({width}x{height}px)")
    print(f"  Summary Glyph: {summary_path}")
    print(f"  Content Hash: {content_hash}")
    print(f"  Storage: {output_dir.absolute()}")
    print(f"  Files Processed: {file_count}")
    print(f"  Total Size: {total_size/1024/1024:.1f}MB")

def create_smart_summary(content, file_count, total_size):
    """Create an intelligent summary optimized for glyph compression"""

    lines = content.split('\n')
    summary_parts = []

    # Extract key structural information
    summary_parts.append("CODEBASE SUMMARY")
    summary_parts.append(f"Files: {file_count}")
    summary_parts.append(f"Total Size: {total_size:,} bytes")
    summary_parts.append("")

    # Extract important headers and structure
    important_lines = []
    for line in lines:
        if line.startswith('#') and len(line.strip()) > 5:
            important_lines.append(line)
        elif line.startswith('PROTOCOL:') or line.startswith('DOCTRINE:'):
            important_lines.append(line)

    # Limit to prevent oversized glyphs
    if len(important_lines) > 30:
        important_lines = important_lines[:30]
        important_lines.append("[... TRUNCATED FOR COMPRESSION ...]")

    summary_parts.extend(important_lines)

    return '\n'.join(summary_parts)

if __name__ == "__main__":
    main()