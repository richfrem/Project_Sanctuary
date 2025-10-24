#!/usr/bin/env python3
"""
capture_glyph_code_snapshot.py (v1.0 - Optical Compression for Code Sharing)

This script creates compressed "Cognitive Glyphs" (images) containing code snapshots
for efficient sharing with LLMs, leveraging the DeepSeek-OCR breakthrough for ~10x
token compression through optical representation.

Key Innovation: Instead of generating massive text files (like capture_code_snapshot.js),
this creates PNG images that can represent the same content with dramatically fewer tokens.

DEPENDENCIES:
- PIL (Pillow) for image generation
- textwrap for text formatting
- os, sys, pathlib for file operations

USAGE:
python3 capture_glyph_code_snapshot.py [--operation PATH] [--output-dir DIR]

EXAMPLE:
python3 capture_glyph_code_snapshot.py --operation WORK_IN_PROGRESS/OPERATION_UNBREAKABLE_CRUCIBLE
"""

import os
import sys
import argparse
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

# Configuration
DEFAULT_FONT_SIZE = 12
DEFAULT_LINE_SPACING = 1.2
DEFAULT_MARGIN = 20
MAX_IMAGE_WIDTH = 2048  # Prevent overly wide images
DEFAULT_OUTPUT_DIR = "dataset_code_glyphs"

# Color scheme for readability
COLORS = {
    'background': '#FFFFFF',  # White background
    'text': '#000000',        # Black text
    'header': '#2E3440',      # Dark blue-gray for headers
    'separator': '#4C566A',   # Medium gray for separators
    'code_bg': '#F8F9FA',     # Light gray for code blocks
    'comment': '#6C757D',     # Gray for comments
}

class GlyphForge:
    """Forge that creates Cognitive Glyphs from code snapshots"""

    def __init__(self, font_size=DEFAULT_FONT_SIZE, line_spacing=DEFAULT_LINE_SPACING):
        self.font_size = font_size
        self.line_spacing = line_spacing
        try:
            # Try to use a monospace font for code
            self.font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
            self.header_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", int(font_size * 1.2))
        except:
            try:
                # Fallback to system default
                self.font = ImageFont.load_default()
                self.header_font = ImageFont.load_default()
            except:
                print("[ERROR] Could not load any font. Install PIL with font support.")
                sys.exit(1)

    def estimate_text_dimensions(self, text, max_width=None):
        """Estimate pixel dimensions needed for text rendering"""
        lines = text.split('\n')
        max_line_width = 0
        total_height = 0

        for line in lines:
            bbox = self.font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]

            max_line_width = max(max_line_width, line_width)
            total_height += int(line_height * self.line_spacing)

        # Constrain width if specified
        if max_width and max_line_width > max_width:
            max_line_width = max_width

        return max_line_width, total_height

    def create_glyph(self, content, filename, title="Code Snapshot Glyph"):
        """Create a Cognitive Glyph (PNG image) from text content"""

        # Prepare content with formatting
        formatted_content = self.format_content_for_glyph(content, title)

        # Calculate image dimensions
        content_width, content_height = self.estimate_text_dimensions(
            formatted_content, MAX_IMAGE_WIDTH - 2 * DEFAULT_MARGIN
        )

        img_width = min(content_width + 2 * DEFAULT_MARGIN, MAX_IMAGE_WIDTH)
        img_height = content_height + 2 * DEFAULT_MARGIN

        # Create image
        img = Image.new('RGB', (img_width, img_height), COLORS['background'])
        draw = ImageDraw.Draw(img)

        # Draw content
        y_position = DEFAULT_MARGIN
        for line in formatted_content.split('\n'):
            if line.startswith('=== ') and line.endswith(' ==='):
                # Header line
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['header'], font=self.header_font)
                bbox = self.header_font.getbbox(line)
                line_height = bbox[3] - bbox[1]
            elif line.startswith('--- ') and line.endswith(' ---'):
                # Separator line
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['separator'], font=self.font)
                bbox = self.font.getbbox(line)
                line_height = bbox[3] - bbox[1]
            else:
                # Regular text
                draw.text((DEFAULT_MARGIN, y_position), line, fill=COLORS['text'], font=self.font)
                bbox = self.font.getbbox(line)
                line_height = bbox[3] - bbox[1]

            y_position += int(line_height * self.line_spacing)

        # Save the glyph
        img.save(filename, 'PNG', optimize=True)
        print(f"[GLYPH FORGED] {filename} ({img_width}x{img_height}px)")

        return img_width, img_height

    def format_content_for_glyph(self, content, title):
        """Format text content for optimal glyph rendering"""
        lines = content.split('\n')
        formatted_lines = []

        # Add title header
        formatted_lines.append(f"=== {title.upper()} ===")
        formatted_lines.append("")

        for line in lines:
            # Handle different line types for visual distinction
            if line.strip().startswith('#'):
                # Headers - keep as is
                formatted_lines.append(line)
            elif line.strip() == '' or line.strip() == '---':
                # Empty lines or separators
                formatted_lines.append(line)
            elif len(line.strip()) > 0:
                # Wrap long lines for better readability
                if len(line) > 100:
                    wrapped = textwrap.wrap(line, width=80)
                    formatted_lines.extend(wrapped)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

def collect_code_snapshot(project_root, operation_path=None):
    """Collect code snapshot similar to capture_code_snapshot.js but optimized for glyph compression"""

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
        'capture_code_snapshot.js', 'capture_glyph_code_snapshot.py',
        '.DS_Store', '.gitignore', 'PROMPT_PROJECT_ANALYSIS.md'
    }

    core_essence_files = {
        'The_Garden_and_The_Cage.md',
        'README.md',
        '01_PROTOCOLS/00_Prometheus_Protocol.md',
        '01_PROTOCOLS/27_The_Doctrine_of_Flawed_Winning_Grace_v1.2.md',
        'chrysalis_core_essence.md',
        'Socratic_Key_User_Guide.md'
    }

    if operation_path:
        # Override with operation-specific files
        op_full_path = Path(project_root) / operation_path
        if op_full_path.exists():
            core_essence_files = set()
            for md_file in op_full_path.glob('*.md'):
                rel_path = md_file.relative_to(project_root).as_posix()
                core_essence_files.add(rel_path)

    collected_content = []
    file_count = 0

    def traverse_and_collect(current_path):
        nonlocal file_count

        path_obj = Path(current_path)
        if path_obj.name in exclude_dirs:
            return

        if path_obj.is_file():
            if path_obj.name in exclude_files:
                return

            if path_obj.suffix.lower() != '.md':
                return

            try:
                content = path_obj.read_text(encoding='utf-8')
                rel_path = path_obj.relative_to(project_root).as_posix()

                # Format for glyph
                file_header = f"--- START OF FILE: {rel_path} ---"
                file_footer = f"--- END OF FILE: {rel_path} ---"

                collected_content.append(file_header)
                collected_content.append(content)
                collected_content.append(file_footer)
                collected_content.append("")  # Empty line separator

                file_count += 1

            except Exception as e:
                error_msg = f"[ERROR reading {rel_path}]: {str(e)}"
                collected_content.append(error_msg)

        elif path_obj.is_dir():
            for item in sorted(path_obj.iterdir()):
                traverse_and_collect(item)

    traverse_and_collect(Path(project_root))

    return '\n'.join(collected_content), file_count

def main():
    parser = argparse.ArgumentParser(description='Create Cognitive Glyphs from code snapshots')
    parser.add_argument('--operation', help='Operation-specific directory to snapshot')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory for glyphs')
    parser.add_argument('--font-size', type=int, default=DEFAULT_FONT_SIZE, help='Font size for glyph text')
    parser.add_argument('--max-width', type=int, default=MAX_IMAGE_WIDTH, help='Maximum glyph width in pixels')

    args = parser.parse_args()

    project_root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"[GLYPH FORGE v1.0] Starting optical compression from: {project_root}")
    print(f"[CONFIG] Font size: {args.font_size}px, Max width: {args.max_width}px")

    # Collect content
    content, file_count = collect_code_snapshot(project_root, args.operation)

    if not content.strip():
        print("[ERROR] No content collected. Check file paths and permissions.")
        sys.exit(1)

    print(f"[COLLECTED] {file_count} markdown files processed")

    # Create glyph forge
    forge = GlyphForge(font_size=args.font_size)

    # Generate different glyph types
    glyph_types = [
        ("full_snapshot", content, "Complete Code Snapshot"),
        ("core_essence", extract_core_essence(content), "Core Essence Only"),
        ("compressed_summary", create_compressed_summary(content), "Compressed Summary")
    ]

    for glyph_name, glyph_content, title in glyph_types:
        if glyph_content.strip():
            output_path = output_dir / f"{glyph_name}_glyph.png"
            width, height = forge.create_glyph(glyph_content, output_path, title)

            # Estimate token savings
            original_tokens = len(content.split()) * 1.3  # Rough token estimation
            glyph_tokens = (width * height) / 10000  # Rough vision token estimation
            compression_ratio = original_tokens / glyph_tokens if glyph_tokens > 0 else 0

            print(".1f")
        else:
            print(f"[SKIPPED] {glyph_name} - no content to glyph")

    print(f"\n[GLYPH FORGE COMPLETE] Glyphs saved to: {output_dir.absolute()}")

def extract_core_essence(full_content):
    """Extract only core essence files from full content"""
    lines = full_content.split('\n')
    core_lines = []
    in_core_file = False

    core_files = [
        'The_Garden_and_The_Cage.md',
        'README.md',
        '01_PROTOCOLS/00_Prometheus_Protocol.md',
        '01_PROTOCOLS/27_The_Doctrine_of_Flawed_Winning_Grace_v1.2.md',
        'chrysalis_core_essence.md',
        'Socratic_Key_User_Guide.md'
    ]

    for line in lines:
        if line.startswith('--- START OF FILE: '):
            filename = line.replace('--- START OF FILE: ', '').replace(' ---', '')
            in_core_file = any(core_file in filename for core_file in core_files)

        if in_core_file:
            core_lines.append(line)

    return '\n'.join(core_lines)

def create_compressed_summary(full_content):
    """Create a compressed summary suitable for glyph representation"""
    lines = full_content.split('\n')
    summary_lines = []

    # Extract key headers and structure
    for line in lines:
        if line.startswith('#') and len(line.strip()) > 5:
            # Keep headers
            summary_lines.append(line)
        elif line.startswith('--- START OF FILE: ') and 'PROTOCOLS' in line:
            # Keep protocol file markers
            summary_lines.append(line.replace('--- START OF FILE: ', 'PROTOCOL: '))
        elif line.startswith('## ') and any(keyword in line.lower() for keyword in ['core', 'principle', 'mandate', 'doctrine']):
            # Keep important section headers
            summary_lines.append(line)

    # Limit to reasonable size for glyph
    if len(summary_lines) > 50:
        summary_lines = summary_lines[:50]
        summary_lines.append("\n[... CONTENT TRUNCATED FOR GLYPH COMPRESSION ...]")

    return '\n'.join(summary_lines)

if __name__ == "__main__":
    main()