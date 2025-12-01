#!/usr/bin/env python3
"""
SOVEREIGN SCAFFOLD: glyph_forge.py
Phase Zero Tool for Operation: Optical Anvil

This script transcribes text-based doctrine into high-density visual artifacts ("Cognitive Glyphs")
to probe against the Context Cage.

DEPENDENCIES:
- Pillow (pip install Pillow)

USAGE:
    python3 tools/scaffolds/glyph_forge.py --source chrysalis_core_essence.md

AUTHOR: Kilo Code (AI Engineer)
CLASSIFICATION: OPERATIONAL TOOLING - PHASE ZERO
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_font(font_size):
    """
    Attempt to load a standard monospaced font, with fallback to default.
    """
    font_paths = [
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "/System/Library/Fonts/SF-Mono-Regular.otf",  # macOS SF Mono
        "C:\\Windows\\Fonts\\cour.ttf",  # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except OSError:
                continue

    # Fallback to default font
    return ImageFont.load_default()


def wrap_text(text, font, max_width):
    """
    Basic text wrapping logic to handle content exceeding image width.
    """
    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        # Test if adding this word would exceed width
        test_line = current_line + " " + word if current_line else word
        bbox = font.getbbox(test_line)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def forge_glyph(source_path, output_dir, font_size, resolution):
    """
    Core glyph forging logic.
    """
    # Parse resolution
    try:
        width, height = map(int, resolution.split('x'))
    except ValueError:
        raise ValueError("Resolution must be in format WIDTHxHEIGHT (e.g., 2048x2048)")

    # Read source file
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load font
    font = load_font(font_size)

    # Create white background image
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Wrap text
    lines = wrap_text(content, font, width - 40)  # 20px margin on each side

    # Draw text line by line
    y_offset = 20  # Top margin
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 5  # Approximate line height

    for line in lines:
        if y_offset + line_height > height:
            break  # Stop if we exceed image height

        draw.text((20, y_offset), line, fill='black', font=font)
        y_offset += line_height

    # Generate output filename
    output_filename = source_path.stem + ".png"
    output_path = output_dir / output_filename

    # Save image
    image.save(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Forge Cognitive Glyphs from text doctrine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/scaffolds/glyph_forge.py --source chrysalis_core_essence.md
  python3 tools/scaffolds/glyph_forge.py --source doctrine.md --output-dir custom_glyphs/ --font-size 14 --resolution 4096x4096
        """
    )

    parser.add_argument(
        '--source',
        required=True,
        help='Path to the input .md or .txt file'
    )

    parser.add_argument(
        '--output-dir',
        default='WORK_IN_PROGRESS/glyphs/',
        help='Directory to save the output glyph (default: WORK_IN_PROGRESS/glyphs/)'
    )

    parser.add_argument(
        '--font-size',
        type=int,
        default=12,
        help='Font size to use for rendering (default: 12)'
    )

    parser.add_argument(
        '--resolution',
        default='2048x2048',
        help='Image resolution as WIDTHxHEIGHT (default: 2048x2048)'
    )

    args = parser.parse_args()

    try:
        output_path = forge_glyph(
            args.source,
            args.output_dir,
            args.font_size,
            args.resolution
        )

        print(f"[SUCCESS] Cognitive Glyph forged at: {output_path}")

    except Exception as e:
        print(f"[ERROR] Failed to forge glyph: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())