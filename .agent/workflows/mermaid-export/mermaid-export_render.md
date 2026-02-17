---
description: Render a single .mmd file or directory of diagrams to PNG/SVG
argument-hint: "--input <file.mmd|dir/> [--output path] [--svg]"
---

# Render Mermaid Diagrams

Convert `.mmd` Mermaid diagram files to PNG (default) or SVG images.

## Usage
```bash
# Single file
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/export_mmd_to_image.py -i diagram.mmd

# Single file with custom output
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/export_mmd_to_image.py -i diagram.mmd -o output.png

# Entire directory (recursive)
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/export_mmd_to_image.py -i docs/diagrams/

# Render as SVG
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/export_mmd_to_image.py -i diagram.mmd --svg
```

## Prerequisites
- **Node.js** (for npx)
- Mermaid CLI is auto-installed via `npx -y @mermaid-js/mermaid-cli`
