---
description: Generate TOOL_INVENTORY.md documentation from the JSON registry
argument-hint: "[--output path/to/TOOL_INVENTORY.md]"
---

# Generate Markdown Docs

Render the JSON inventory as a human-readable Markdown document with category tables.

## Usage
```bash
# Default (writes next to inventory JSON)
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py generate

# Custom output
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py generate --output docs/TOOLS.md
```
