---
description: Discover untracked scripts and auto-create stub entries
argument-hint: "[--auto-stub] [--include-json] [--json]"
---

# Discover Untracked Tools

Scan the `tools/` directory for Python/JS scripts not yet in the inventory.

## Usage
```bash
# Report only
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py discover

# Auto-create stub entries
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py discover --auto-stub

# Include JSON config files
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py discover --auto-stub --include-json

# JSON output
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py discover --json
```
