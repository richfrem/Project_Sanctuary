---
description: Search tools by keyword (JSON) or semantic query (ChromaDB vector search)
argument-hint: "\"query\" [--semantic] [--status compliant|stub|needs_review] [-n 5]"
---

# Search Tools

Two search modes:
- **Keyword** (default): Substring match against name/path/description in JSON
- **Semantic** (`--semantic`): Vector similarity search via ChromaDB

## Usage
```bash
# Keyword search (JSON)
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py search "distiller"

# Filter by compliance status
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manage_tool_inventory.py search --status stub

# Semantic search (ChromaDB)
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/tool_chroma.py search "cache management and cleanup"

# Top N results
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/tool_chroma.py search "diagram rendering" -n 3
```
