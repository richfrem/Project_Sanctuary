---
description: Create a new Architecture Decision Record from template
argument-hint: "\"Title\" [--context text] [--decision text] [--consequences text]"
---

# Create ADR

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/adr_manager.py create "Use ChromaDB for tool search" \
  --context "Need semantic search for tool discovery" \
  --decision "Embed ChromaDB in tool-inventory plugin" \
  --consequences "Adds ~500MB dep, but enables vector search"
```

Auto-assigns next available ADR number. Uses template from `templates/adr-template.md`.
