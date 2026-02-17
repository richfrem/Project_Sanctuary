---
description: High-speed high-fidelity RLM distillation of a tool script using agentic intelligence.
---

# Agent-Driven Distillation Workflow (TOOLS)

Use this workflow to bypass slow local Ollama models when summarizing **scripts** for the RLM tool inventory.

## 🚀 Execution Steps

1.  **Identify the tool script** (e.g., `plugins/tool-inventory/scripts/manage_tool_inventory.py`).
2.  **Read the source code** using `view_file`.
3.  **Read the tool prompt** from [rlm_summarize_tool.md](plugins/tool-inventory/resources/prompts/rlm/rlm_summarize_tool.md).
4.  **Generate a high-fidelity JSON summary** (Strict JSON object).
5.  **Execute the distiller** for the **tool** type (Targets `rlm_tool_cache.json`):

```bash
python3 plugins/tool-inventory/scripts/distiller.py --file <path_to_script> --type tool --summary '<json_summary>'
```

6.  **Verify** that `rlm_tool_cache.json` and `tool_inventory.json` are updated.
