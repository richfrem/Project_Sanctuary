---
description: High-speed RLM distillation of project documentation using agentic intelligence.
---

# Agent-Driven Distillation Workflow (PROJECT)

Use this workflow to bypass slow local Ollama models when summarizing **documentation** (MD, TXT) for the RLM project cache.

## 🚀 Execution Steps

1.  **Identify the document** (e.g., `docs/architecture_overview.md`).
2.  **Read the document** using `view_file`.
3.  **Read the project prompt** from [rlm_summarize_sanctuary.md](plugins/rlm-factory/resources/prompts/rlm/rlm_summarize_sanctuary.md).
4.  **Generate a high-quality one-sentence summary** (Concise text sentence).
5.  **Execute the distiller** for the **project** type (Targets `rlm_summary_cache.json`):

```bash
python3 plugins/rlm-factory/scripts/distiller.py --file <path_to_doc> --type project --summary '<summary_text>'
```

6.  **Verify** that `rlm_summary_cache.json` is updated.
