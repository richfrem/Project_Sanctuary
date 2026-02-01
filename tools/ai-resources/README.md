# AI Resources for LLM Analysis

This directory provides prompts, checklists, and guidance for using LLMs in Project Sanctuary workflows.

## Quick Start

1. **Load the System Prompt** → `prompts/Context_Bundler_System_Prompt.md`
2. **Gather Context** → Use checklist in `checklists/context-gathering-checklist.md`
3. **Choose a Task Prompt:**

| Task | Prompt File |
|------|-------------|
| Extract Business Rules | `prompts/LLM-prompt-to-find-businessrules.md` |
| Analyze Access Control | `prompts/AccessControl_DeepDive_Prompt.md` |
| Code Analysis | `prompts/Code_Conversion_Prompt.md` |

## Directory Structure

```
tools/ai-resources/
├── prompts/
│   ├── Context_Bundler_System_Prompt.md       # Load first (persona + file map)
│   ├── LLM-prompt-to-find-businessrules.md    # Business rule extraction
│   ├── AccessControl_DeepDive_Prompt.md       # Access control analysis
│   └── Code_Conversion_Prompt.md              # Code analysis and conversion
└── checklists/
    └── context-gathering-checklist.md          # What to paste into LLM
```

## Key Features

- **Confidence Ratings** - Each finding rated HIGH/MEDIUM/LOW
- **Traceability** - All findings cite specific files and lines
- **Verification Steps** - Built-in checklists to catch hallucinations
- **Examples** - Each prompt includes worked examples

## Related Resources

- [Tool Inventory](../../tools/TOOL_INVENTORY.md) - Available CLI tools
- [Workflow Inventory](../../docs/antigravity/workflow/WORKFLOW_INVENTORY.md) - Available workflows
- [RLM Summary Cache](../../.agent/learning/rlm_summary_cache.json) - Semantic context
