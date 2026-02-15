# AI Resources for LLM Analysis

This directory provides prompts, checklists, and guidance for using LLMs to analyze and modernize Oracle Forms code.

## Quick Start

1. **Load the System Prompt** → `prompts/Investment_stock_valuation_expert_prompt.md`
2. **Gather Context** → Use checklist in `checklists/context-gathering-checklist.md`
3. **Choose a Task Prompt:**

| Task | Prompt File |
|------|-------------|
| Extract Business Rules | `prompts/LLM-prompt-to-find-businessrules.md` |
| Analyze Access Control | `prompts/AccessControl_DeepDive_Prompt.md` |
| Convert to React/.NET | `prompts/Code_Conversion_Prompt.md` |

## Directory Structure

```
tools/ai-resources/
├── prompts/
│   ├── Investment_stock_valuation_expert_prompt.md   # Load first (persona + file map)
│   ├── LLM-prompt-to-find-businessrules.md    # Business rule extraction
│   ├── AccessControl_DeepDive_Prompt.md       # Access control analysis
│   └── Code_Conversion_Prompt.md              # Oracle → React/.NET conversion
└── checklists/
    └── context-gathering-checklist.md          # What to paste into LLM
```

## Key Features

- **Confidence Ratings** - Each finding rated HIGH/MEDIUM/LOW
- **Traceability** - All findings cite specific files and lines
- **Verification Steps** - Built-in checklists to catch hallucinations
- **Examples** - Each prompt includes worked examples

## Related Resources

- [Form Relationships] (Reference Missing: ) - Dependency analysis scripts
- [Business Rule Extraction] (Reference Missing: ) - Analysis templates
- [Analysis Outputs] (Reference Missing: ) - Previous analysis results
