# Bridge Mapping Matrix

This document outlines how files from the **Source of Truth** (Spec Kitty) are transformed and mapped to each target AI agent by the `speckit_system_bridge.py` script.

## Sources
1.  **Rules**: `.kittify/memory/*.md` (e.g., `constitution.md`)
2.  **Workflows**: `.windsurf/workflows/*.md` (e.g., `spec-kitty.accept.md`)

## Mapping Table

| Source Type | Source Path | Target Agent | Destination Path | Transformation Details |
| :--- | :--- | :--- | :--- | :--- |
| **Rules** | `.kittify/memory/*.md` | **Antigravity** | `.agent/rules/{name}.md` | Direct content copy. |
| **Rules** | `.kittify/memory/*.md` | **Claude** | `.claude/CLAUDE.md` | Concatenated into single context file with headers. |
| **Rules** | `.kittify/memory/*.md` | **Gemini** | `GEMINI.md` (Project Root) | Concatenated into single context file with headers. |
| **Rules** | `.kittify/memory/*.md` | **Copilot** | `.github/copilot-instructions.md` | Concatenated into single instruction file. |
| | | | | |
| **Workflows** | `.windsurf/workflows/*.md` | **Antigravity** | `.agent/workflows/{name}.md` | Actor swap: `--actor "windsurf"` -> `--actor "antigravity"`. |
| **Workflows** | `.windsurf/workflows/*.md` | **Claude** | `.claude/commands/{name}.md` | Actor swap: `--actor "windsurf"` -> `--actor "claude"`. |
| **Workflows** | `.windsurf/workflows/*.md` | **Gemini** | `.gemini/commands/{name}.toml` | Wrapped in TOML `prompt = """..."""`.<br>Actor swap: `--actor "gemini"`.<br>Args swap: `$ARGUMENTS` -> `{{args}}`. |
| **Workflows** | `.windsurf/workflows/*.md` | **Copilot** | `.github/prompts/{name}.prompt.md` | Actor swap: `--actor "windsurf"` -> `--actor "copilot"`.<br>Extension change: `.md` -> `.prompt.md`. |
