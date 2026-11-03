# Connectors

This plugin is the **Forge** for Project_Sanctuary. Its purpose is to evolve the Sanctuary Guardian
and grow the capabilities of the entire Sanctuary ecosystem.

## Required: Generation Backend

| Dependency | Tool | Notes |
|-----------|------|-------|
| `copilot-cli` | `copilot -p "..."` | **Required.** GPT-5.0 mini via GitHub Copilot account. Free, zero tokens, M1-friendly. |

**No other LLM backends permitted.** `claude-cli` and `gemini-cli` are NOT free and must not be used in the evolution loop.

## Sanctuary Ecosystem Dependencies

| Plugin | Purpose |
|--------|---------|
| `sanctuary-guardian` | Primary identity + memory - this forge evolves the guardian |
| `chronicle-manager` | Living Chronicle - auto-journal keep events and milestones |
| `protocol-manager` | Governance - major architectural decisions require protocol entries |

## Installed Skill Dependencies

| Skill | Purpose |
|-------|---------|
| `continuous-skill-optimizer` | Optimizes SKILL.md trigger descriptions after ledger updates |
| `create-skill` | When a gap requires a full new SKILL.md (not just a script) |
| `create-sub-agent` | When a gap requires a new autonomous sub-agent |
| `orchestrator` | Routing - decides which loop pattern to use per gap |
| `copilot-cli-agent` | Inner loop executor - GPT-5.0 mini, free, zero tokens |

## Degradation Behavior (Standalone Mode)

If companion plugins are unavailable:
- Chronicle auto-journaling skipped (manual `/chronicle` entry recommended)
- Continuous optimization skipped (manual trigger required)
- Core gap-identification + script generation + sandbox loop runs fully unaffected

## Optional MCP Tool Connectors

| Category | Examples | Used By |
|----------|---------|---------|
| `~~source-control` | GitHub MCP, GitLab MCP | `commit_tool.py` - enhanced commit metadata |
| `~~code-execution` | Jupyter kernel, E2B sandbox | `sandbox_runner.py` - isolated execution environment |
| `~~vector-search` | ChromaDB, Pinecone | Gap scanner - semantic similarity search for existing tools |
