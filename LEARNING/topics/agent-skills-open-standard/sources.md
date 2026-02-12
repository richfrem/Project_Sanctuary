# Sources: Agent Skills Open Standard Research

**Date:** 2026-02-11
**Researcher:** Guardian (Antigravity Session)

## Primary Sources

### 1. agentskills.io Specification
- **URL:** https://agentskills.io
- **Status:** [VERIFIED via read_url_content]
- **Content:** Open specification for portable AI agent skills using SKILL.md format
- **Key Contribution:** Defines the canonical skill structure (frontmatter + body)

### 2. Anthropic claude-code-skills Repository
- **URL:** https://github.com/anthropics/claude-code-skills
- **Status:** [VERIFIED - cloned and analyzed]
- **Content:** Reference implementations including skill-creator, doc-coauthoring, mcp-builder
- **Key Contribution:** Progressive disclosure pattern, skill-creator meta-skill

### 3. Anthropic claude-plugins-official Repository
- **URL:** https://github.com/anthropics/claude-plugins-official
- **Status:** [VERIFIED - cloned and analyzed]
- **Content:** 28 plugins including code-review, ralph-loop, security-guidance, LSP integrations
- **Key Contribution:** Multi-agent review with confidence scoring, self-referential iteration

### 4. Anthropic knowledge-work-plugins Repository
- **URL:** https://github.com/anthropics/knowledge-work-plugins
- **Status:** [VERIFIED - cloned and analyzed]
- **Content:** Domain-specific plugins for productivity, product-management, data, finance, etc.
- **Key Contribution:** memory-management tiered architecture, task-management patterns

## Analysis Methods
- Direct filesystem analysis of cloned repositories
- README.md and SKILL.md review for each relevant plugin/skill
- Source code review of hooks (security_reminder_hook.py)
- Cross-referencing with Project Sanctuary's existing skill architecture
