# Agent Skills as an Open Standard

**Date:** 2026-02-11
**Source:** Direct analysis of Anthropic repos + agentskills.io spec
**Status:** Active Research

## 1. The Emerging Standard

The agent skills ecosystem is converging on a portable format:

```
skill-name/
├── SKILL.md          ← Frontmatter (name, description) + instructions
├── references/       ← Progressive disclosure (loaded on demand)
│   └── detailed.md
├── scripts/          ← Helper scripts
├── examples/         ← Reference implementations
└── resources/        ← Assets, templates
```

**Key Insight:** The `SKILL.md` file acts as both a **trigger** (the `description` field in frontmatter) and a **procedure** (the markdown body). This dual role means the description must be rich enough for the agent to self-select when to use the skill.

## 2. Source Analysis

### 2.1 agentskills.io Specification
- Defines `SKILL.md` with YAML frontmatter: `name`, `description`
- Body contains instructions in markdown
- Directory name should match the skill name (lowercase-hyphens)
- No prescribed sub-directory structure — flexible by design

### 2.2 Anthropic's claude-code-skills Repo
- Reference implementations of the spec
- `skill-creator` is the meta-skill (skill for creating skills)
- Key pattern: **Progressive Disclosure** — keep SKILL.md < 500 lines, defer detail to `references/`
- Anti-pattern: Monolithic SKILL.md files that load too much context

### 2.3 Anthropic's claude-plugins-official Repo
- **Plugins ≠ Skills**. Plugins include hooks (pre/post tool execution), MCP servers, and slash commands
- The `code-review` plugin uses multi-agent parallel review with confidence scoring
- The `ralph-loop` plugin implements iterative self-correction via stop hooks
- The `security-guidance` plugin is a pure hook (no skill file) — pattern-matches code edits
- Plugins are Claude Code specific; skills are portable across agents

### 2.4 Anthropic's knowledge-work-plugins Repo
- Domain-specific skill bundles (productivity, product-management, data, etc.)
- `memory-management` skill: Tiered hot/cold memory with CLAUDE.md as working memory
- `task-management` skill: Simple TASKS.md file with sections (Active/Waiting/Done)
- These are workplace productivity skills, not developer tools

## 3. Key Architectural Learnings

### 3.1 The Skill vs Plugin Distinction
| | Skill | Plugin |
|---|---|---|
| **Format** | `SKILL.md` + resources | Hooks + commands + MCP |
| **Portability** | Cross-agent (any AI) | Agent-specific (Claude Code) |
| **Activation** | Description-matching | Hook triggers, slash commands |
| **State** | Stateless (reads files) | Can be stateful (hooks) |
| **Project Sanctuary** | `.agent/skills/` | Not yet adopted |

### 3.2 Progressive Disclosure Pattern
Top skills follow a 3-tier loading pattern:
1. **Tier 1 (Always):** SKILL.md frontmatter — loaded at session start for matching
2. **Tier 2 (On match):** SKILL.md body — loaded when the skill is activated
3. **Tier 3 (On demand):** `references/` files — loaded only when specific detail is needed

This maps directly to our memory-management architecture:
- Tier 1 = Hot cache (cognitive_primer.md)
- Tier 2 = Boot files (guardian_boot_digest.md)
- Tier 3 = Deep storage (LEARNING/topics/)

### 3.3 Confidence-Based Code Review
The code-review plugin introduced a powerful pattern:
- Launch N independent review perspectives in parallel
- Each flags issues with confidence scores (0-100)
- Filter at threshold (default: 80) to eliminate false positives
- This reduces reviewer fatigue dramatically

**Application to Project Sanctuary:** Our `/sanctuary-end` pre-commit check could use this pattern.

### 3.4 Self-Referential Iteration (Ralph Loop)
The Ralph Loop concept — where an agent repeatedly executes the same prompt, seeing its own previous work in files — is philosophically aligned with Protocol 128's recursive learning:
- Both preserve state across iterations via files
- Both use validation gates to determine completion
- Key difference: Ralph Loop is mechanical (bash while loop); Protocol 128 has HITL gates

## 4. What We Built From This

| New Artifact | Source Inspiration | Adaptation |
|---|---|---|
| `memory-management` skill | knowledge-work-plugins | Mapped to our LEARNING/ architecture |
| `code-review` skill | claude-plugins-official | Extracted as portable skill with confidence scoring |
| `references/security-patterns.md` | security-guidance hook | Converted hook patterns to reference table |
| `references/self-correction.md` | ralph-loop plugin | Extracted iteration philosophy for Phase VIII |
| Protocol 128 v4.0 | All sources | Added Skills Integration Layer |

## 5. Open Questions

1. **Should we adopt the plugin model?** Hooks could enforce Zero Trust (e.g., block `git push` without approval). But this requires Claude Code-specific infrastructure.
2. **Skill discovery across agents?** Our `sync_skills.py` copies files, but Gemini/Copilot/Antigravity load skills differently. Is the current approach sufficient?
3. **Skill versioning?** The agentskills.io spec has no versioning mechanism. Should we add one?
