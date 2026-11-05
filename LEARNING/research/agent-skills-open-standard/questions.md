# Questions: Agent Skills Open Standard

**Date:** 2026-02-11

## Answered

1. **What is the canonical structure for an agent skill?**
   → `SKILL.md` with YAML frontmatter (name, description) + markdown body. Optional `references/`, `scripts/`, `examples/` directories.

2. **How do skills differ from plugins?**
   → Skills are portable (any agent), stateless, description-triggered. Plugins are agent-specific, can be stateful, use hooks/commands.

3. **What's the best pattern for skill content management?**
   → Progressive disclosure: keep SKILL.md < 500 lines, defer detail to `references/`.

4. **How does multi-agent code review work?**
   → N independent perspectives run in parallel, each scores findings 0-100, filter at threshold (80) to reduce false positives.

## Open

5. **Should Project Sanctuary adopt the plugin model (hooks)?**
   → Hooks could enforce Zero Trust policies. But requires Claude Code-specific infrastructure. Needs further investigation.

6. **How should skills be versioned?**
   → agentskills.io spec has no versioning. Consider adding version field to frontmatter.

7. **Can the tiered memory model from memory-management skill be formalized as Protocol 128.1?**
   → The hot cache ↔ deep storage pattern is already implicit in Protocol 128 but not explicitly named. Worth formalizing.
