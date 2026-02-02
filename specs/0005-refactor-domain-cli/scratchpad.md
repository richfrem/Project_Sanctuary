# Scratchpad

**Spec**: 0005-refactor-domain-cli
**Created**: 2026-02-01

> **Purpose**: Capture ideas as they come up, even if out of sequence.
> At the end of the spec, process these into the appropriate places.

---

## Spec-Related Ideas
<!-- Clarifications, scope questions, requirements, "what are we actually building?" -->

- [ ] Should we also consolidate any other `scripts/*.py` CLIs beyond domain_cli.py?
- [ ] Consider adding `--json` output flag for all domain commands (future enhancement)?

---

## Plan-Related Ideas
<!-- Architecture, design decisions, alternatives, "how should we build it?" -->

- [ ] The domain commands could be organized under a `domain` parent command (e.g., `cli.py domain chronicle list`) vs flat (`cli.py chronicle list`). Current approach: flat for consistency with existing patterns.

---

## Task-Related Ideas
<!-- Things to add to tasks.md, step refinements, "what specific work needs doing?" -->

- [ ] 

---

## Out-of-Scope (Future Backlog)
<!-- Ideas for /create-task after this spec closes, "good idea but not now" -->

- [ ] Consider consolidating `scripts/cortex_cli.py` deprecation (already at parity, just needs deletion).
- [ ] Add `--json` output for programmatic consumption of all CLI commands.

---

## Processing Checklist (End of Spec)
- [ ] Reviewed all items above
- [ ] Spec-related items incorporated into `spec.md` or discussed
- [ ] Plan-related items incorporated into `plan.md` or discussed
- [ ] Task-related items added to `tasks.md`
- [ ] Out-of-scope items logged via `/create-task`
