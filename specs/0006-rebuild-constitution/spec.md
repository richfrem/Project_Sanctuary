---
spec_id: "0006"
title: "Rebuild Project Sanctuary Constitution v3"
status: draft
created: 2026-02-01
---

# Spec 0006: Rebuild Project Sanctuary Constitution

## Problem Statement
The current constitution is being habitually ignored by the agent. The Human Gate policy, despite being the most critical rule, is buried in dense text and lacks enforcement hooks.

## Goal
Create a **shorter, more enforceable** Constitution v3 that:
1.  Places Human Gate as the FIRST and MOST PROMINENT principle.
2.  Uses hard, unambiguous language ("MUST", not "should").
3.  Explicitly defines what constitutes a VIOLATION.
4.  Links directly to all critical policy files.

## Scope
- **In Scope:**
  - Rewrite `constitution.md` to be ~50 lines max.
  - Ensure all key policies are referenced (`human_gate_policy.md`, `git_workflow_policy.md`, `tool_discovery_and_retrieval_policy.md`, etc.).
  - Archive the old constitution.
- **Out of Scope:**
  - Changing the underlying policy files themselves (they are fine, the problem is the Constitution's structure).

## Success Criteria
- [ ] New constitution is < 60 lines.
- [ ] Human Gate is the first item after the title.
- [ ] Each principle explicitly defines a "VIOLATION" condition.
- [ ] All critical policy files are hyperlinked.
- [ ] Agent demonstrates compliance in a test session.

## Key References
- Current: `.agent/rules/constitution.md` (142 lines)
- Template: `.agent/rules/constitution_template.md`
- Human Gate: `.agent/rules/human_gate_policy.md`
- Git Workflow: `.agent/rules/git_workflow_policy.md`
