# Procedural Memory Architecture (`LEARNING/procedures/`)

Procedural memory represents **executable, repeatable playbooks and action recipes** discovered by autonomous agents in Project Sanctuary.

Unlike static documentation, every procedural recipe in Sanctuary:
1. Is bound to an explicit schema with inputs, prerequisites, steps, success criteria, and abort/recovery paths.
2. Tracks empirical outcomes (`success_count`, `failure_count`, `blocked_count`) bound to specific **versions and runtime environments**.
3. Integrates with the progressive disclosure engine (`managing-sovereign-memory`).

---

## Procedure Specification Format

Every procedure file must begin with YAML frontmatter conforming to this schema:

```yaml
---
id: proc-unique-kebab-slug
name: Human Readable Title
version: 1.0.0
domain: git | architecture | testing | fine-tuning
applicability: "When and why to use this procedure."
inputs:
  - param_name: "Description and format"
prerequisites:
  - "Environmental condition that must be true prior to execution"
steps:
  - "1. First step"
  - "2. Second step"
success_criteria:
  - "Deterministic check that verifies success (command or assertion)"
abort_conditions:
  - "Condition that requires halting immediately"
recovery_guidance:
  - "Action to take if execution fails or is blocked"
outcomes:
  environment: "macOS-darwin"
  success_count: 0
  failure_count: 0
  blocked_count: 0
  last_verified: "YYYY-MM-DD"
---
```

---

## Progressive Disclosure Integration

Procedural recipes are accessed progressively via `managing-sovereign-memory`:
- **Tier 1 (Index)**: Lists available procedures, applicability, and empirical outcome counters.
- **Tier 2 (Summary)**: Returns prerequisites, inputs, and recovery guidance.
- **Tier 3 (Deep)**: Full step-by-step recipe with code blocks and verification checks.

Outcomes are recorded via `python3 scripts/record_procedure_outcome.py --id <proc-id> --outcome success|failure|blocked`.
