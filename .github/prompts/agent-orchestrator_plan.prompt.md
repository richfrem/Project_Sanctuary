---
description: "Start the outer loop: specify, plan, and generate tasks via spec-kitty"
---

# /agent-handoff:plan — Outer Loop: Strategy

## Prerequisites
- `spec-kitty-cli` installed (`npm i -g spec-kitty-cli`)
- Project initialized (`spec-kitty init .`)

## Steps

1. **Specify** — Create or update the feature spec:
   ```bash
   spec-kitty specify
   ```

2. **Plan** — Generate the implementation plan:
   ```bash
   spec-kitty plan
   ```

3. **Tasks** — Break plan into work packages:
   ```bash
   spec-kitty tasks
   ```

4. **Verify artifacts exist**:
   ```bash
   python3 scripts/agent_handoff.py scan --spec-dir <kitty-specs/NNN-feature/>
   ```
   Confirm: `spec.md`, `plan.md`, `tasks.md` all present.

5. **Select first WP** for execution and proceed to `/agent-handoff:delegate`.

> [!TIP]
> Each spec directory is a learning iteration. The history of specs shows how understanding evolved.
