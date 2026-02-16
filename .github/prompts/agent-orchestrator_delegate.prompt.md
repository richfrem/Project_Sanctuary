---
description: "Generate a strategy packet and hand off execution to the inner loop agent"
---

# /agent-handoff:delegate — Handoff to Inner Loop

## Steps

1. **Create isolated workspace** (if using worktrees):
   ```bash
   spec-kitty implement WP-NN
   ```

2. **Generate strategy packet**:
   ```bash
   python3 scripts/agent_handoff.py packet \
     --wp WP-01 \
     --spec-dir kitty-specs/NNN-feature/
   ```
   Output: `.agent/handoffs/task_packet_NNN.md`

3. **Review the packet** — Confirm it contains:
   - [ ] Clear objective
   - [ ] Acceptance criteria
   - [ ] File scope (what to touch)
   - [ ] Constraints (what NOT to do)

4. **Signal ready** — Tell the user:
   > "Strategy packet ready at `.agent/handoffs/task_packet_NNN.md`. Launch inner loop agent."

5. **User launches inner loop**:
   ```bash
   claude "Read .agent/handoffs/task_packet_NNN.md and execute the mission. Do NOT use git commands."
   ```

> [!IMPORTANT]
> The inner loop agent must NOT run git commands. All version control is the outer loop's responsibility.
