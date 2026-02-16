---
description: "Inspect inner loop output against acceptance criteria. Pass or generate correction packet."
---

# /agent-handoff:verify — Verification & Correction Loop

## Steps

1. **Inspect changes**:
   ```bash
   python3 scripts/agent_handoff.py verify \
     --packet .agent/handoffs/task_packet_NNN.md \
     --worktree .worktrees/WP-NN
   ```
   This checks:
   - Files modified match the expected scope
   - Each acceptance criterion is addressed
   - No out-of-scope changes

2. **Decision gate**:

   ### ✅ PASS
   - Commit changes in the worktree
   - Update task lane: `spec-kitty agent tasks move-task WP-NN --to done`
   - Proceed to next WP or `/agent-handoff:review`

   ### ❌ FAIL
   - Auto-generate correction packet:
     ```bash
     python3 scripts/agent_handoff.py correct \
       --packet .agent/handoffs/task_packet_NNN.md \
       --feedback "Description of what failed"
     ```
   - Output: `.agent/handoffs/correction_packet_NNN.md`
   - Re-delegate to inner loop with the correction packet
   - Return to `/agent-handoff:delegate`

> [!TIP]
> The correction loop IS the learning mechanism. Each correction refines understanding.
