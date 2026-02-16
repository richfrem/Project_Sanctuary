---
description: "Session retrospective: what worked, what failed, fix one thing now"
---

# /agent-handoff:retro — Retrospective & Improvement

## Steps

1. **Generate retrospective template**:
   ```bash
   python3 scripts/agent_handoff.py retro \
     --spec-dir kitty-specs/NNN-feature/ \
     --output .agent/retros/retro_TIMESTAMP.md
   ```

2. **Fill in the retrospective** (agent + human):

   ### What went well?
   - Corrections that improved quality
   - Effective handoff patterns
   - Good spec/plan coverage

   ### What was frustrating?
   - Unnecessary iterations
   - Unclear acceptance criteria
   - Inner loop scope violations

   ### Boy Scout Rule: Fix one thing NOW
   - Identify one small improvement
   - Implement it before closing
   - Examples: update a template, add a missing check, clarify a constraint

3. **Record** — Save the retrospective for future sessions.

> [!TIP]
> The retrospective is where the HUMAN learns too. It's not just for the agent.
