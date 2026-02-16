---
description: "Bundle project context into a single markdown for red-team or human review"
---

# /agent-handoff:review — Context Bundling for Review

## Steps

1. **Bundle context for review**:
   ```bash
   python3 scripts/agent_handoff.py bundle \
     --files file1.py file2.md spec.md plan.md \
     --output .agent/reviews/review_bundle_TIMESTAMP.md
   ```
   Or pass a manifest:
   ```bash
   python3 scripts/agent_handoff.py bundle \
     --manifest review_manifest.json \
     --output .agent/reviews/review_bundle.md
   ```

2. **Present to reviewer**:
   - Human: Share the bundle file for review
   - AI Red Team: Pipe to another agent:
     ```bash
     claude "You are a code reviewer. Read .agent/reviews/review_bundle.md and provide feedback."
     ```

3. **Process feedback**:
   - If changes needed → create correction packet → re-delegate
   - If approved → proceed to `/agent-handoff:retro`

> [!NOTE]
> The bundle is a self-contained markdown with file headers and content. No external tools needed.
