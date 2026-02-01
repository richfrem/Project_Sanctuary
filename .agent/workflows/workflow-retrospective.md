---
description: Mandatory self-retrospective and continuous improvement check after completing any codify workflow.
tier: 1
---

**Command:**
- `python tools/cli.py workflow retrospective`
- `scripts/bash/workflow-retrospective.sh` (Wrapper)

**Purpose:** Enforce the "Boy Scout Rule" - leave the codebase better than you found it. Every workflow execution should improve tooling, documentation, or process.

> **Scope**: This retrospective covers observations from BOTH the **User** AND the **Agent**. Both parties should contribute improvement ideas.

**This is an ATOMIC workflow (Tier 1).**

**Called By:** All `/codify-*` workflows (called before `/workflow-end`)

---

## Step 0: Collect User Feedback (MANDATORY FIRST)

> [!CRITICAL] **STOP! READ THIS CAREFULLY!**
> Do NOT check any boxes. Do NOT run any scripts.
> You MUST output the questions below to the User and **WAIT** for their reply.
> **Failure to do this is a Protocol Violation.**

**Questions to ask:**
1. What went well for you during this workflow?
2. What was frustrating or confusing?
3. Did I (the Agent) ignore any of your questions or feedback?
4. Do you have any suggestions for improvement?

**Agent Action:** Copy the User's answers into Part A of `retrospective.md`.

---

## Step 1: Workflow Smoothness Check (Agent Self-Assessment)

How many times did you have to correct yourself or retry a step?

- [ ] **0-1 Retries**: Smooth execution.
- [ ] **2+ Retries**: Bumpy execution. (Document why below)

**If bumpy, note:**
- Which step(s) failed?
- What caused the retry?
- Was it a tool bug, unclear documentation, or missing data?

---

## Step 2: Tooling & Documentation Gap Analysis

Check each area for gaps:

- [ ] **CLI Tools**: Did any `cli.py` commands fail or produce confusing output?
- [ ] **Template Check**: Did the Overview template lack a section for data you found?
  - If yes: Update the template file immediately.
- [ ] **Workflow Check**: Was any step in this workflow unclear or missing?
  - If yes: Note which step needs clarification.
- [ ] **Inventory Check**: Did the inventory scan correctly pick up your new artifacts?

---

## Step 2.5: Backlog Identification (Continuous Improvement)

Did you identify any non-critical issues, technical debt, or naming inconsistencies?

- [ ] **No**: Proceed.
- [ ] **Yes**:
    1.  **Create Task**: Run `/create-task` to log the item in `tasks/backlog/`.
    2.  **Log It**: Mention the new task ID in your closing summary.

---

## Step 3: Immediate Improvement (The "Boy Scout Rule")

**You MUST strictly choose one action:**

- [ ] **Option A (Fix Code)**: Fix a script bug identified in this run. (Do it now)
- [ ] **Option B (Fix Docs)**: Clarify a confusing step in the workflow file. (Do it now)
- [ ] **Option C (New Task)**: The issue is too big for now. Create a new Task file in `tasks/backlog/`.
- [ ] **Option D (No Issues)**: The workflow was flawless. (Rare but possible)

---

## Step 4: Next Spec Planning (After Current Spec Closes)

**If this is the FINAL task of the current Spec**, prepare for the next work:

1. **List Backlog Candidates**: Review `tasks/backlog/` for high-priority items.
2. **Recommend Next Spec**: Propose 2-3 options with brief rationale.
3. **Get User Confirmation**: Wait for user to select the next Spec.
4. **Create Next Spec**: Run `/speckit-specify [ChosenItem]` to start the next cycle.

**Example Recommendation:**
> Based on this Spec's learnings, I recommend:
> 1. **0116-enhance-retrospective** (High) - Process improvement we identified
> 2. **0118-pure-python-orchestration** (ADR-0031) - Deferred from this Spec
> 3. **0119-fix-mermaid-export** (Low) - Tool gap
>
> Which should be next?

---

## Execution Protocol

1. **Ask User** for their feedback (Step 0).
2. **Select** one improvement option.
3. **Perform** the selected improvement NOW, before calling `/workflow-end`.
4. **Record** what you improved in the git commit message.

> [!IMPORTANT]
> Do NOT mark the workflow as complete until you've:
> 1. Collected User feedback
> 2. Performed your retrospective action

---

## Usage in Parent Workflows

Insert before the closure phase:
```markdown
### Step N: Self-Retrospective
/workflow-retrospective

### Step N+1: Closure
/workflow-end "docs: ..." tasks/in-progress/[TaskFile]
```

// turbo-all
