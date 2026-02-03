---
description: Standard post-flight closure for all codify/investigate workflows. Handles human review, git commit, PR verification, and cleanup.
tier: 1
---

**Command:**
- `python tools/cli.py workflow end`

**Purpose:** Standardized closure sequence for all workflows. Ensures consistent human review gates, git hygiene, and task tracking.

**This is an ATOMIC workflow (Tier 1).**

**Called By:** All `/codify-*` and `/speckit-*` workflows

---

## Step 1: Human Review Approval

1. **Present Checklist**: Show the completed granular subtasks from task file.
2. **Present Links**: Provide the **Review Items** section with artifact links.
3. **Wait for LGTM**: Obtain explicit developer approval in chat.

> > [!IMPORTANT] **Protocol 128 Pre-Requisites (Must Complete First):**
> **Do NOT proceed** until user explicitly approves (e.g., "LGTM", "approved", "go ahead").
> 1. **Seal** → `/workflow-seal` (snapshot created)
> 2. **Persist** → `/workflow-persist` (HuggingFace upload)
> 3. **Retrospective** → `/workflow-retrospective` (self-reflection)

---

## Step 2: Final Git Commit

```bash
scripts/bash/workflow-end.sh "[CommitMessage]" [Optional:FilePaths]
```

*Example:* `scripts/bash/workflow-end.sh "docs: add new workflow component"`

---

## Step 3: PR Verification (Critical Gate)

**STOP AND ASK:** "Has the Pull Request been merged?"

> [!CAUTION]
> **Wait** for explicit "Yes" or "merge complete" from User.
> Do NOT proceed until confirmed.

---

## Step 4: Cleanup & Closure

After merge confirmation:
```bash
scripts/bash/workflow-cleanup.sh
```

---

## Step 5: Task File Closure

```bash
mv tasks/in-progress/[TaskFile] tasks/done/
```

*Example:* `mv tasks/in-progress/0099-implement-feature.md tasks/done/`

---

## Output
- Git branch merged and deleted
- Task file moved to `tasks/done/`
- Ready for next task

// turbo-all
