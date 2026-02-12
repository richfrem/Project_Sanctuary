---
name: Spec Kitty Workflow
description: Standard operating procedures for the Spec Kitty agentic workflow (Plan -> Implement -> Review -> Merge).
---

# Spec Kitty Workflow

Standard lifecycle for implementing features using Spec Kitty.

## CRITICAL: Anti-Simulation Rules

> **YOU MUST ACTUALLY RUN EVERY COMMAND LISTED BELOW.**
> Describing what you "would do", summarizing expected output, or marking
> a step complete without pasting real tool output is a **PROTOCOL VIOLATION**.
>
> **Proof = pasted command output.** No output = not done.

### Known Agent Failure Modes (DO NOT DO THESE)
1. **Checkbox theater**: Marking `[x]` without running the command or verification tool
2. **Manual file creation**: Writing spec.md/plan.md/tasks.md by hand instead of using CLI
3. **Kanban neglect**: Not updating task lanes, so dashboard shows stale state
4. **Verification skip**: Marking a phase complete without running `verify_workflow_state.py`
5. **Closure amnesia**: Finishing code but skipping review/merge/closure steps

---

## 0. Mandatory Planning Phase (Do NOT Skip)

Before implementing any code, you MUST generate artifacts using the CLI.
**Manual creation of `spec.md`, `plan.md`, or `tasks/` files is STRICTLY FORBIDDEN.**

### Step 0a: Specify
```bash
/spec-kitty.specify
```
**PROOF**: Paste output confirming spec.md was generated.

Then verify:
```bash
python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase specify
```
**PROOF**: Paste the verification output showing the checkmark.
**STOP**: Do NOT proceed to Plan until verification passes.

### Step 0b: Plan
```bash
/spec-kitty.plan
```
**PROOF**: Paste output confirming plan.md was generated.

Then verify:
```bash
python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase plan
```
**PROOF**: Paste the verification output.
**STOP**: Do NOT proceed to Tasks until verification passes.

### Step 0c: Tasks
```bash
/spec-kitty.tasks
```
**PROOF**: Paste output confirming tasks.md and WP files were generated.

Then verify:
```bash
python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase tasks
```
**PROOF**: Paste the verification output.
**STOP**: Do NOT proceed to Implementation until verification passes.

---

## 1. Start a Work Package (WP)

### Step 1a: Create worktree
```bash
spec-kitty agent workflow implement --task-id <WP-ID> --agent "<AGENT-NAME>"
```
**PROOF**: Paste the output. Extract the worktree path from it.

If output is truncated or unclear:
```bash
git worktree list
```
**CRITICAL**: Do NOT guess the path. Verify it exists before proceeding.

### Step 1b: Update kanban
```bash
python3 .kittify/scripts/tasks/tasks_cli.py update <FEATURE> <WP-ID> doing \
  --agent "<AGENT-NAME>" --note "Starting implementation"
```
**PROOF**: Paste the CLI output confirming lane change.

Then verify the board:
```bash
/spec-kitty.status
```
**PROOF**: Paste the kanban board. Confirm your WP shows in "doing" lane.
**STOP**: Do NOT start coding until the kanban shows the WP in "doing".

---

## 2. Implementation Loop

1. **Navigate**: `cd .worktrees/<WP-ID>` — verify with `pwd`
2. **Setup**: Install dependencies if needed
3. **Code**: Implement the feature
4. **Test**: Run tests or manual verification
5. **Commit**: `git add . && git commit -m "feat(<WP>): description"` (local worktree)

---

## 3. Review & Handover

### Step 3a: Verify clean state
```bash
python3 tools/orchestrator/verify_workflow_state.py --wp <WP-ID> --phase review
```
**PROOF**: Paste the output. Must show "Worktree is clean".
**STOP**: Do NOT proceed if there are uncommitted changes.

### Step 3b: Update kanban to for_review
```bash
python3 .kittify/scripts/tasks/tasks_cli.py update <FEATURE> <WP-ID> for_review \
  --agent "<AGENT-NAME>" --note "Implementation complete, ready for review"
```
**PROOF**: Paste the CLI output.

### Step 3c: Verify kanban updated
```bash
/spec-kitty.status
```
**PROOF**: Paste the board. WP must show in "for_review" lane.

### Step 3d: Sync specs in main repo
```bash
cd <PROJECT_ROOT>
git add kitty-specs
git commit -m "docs(specs): mark <WP-ID> complete"
```

---

## 4. Merge & Cleanup

### Step 4a: Accept
```bash
spec-kitty accept
```

### Step 4b: Merge
```bash
spec-kitty merge
```
If this fails, use the manual fallback:
```bash
git merge <WP-BRANCH-NAME>
git worktree remove .worktrees/<WP-FOLDER>
git branch -d <WP-BRANCH-NAME>
```

### Step 4c: Update kanban to done
```bash
python3 .kittify/scripts/tasks/tasks_cli.py update <FEATURE> <WP-ID> done \
  --agent "<AGENT-NAME>" --note "Merged and cleaned up"
```
**PROOF**: Paste CLI output + final `/spec-kitty.status` board.

---

## 5. Dual-Loop Mode (Protocol 133)

When Spec Kitty runs inside a Dual-Loop session, roles are split:

| Step | Who | Action |
|------|-----|--------|
| Specify/Plan/Tasks | **Outer Loop** (Antigravity) | Generates all artifacts |
| Implement | **Outer Loop** creates worktree, then **Inner Loop** (Claude) codes | Inner Loop receives Strategy Packet |
| Review/Merge | **Outer Loop** | Verifies output, commits, merges |

**Inner Loop constraints**:
- No git commands — Outer Loop owns version control
- Scope limited to the Strategy Packet — no exploratory changes
- If worktree is inaccessible, may implement on feature branch (fallback — log in friction log)

**Cross-reference**: [dual-loop-supervisor SKILL](../dual-loop-supervisor/SKILL.md) | [Protocol 133 workflow](../../workflows/sanctuary_protocols/dual-loop-learning.md)

---

## 6. Task Management CLI

The tasks CLI manages WP lane transitions. **Always use this instead of manually editing frontmatter or checkboxes.**

```bash
# List WPs and their lanes
python3 .kittify/scripts/tasks/tasks_cli.py list <FEATURE-SLUG>

# Move a WP between lanes (planned → doing → for_review → done)
python3 .kittify/scripts/tasks/tasks_cli.py update <FEATURE-SLUG> <WP-ID> <LANE> \
  --agent "<AGENT-NAME>" --note "reason"

# Append activity log entry without changing lane
python3 .kittify/scripts/tasks/tasks_cli.py history <FEATURE-SLUG> <WP-ID> \
  --note "what happened" --agent "<AGENT-NAME>"

# Roll back to previous lane
python3 .kittify/scripts/tasks/tasks_cli.py rollback <FEATURE-SLUG> <WP-ID>

# Check feature acceptance readiness
python3 .kittify/scripts/tasks/tasks_cli.py status --feature <FEATURE-SLUG>
```

**Valid lanes**: `planned`, `doing`, `for_review`, `done`

**Dashboard**: `/spec-kitty.dashboard` reads lane data from WP frontmatter.

---

## Common Issues

- **"Base workspace not found"**: WP depends on a merged WP. Create worktree off `main`:
  ```bash
  git worktree add .worktrees/<WP-FOLDER> main
  cd .worktrees/<WP-FOLDER>
  git checkout -b <WP-BRANCH-NAME>
  ```
- **"Already on main"**: Merge commands must run from project root, not inside a worktree.
- **Kanban not updating**: Verify you're using the CLI, not manually editing frontmatter.
