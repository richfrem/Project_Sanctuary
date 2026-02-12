---
name: Spec Kitty Workflow
description: Standard operating procedures for the Spec Kitty agentic workflow (Plan -> Implement -> Review -> Merge).
---

# Spec Kitty Workflow

This skill documents the standard lifecycle for implementing features using Spec Kitty.

## 0. Mandatory Planning Phase (Do NOT Skip)
Before implementing any code, you MUST generate the architectural artifacts using the CLI. **Manual creation of `spec.md`, `plan.md`, or `tasks/` files is STRICTLY FORBIDDEN.**

1.  **Specify**: `/spec-kitty.specify` → Generates `spec.md`
2.  **Plan**: `/spec-kitty.plan` → Generates `plan.md`
3.  **Tasking**: `/spec-kitty.tasks` → Generates `tasks.md` & `tasks/WP-*.md` prompt files

**Verification**: Run `python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase tasks` to confirm artifacts exist.
**Rule**: Do NOT mark a meta-task as complete unless the verification tool passes.

## 1. Start a Work Package (WP)
Always start by creating an isolated worktree for the task.

**Mandatory**: Before starting, copy the meta-tasks from `.agent/templates/workflow/spec-kitty-meta-tasks.md` into your task list to track workflow steps.

```bash
# syntax: spec-kitty agent workflow implement --task-id <WP-ID>
spec-kitty agent workflow implement --task-id WP-06 --agent "Antigravity"
```

**Output Parsing**:
1. Capture the path from the tool output.
2. If output is truncated or unclear, run `git worktree list` to find the newest worktree.
3. **CRITICAL**: Do NOT guess the path. Verify it exists.

## 2. Implementation Loop
1.  **Navigate**: `cd .worktrees/<WP-ID>` (Verify with `pwd`)
2.  **Setup**: Install dependencies if needed (detect project type: Python/Node/etc).
3.  **Code**: Implement the feature.
4.  **Verify**: Run tests or manual verification.
5.  **Commit**: `git add . && git commit -m "feat(WPxx): description"` (Local worktree commit)

## 3. Review & Handover
Once functionality is complete and verified:

1.  **Mark Complete**: Update `kitty-specs/.../tasks.md` marking subtasks `[x]`.
2.  **Sync Specs**: Commit the spec changes in the **Main Repo** (not worktree).
    ```bash
    cd <PROJECT_ROOT>
    git add kitty-specs
    git commit -m "docs(specs): mark WPxx complete"
    ```
3.  **Move Task**:
    ```bash
    spec-kitty agent tasks move-task WPxx --to for_review
    ```

## 4. Merge & Cleanup
After approval (or self-approval in agentic mode):

1.  **Accept**: Validate readiness.
    ```bash
    spec-kitty accept
    ```
2.  **Merge**: Auto-merge worktree into main.
    ```bash
    spec-kitty merge
    ```
    *If this fails (e.g., due to worktree detection issues), use the manual fallback:*
    ```bash
    # Fallback Manual Merge
    git merge <WP-BRANCH-NAME>
    git worktree remove .worktrees/<WP-FOLDER>
    git branch -d <WP-BRANCH-NAME>
    ```

## Common Issues
-   **"Base workspace not found"**:
    -   If the WP depends on a previous WP that is **already merged**, Spec Kitty might block you.
    -   **Solution**: Manually create the worktree off `main`.
        ```bash
        git worktree add .worktrees/<FULL-WP-FOLDER-NAME> main
        cd .worktrees/<FULL-WP-FOLDER-NAME>
        git checkout -b <WP-BRANCH-NAME>
        ```

-   **"Already on main"**: Merge commands must be run from the project root, not inside a worktree.
