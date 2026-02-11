---
name: Spec Kitty Workflow
description: Standard operating procedures for the Spec Kitty agentic workflow (Plan -> Implement -> Review -> Merge).
---

# Spec Kitty Workflow

This skill documents the standard lifecycle for implementing features using Spec Kitty.

## 1. Start a Work Package (WP)
Always start by creating an isolated worktree for the task.

```bash
# syntax: spec-kitty implement <WP-ID>
spec-kitty implement WP06
```

**Note**: If the previous WP was merged to main, use `--base main` if the tool doesn't auto-detect it.
```bash
spec-kitty implement WP06 --base main
```

## 2. Implementation Loop
1.  **Navigate**: `cd .worktrees/<WP-ID>`
2.  **Install**: `npm install` (backend/frontend)
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
