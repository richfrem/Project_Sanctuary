# Standard Spec-Kitty Workflow

> **Visual Guide:** [Workflow Diagram](../diagrams/standard-spec-kitty-workflow.mmd)

This document outlines the **Correct** Spec-Kitty lifecycle. Unlike the "Incremental Safety Protocol" (which merges per WP), the Standard Workflow is designed to handle multiple Work Packages in parallel or sequence, performing a **Batch Merge** only when the entire feature is complete.

## The Process Flow

### 1. Feature Definition (Root)
- **Start:** Define the feature.
- **Commands:**
  - `spec-kitty specify` (Creates Spec)
  - `spec-kitty plan` (Creates Plan)
  - `spec-kitty tasks` (Generates WPs)

### 2. WP Execution Loop (Worktree Isolated)
For **EACH** Work Package (WP01, WP02, ...):

1.  **Initialize:**
    - Command: `spec-kitty implement WP-xx` (Use `--base main` if parent branch is missing)
    - *System Actions:* Creates isolated worktree (`.worktrees/WP-xx`) and branch (`WP-xx`).
    - *User Action:* **MUST** `cd .worktrees/WP-xx` immediately.

2.  **Implement:**
    - Write code, test, and verify **inside the worktree**.
    - **NO** changes to Root/Main.

3.  **Commit (Local):**
    - `git add .`
    - `git commit -m "feat(WP-xx): ..."`
    - *Constraint:* Must commit to local feature branch before moving task.

3b. **Backup (Recommended):**
    - Command: `git push origin WP-xx`
    - *Purpose:* Safely backs up work to remote without merging.

4.  **Submit for Review:**
    - Command: `spec-kitty agent tasks move-task WP-xx --to for_review`
    - *System Actions:* Updates `tasks.md`, signals readiness.

5.  **Review:**
    - Command: `spec-kitty review WP-xx`
    - *System Actions:* Verifies dependencies, moves task to `done`.

### 3. Feature Completion (Root)
Once **ALL** WPs are in the `done` column:

1.  **Acceptance:**
    - Command: `spec-kitty accept`
    - *System Actions:* Checks all tasks are done, verifies spec requirements.

2.  **The Big Merge (Automated):**
    - **Context:** `Main Repo Root` (Directory: `InvestmentToolkit/`)
    - **Command:** `spec-kitty merge`
    - *System Actions:*
        - **Detects:** All feature worktrees.
        - **Merges:** Sequentially merges `WP-01`, `WP-02`... to **LOCAL** `main`.
        - **Cleans:** Auto-removes all `.worktrees/WP-xx` directories and local `WP-xx` branches.
        - **Push (Optional):** Pushes to `origin` **only** if `--push` flag used.

## Key Differences from Safety Protocol
| Feature | Standard Spec-Kitty | Incremental Safety |
| :--- | :--- | :--- |
| **Merge Timing** | **End of Feature** (All WPs) | **End of WP** (Immediate) |
| **Worktree Life** | Persists until Feature Done | Deleted after WP Done |
| **Branching** | Multiple Feature Branches exist simultaneously | Single Feature Branch exists briefly |
| **Git Command** | `spec-kitty merge` (Automated) | `git merge --squash` (Manual) |

## Why this is Difficult for Agents
Agents often struggle here because:
1.  **Context Switching:** They forget to `cd` into the worktree (Phase 2).
2.  **Persistence:** They lose track of the "Feature" scope while focusing on a single "WP".
3.  **Cleanup:** They try to manually delete worktrees before `spec-kitty merge` runs, breaking the automation.
4.  **Committing to Main:** They make doc/task updates directly to main instead of in the worktree, causing divergence.
5.  **Naming Confusion:** Inconsistent worktree/branch naming makes it hard to track locations.

## ⛔ CRITICAL: What NEVER to Do

| ❌ WRONG | ✅ CORRECT |
|----------|------------|
| `git commit` while in `/InvestmentToolkit/` | `git commit` while in `/InvestmentToolkit/.worktrees/WP-xx/` |
| `git push origin main` | `git push origin <feature-branch>` (e.g., `002-feature-WP07`) |
| Editing `kitty-specs/` from main repo | Editing code files in worktree; `kitty-specs/` is auto-managed |
| Manual worktree deletion | `spec-kitty merge` handles cleanup |
| Pushing directly to protected main | Create feature branch → PR → merge |

## ⚠️ Branch Protection Workflow

If `origin/main` has branch protection enabled:

1. **Merge locally:** `spec-kitty merge` merges WP branches into LOCAL main
2. **Create feature branch:** `git checkout -b feature/<feature-slug>-final`
3. **Push feature branch:** `git push origin feature/<feature-slug>-final`
4. **Create PR:** Merge feature branch → main via GitHub PR
5. **Cleanup:** After PR merged, run worktree cleanup manually

## Manual Merge Fallback

If `spec-kitty merge` fails (e.g., missing worktrees for earlier WPs):

```bash
# 1. Reset local main to origin
git checkout main
git fetch origin main
git reset --hard origin/main

# 2. Merge the WP branch directly
git merge <WP-branch-name> -m "feat: merge WP## - description"

# 3. Create feature branch for PR
git checkout -b feature/<slug>-final
git push origin feature/<slug>-final

# 4. Create and merge PR on GitHub
```

## Reference

See [Agent Worktree Reference](./agent-worktree-reference.md) for naming conventions and common mistakes.
