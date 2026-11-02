---
id: proc-git-feature-branch
name: Zero-Trust Feature Branch Creation
version: 1.0.0
domain: git
applicability: Creating and verifying an isolated git feature branch adhering to the
  Supreme Law and Zero-Trust Git discipline.
inputs:
- branch_name: Branch name adhering to convention (feat/*, fix/*, refactor/*, docs/*)
prerequisites:
- Local working tree is clean (git status --porcelain is empty)
- Local repository is synced with origin/main
steps:
- '1. Inspect working tree status: git status'
- '2. Checkout and pull main: git checkout main && git pull origin main'
- '3. Branch creation: git checkout -b <branch_name>'
- '4. Validate branch reference: git rev-parse --abbrev-ref HEAD'
success_criteria:
- git rev-parse --abbrev-ref HEAD matches <branch_name>
- git status indicates on branch <branch_name> with clean tree
abort_conditions:
- Working tree has uncommitted unstaged modifications
- Branch name lacks valid prefix (feat/, fix/, refactor/, docs/)
recovery_guidance:
- If working tree is dirty, either commit on existing branch or run git stash before
  branching.
outcomes:
  environment: macOS-darwin
  success_count: 9
  failure_count: 0
  blocked_count: 0
  last_verified: '2026-11-02'
---

# Zero-Trust Feature Branch Creation

## 1. Overview
Under Project Sanctuary's Zero-Trust Git policy, agents must never make direct commits on `main`. All exploratory and deterministic implementations must take place on dedicated feature branches.

## 2. Execution Playbook

```bash
# Step 1: Ensure working tree is clean
git status

# Step 2: Ensure base branch is up to date
git checkout main
git pull origin main

# Step 3: Create the feature branch
git checkout -b <branch_name>

# Step 4: Verify current branch
git branch --show-current
```

## 3. Post-Condition Verification
Verify that `git branch --show-current` outputs the expected branch name and that no commits were made to `main`.
