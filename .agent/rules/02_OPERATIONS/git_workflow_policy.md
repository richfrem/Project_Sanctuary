---
trigger: always_on
---

# Git Workflow Policy

## Overview

This policy defines the Git branching strategy and workflow for Project Sanctuary.

## 1. Branch Protection: No Direct Commits to Main

**Rule**: Never commit directly to the `main` branch.

**Before starting work:**
```bash
# Check current branch
git branch

# If on main, create feature branch
git checkout -b feat/your-feature-name
```

**Branch naming conventions:**
- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

## 2. Feature Branch Workflow

**Recommended**: Focus on one feature at a time for clarity and easier code review.

**Why**: Multiple concurrent branches can lead to:
- Merge conflicts
- Context switching overhead
- Difficulty tracking what's in progress

## 3. Feature Development Lifecycle

### Starting a Feature
```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feat/your-feature-name
```

### During Development
```bash
# Make changes, test locally

# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new workflow component"

# Push to remote
git push origin feat/your-feature-name
```

### Completing a Feature
```bash
# 1. Create pull request on GitHub/GitLab
# 2. Wait for code review and approval
# 3. Merge via PR interface
# 4. Clean up locally
git checkout main
git pull origin main
git branch -d feat/your-feature-name

# 5. Clean up remote (if not auto-deleted)
git push origin --delete feat/your-feature-name
```

## 4. Commit Message Standards

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

**Examples:**
```
feat: add new CLI command for vector search

fix: resolve navigation issue in workflow

docs: update README with new architecture diagrams

refactor: extract common validation to shared utility
```

## 5. Handling Conflicts

If `main` has moved ahead while you're working:

```bash
# On your feature branch
git fetch origin
git merge origin/main

# Resolve conflicts if any
# Test to ensure everything still works

git add .
git commit -m "merge: resolve conflicts with main"
git push origin feat/your-feature-name
```

## 6. Best Practices

- **Commit Often**: Small, logical commits are easier to review
- **Pull Frequently**: Stay up to date with `main` to avoid large conflicts
- **Test Before Push**: Ensure your code works locally
- **Descriptive Messages**: Future you (and reviewers) will thank you
- **Clean History**: Consider squashing commits before merging if there are many tiny fixes