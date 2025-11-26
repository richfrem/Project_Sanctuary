# How to Commit Changes - Step-by-Step Guide

This guide walks you through committing changes to the quantum-diamond-forge project, including pre-commit hook validation and conventional commit format.

## Prerequisites

- Git configured with hooks path: `git config core.hooksPath .githooks`
- Pre-commit hook is executable: `chmod +x .githooks/pre-commit`

## Standard Commit Workflow

### Step 1: Check Current Status

```bash
# See what files have changed
git status

# See detailed changes
git diff
```

### Step 2: Stage Files

**Option A: Stage specific files (recommended)**
```bash
git add path/to/file1.js
git add path/to/file2.md
git add path/to/file3.yml
```

**Option B: Stage all changes**
```bash
git add .
# or
git add --all
```

**Option C: Interactive staging (stage specific lines)**
```bash
git add -p
# Git will show each change and ask: Stage this hunk [y,n,q,a,d,e,?]?
# y = yes, n = no, q = quit, a = all, d = don't stage, e = edit
```

### Step 3: Review Staged Changes

**Quick summary (recommended):**
```bash
# See list of staged files
git status

# Even shorter
git status -s
```

**Detailed diff (optional):**
```bash
# See detailed changes (can be verbose)
git diff --cached

# Press 'q' to exit the diff view
```

**⚠️ IMPORTANT:** Always review your staged changes before committing!

### Step 4: Commit with Conventional Format

```bash
git commit -m "<type>(<scope>): <subject>

<body>

<footer>"
```

**Commit Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code formatting (no logic change)
- `refactor:` - Code restructuring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks (dependencies, build)
- `ci:` - CI/CD changes
- `perf:` - Performance improvements

**Example:**
```bash
git commit -m "feat(security): configure GitHub Advanced Security

- Add Dependabot for dependency scanning
- Add CodeQL workflow for security analysis
- Update CI/CD documentation with security guide

Refs: TASK-0067, ADR-040"
```

### Step 5: Pre-commit Hook Validation

**What happens automatically:**
1. ✅ Hook runs: `.githooks/pre-commit`
2. ✅ Validates no `.env` files (except `.env.example`)
3. ✅ Scans for hardcoded secrets (API keys, tokens, passwords)
4. ✅ If validation passes → commit succeeds
5. ❌ If violations found → commit blocked

**If commit is blocked:**
```bash
# Example error:
COMMIT BLOCKED: Violations found.
VIOLATION: packages/backend/config.js:12 -> OPENAI_API_KEY=<REDACTED>
Fix by removing secrets or using '<REDACTED>'.

# Fix the issue:
# 1. Remove the hardcoded secret
# 2. Use environment variable instead: process.env.OPENAI_API_KEY
# 3. Try committing again
```

**Bypass hook (ONLY if absolutely necessary):**
```bash
git commit --no-verify -m "your message"
# ⚠️ WARNING: Only use --no-verify if you're certain there are no secrets!
```

### Step 6: Push to Remote

```bash
# Push to current branch
git push

# Push to specific branch
git push origin feature/branch-name

# Push to main
git push origin main
```

## Example: Committing TASK-0067 Security Configuration

```bash
# 1. Check status
git status

# 2. Stage security configuration files
git add .github/dependabot.yml
git add .github/workflows/codeql.yml
git add docs/ci-cd/README.md
git add docs/ci-cd/GIT_WORKFLOW.md
git add docs/ci-cd/HOW_TO_COMMIT.md
git add adrs/040_security_scanning_strategy.md
git add adrs/041_git_workflow_automation.md
git add TASKS/in-progress/008_configure_github_security.md
git add TASKS/backlog/009_enhance_precommit_hooks.md
git add scripts/capture_snapshot.js

# Note: Deleted file (.githooks/pre-commit.sh) will be automatically staged
# when you run 'git add .' or will show in 'git status' as deleted

# 3. Review staged changes (quick summary)
git status

# Or see detailed diff (verbose, press 'q' to exit)
# git diff --cached

# 4. Commit with conventional format
git commit -m "feat(security): configure GitHub Advanced Security

- Add Dependabot for npm and GitHub Actions dependency scanning
- Add CodeQL workflow for JavaScript/TypeScript security analysis
- Update CI/CD documentation with comprehensive security scanning guide
- Create ADR-041 for git workflow automation strategy
- Create git workflow quick reference guide
- Enhance snapshot script to exclude agents/feedback directory
- Remove deprecated pre-commit.sh shell script

Deliverables:
- .github/dependabot.yml (weekly scans, grouped PRs)
- .github/workflows/codeql.yml (security-extended queries)
- docs/ci-cd/README.md (175-line security guide, pre-commit hook docs)
- docs/ci-cd/GIT_WORKFLOW.md (conventional commits, aliases, best practices)
- adrs/041_git_workflow_automation.md (no automated git scripts)
- TASKS/backlog/009_enhance_precommit_hooks.md (future ESLint/Prettier integration)

Refs: TASK-0067, ADR-040, ADR-041"

# 5. Pre-commit hook runs automatically (validates no secrets)

# 6. Push to GitHub
git push origin main
```

## Testing Pre-commit Hook

### Test 1: Verify Hook Blocks Secrets

```bash
# Create a test file with a hardcoded secret
echo "OPENAI_API_KEY=<REDACTED>" > test-secret.txt

# Try to commit (should be BLOCKED)
git add test-secret.txt
git commit -m "test: verify pre-commit hook blocks secrets"

# Expected output:
# COMMIT BLOCKED: Violations found.
# VIOLATION: test-secret.txt:1 -> OPENAI_API_KEY=<REDACTED>
# Fix by removing secrets or using '<REDACTED>'.

# Clean up
git reset HEAD test-secret.txt
rm test-secret.txt
```

### Test 2: Verify Hook Blocks .env Files

```bash
# Create a .env file
echo "DATABASE_URL=postgres://localhost" > .env

# Try to commit (should be BLOCKED)
git add .env
git commit -m "test: verify pre-commit hook blocks .env files"

# Expected output:
# COMMIT BLOCKED: Violations found.
# BLOCKED .env file: .env
# Fix by removing secrets or using '<REDACTED>'.

# Clean up
git reset HEAD .env
rm .env
```

### Test 3: Verify Hook Allows Safe Code

```bash
# Create a safe file with environment variable reference
echo "const apiKey = process.env.OPENAI_API_KEY;" > test-safe.js

# Commit (should SUCCEED)
git add test-safe.js
git commit -m "test: verify pre-commit hook allows safe code"

# Expected: Commit succeeds (no violations)

# Clean up
git reset HEAD~1  # Undo last commit
rm test-safe.js
```

## Common Issues and Solutions

### Issue 1: Pre-commit Hook Not Running

**Symptoms:** Commits succeed without validation

**Solution:**
```bash
# Verify hooks path is configured
git config core.hooksPath
# Should output: .githooks

# If not set, configure it
git config core.hooksPath .githooks

# Make hook executable
chmod +x .githooks/pre-commit

# Verify hook exists
ls -la .githooks/pre-commit
```

### Issue 2: Hook Blocks Legitimate Code

**Symptoms:** Hook blocks code that uses environment variables

**Example:**
```javascript
// This might be flagged if not properly formatted
const key = API_KEY;  // ❌ Flagged (looks like hardcoded value)

// Use these patterns instead:
const key = process.env.API_KEY;  // ✅ Safe
const key = import.meta.env.VITE_API_KEY;  // ✅ Safe
const key = config.apiKey;  // ✅ Safe
```

**Solution:** Use whitelisted patterns (see `.githooks/pre-commit` for full list)

### Issue 3: Accidentally Committed Secret

**⚠️ CRITICAL - Act Immediately:**

```bash
# 1. IMMEDIATELY revoke the secret in the service provider
# (e.g., regenerate API key in OpenAI dashboard)

# 2. Remove from git history (if not yet pushed)
git reset HEAD~1  # Undo last commit
# Fix the file, then commit again

# 3. If already pushed, use git filter-branch
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file" \
  --prune-empty --tag-name-filter cat -- --all

# 4. Force push (⚠️ coordinate with team!)
git push origin --force --all

# 5. Update environment variables with new secret
```

## Git Aliases (Optional Shortcuts)

Add these to `~/.gitconfig` for faster workflows:

```gitconfig
[alias]
    # Quick status
    st = status -sb

    # Stage all changes
    aa = add --all

    # Commit with message
    cm = commit -m

    # Show staged changes
    staged = diff --cached

    # Amend last commit
    amend = commit --amend --no-edit

    # Pretty log
    lg = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
```

**Usage:**
```bash
git st              # Instead of: git status -sb
git aa              # Instead of: git add --all
git staged          # Instead of: git diff --cached
git cm "fix: typo"  # Instead of: git commit -m "fix: typo"
```

## Best Practices

1. **Commit often** - Small, focused commits are easier to review and revert
2. **Write clear messages** - Use conventional commit format
3. **Review before committing** - Always run `git diff --cached`
4. **Test locally** - Run `npm run lint` and `npm run test:unit` before committing
5. **Never bypass hooks** - Only use `--no-verify` in emergencies
6. **Keep commits atomic** - One logical change per commit
7. **Reference tasks/issues** - Include `Refs: TASK-XXX` in commit body

## After Pushing to GitHub

Once you push, the following automated checks will run:

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Linting
   - Unit tests
   - Frontend build

2. **CodeQL Analysis** (`.github/workflows/codeql.yml`)
   - Security vulnerability scanning
   - Results in Security tab

3. **Dependabot** (`.github/dependabot.yml`)
   - Dependency vulnerability scanning
   - Automatic PRs for updates

4. **Secret Scanning** (if enabled)
   - Detects committed secrets
   - Alerts in Security tab

Check the **Actions** tab and **Security** tab on GitHub to verify all checks pass.

## References

- [Git Workflow Quick Reference](./git_workflow.md)
- [CI/CD Pipeline Documentation](./overview.md)
- [ADR-041: Git Workflow Automation](../../docs/adr/041_git_workflow_automation.md)
- [Conventional Commits Specification](https://www.conventionalcommits.org/)
