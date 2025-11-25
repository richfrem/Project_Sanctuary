# Git Workflow Quick Reference

This guide provides recommended git workflows and shortcuts for **Project Sanctuary**.

## TL;DR - Recommended Setup

```bash
# 1. Add these aliases to your ~/.gitconfig
git config --global alias.st "status -sb"
git config --global alias.aa "add --all"
git config --global alias.cm "commit -m"
git config --global alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

# 2. Use Protocol 101 commit workflow (see Council Orchestrator section below)

# 3. Let pre-commit hooks validate your changes (Protocol 101 + secret detection)
```

## Conventional Commit Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

| Type | When to Use | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(auth): add OAuth2 login` |
| `fix` | Bug fix | `fix(api): handle null user response` |
| `docs` | Documentation only | `docs(readme): update setup instructions` |
| `style` | Code formatting (no logic change) | `style(components): fix indentation` |
| `refactor` | Code restructuring | `refactor(utils): extract validation logic` |
| `test` | Adding/updating tests | `test(api): add integration tests` |
| `chore` | Maintenance tasks | `chore(deps): update dependencies` |
| `ci` | CI/CD changes | `ci(github): add CodeQL workflow` |
| `perf` | Performance improvements | `perf(db): optimize query performance` |
| `revert` | Revert previous commit | `revert: revert feat(auth) commit` |

## Common Workflows

### 1. Feature Development (Standard)

```bash
# Create feature branch
git checkout -b feature/add-security-scanning

# Make changes, then stage specific files
git add .github/dependabot.yml
git add .github/workflows/codeql.yml
git add docs/ci-cd/README.md

# Review what you're about to commit
git diff --cached

# Commit with conventional format
git commit -m "feat(security): configure GitHub Advanced Security

- Add Dependabot for dependency scanning
- Add CodeQL for security analysis
- Update CI/CD docs with security guide

Refs: TASK-0067, ADR-040"

# Push to remote
git push origin feature/add-security-scanning

# Create PR on GitHub
# After PR approval, merge via GitHub UI
```

### 2. Quick Fix (Using Aliases)

```bash
# Fix a typo in documentation
git aa  # Stage all changes
git cm "docs(readme): fix typo in installation steps"
git push
```

### 3. Multi-file Changes (Interactive Staging)

```bash
# Stage specific lines from files
git add -p

# Review staged changes
git diff --cached

# Commit
git commit -m "refactor(api): extract error handling logic"

# Push
git push
```

### 4. Amend Last Commit

```bash
# Forgot to add a file to last commit
git add forgotten-file.js
git commit --amend --no-edit

# Or change the commit message
git commit --amend -m "feat(auth): add OAuth2 login (updated message)"

# Force push (only if not yet merged!)
git push --force-with-lease
```

## Useful Git Aliases

Add these to your `~/.gitconfig`:

```gitconfig
[alias]
    # Quick status
    st = status -sb

    # Stage all changes
    aa = add --all

    # Commit with message
    cm = commit -m

    # Amend last commit
    amend = commit --amend --no-edit

    # Pretty log
    lg = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit

    # Show staged changes
    staged = diff --cached

    # Undo last commit (keep changes)
    undo = reset HEAD~1

    # List branches sorted by last modified
    branches = branch --sort=-committerdate

    # Show files in last commit
    last = show --name-only
```

## Pre-commit Hook Usage (Protocol 101)

Our pre-commit hook (`.git/hooks/pre-commit`) enforces **Protocol 101: The Doctrine of the Unbreakable Commit**:
- ✅ Requires `commit_manifest.json` with SHA-256 hashes
- ✅ Verifies file integrity against manifest
- ✅ No `.env` files (except `.env.example`)
- ✅ No hardcoded secrets (API keys, tokens, passwords)
- ✅ Blocks commit if violations found

**Recommended Commit Workflow:**

```bash
# Option A: Use Council Orchestrator (recommended)
cat > command.json << 'EOF'
{
  "command_type": "git_operations",
  "git_operations": {
    "files_to_add": ["path/to/file.py"],
    "commit_message": "feat(component): add new feature",
    "push_after_commit": false
  }
}
EOF
python3 council_orchestrator/app/main.py command.json

# Option B: Manual manifest generation
python3 tools/generate_manifest.py path/to/file.py
git add path/to/file.py commit_manifest.json
git commit -m "feat(component): add new feature"

# Option C: Emergency bypass (Guardian approval required)
git commit --no-verify -m "Emergency: critical fix"
```

**⚠️ WARNING:** Never use `--no-verify` except for emergencies with Guardian approval!

## Branch Naming Conventions

```
feature/short-description    # New features
fix/bug-description          # Bug fixes
docs/documentation-update    # Documentation
refactor/code-improvement    # Code refactoring
test/add-tests              # Test additions
chore/maintenance-task      # Maintenance
```

**Examples:**
- `feature/github-security-scanning`
- `fix/null-pointer-in-auth`
- `docs/update-ci-cd-guide`
- `refactor/extract-validation-logic`

## Commit Message Examples

### Good Commit Messages ✅

```bash
# Feature with detailed body
git commit -m "feat(security): add Dependabot and CodeQL workflows

- Configure Dependabot for npm and GitHub Actions
- Add CodeQL workflow for JavaScript/TypeScript analysis
- Update CI/CD documentation with security scanning guide

This implements the security scanning layer documented in ADR-040.

Refs: TASK-0067, ADR-040"

# Bug fix with issue reference
git commit -m "fix(auth): handle null user response from Supabase

Fixes #123"

# Documentation update
git commit -m "docs(ci-cd): add security scanning interpretation guide"

# Dependency update
git commit -m "chore(deps): bump axios from 0.21.1 to 1.6.0

Fixes CVE-2023-45857 (High severity)"
```

### Bad Commit Messages ❌

```bash
# Too vague
git commit -m "fix stuff"
git commit -m "update files"
git commit -m "changes"

# No type prefix
git commit -m "added security scanning"

# Too long subject line (>72 chars)
git commit -m "feat(security): add Dependabot and CodeQL workflows for automated dependency scanning and security analysis"
```

## IDE Git Integration

### Visual Studio Code

1. **Stage files:** Click `+` next to file in Source Control panel
2. **Review changes:** Click file to see diff
3. **Commit:** Type message in input box, press `Ctrl+Enter`
4. **Push:** Click `...` → Push

**Recommended extensions:**
- GitLens - Enhanced git capabilities
- Git Graph - Visualize branch history

### JetBrains IDEs (WebStorm, IntelliJ)

1. **Commit:** `Ctrl+K` (Windows/Linux) or `Cmd+K` (Mac)
2. **Review changes:** Check boxes for files to stage
3. **Commit message:** Type in message box
4. **Commit and Push:** Click dropdown → Commit and Push

## Troubleshooting

### Pre-commit hook not running

```bash
# Make hook executable
chmod +x .githooks/pre-commit

# Verify git hooks path
git config core.hooksPath .githooks
```

### Accidentally committed secret

```bash
# 1. IMMEDIATELY revoke the secret in the service provider
# 2. Remove from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Force push (⚠️ coordinate with team!)
git push origin --force --all

# 4. Update environment variables with new secret
```

### Merge conflict

```bash
# 1. Pull latest changes
git pull origin main

# 2. Resolve conflicts in your editor
# Look for <<<<<<< HEAD markers

# 3. Stage resolved files
git add resolved-file.js

# 4. Complete merge
git commit -m "merge: resolve conflicts with main"

# 5. Push
git push
```

## Best Practices

1. **Commit often** - Small, focused commits are easier to review and revert
2. **Write clear messages** - Future you will thank present you
3. **Review before committing** - Always run `git diff --cached`
4. **Test before pushing** - Run `pytest` and linting checks
5. **Use Protocol 101** - Generate commit manifests via Council Orchestrator
6. **Pull before pushing** - Avoid merge conflicts
7. **Use branches** - Never commit directly to `main`
8. **Keep commits atomic** - One logical change per commit

## References

- [Protocol 101: The Doctrine of the Unbreakable Commit](../../ADRs/019_protocol_101_unbreakable_commit.md)
- [Council Orchestrator GitOps Documentation](../../council_orchestrator/docs/howto-commit-command.md)
- [Git Safety Rules](.agent/git_safety_rules.md)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
