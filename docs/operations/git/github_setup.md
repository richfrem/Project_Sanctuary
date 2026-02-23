# GitHub Repository Configuration Guide

This guide documents how to configure the **Project Sanctuary** GitHub repository to enable CI/CD pipelines, security scanning, and automated workflows.

## Prerequisites

- Admin access to the GitHub repository
- Repository: `https://github.com/richfrem/Project_Sanctuary`

## Best Practices for AI-Assisted Development

**⚠️ Important for AI Coding Assistants (Antigravity, Cursor, etc.):**

When working with CI/CD pipelines that include security scans (CodeQL, Dependabot, Trivy), follow these practices:

1. **Use Protocol 101 workflow** - Commit via Council Orchestrator to auto-generate manifests
2. **Batch commits locally** - Make multiple commits on your feature branch before pushing
3. **Push once when ready** - Only push when the feature is complete and tested locally
4. **Use draft PRs** - Mark PRs as "Draft" while still working
5. **Avoid rapid push cycles** - Security scans can take 2-3 minutes per run

**Why:** Protocol 101 verification + security scans are resource-intensive. Pushing every small change creates unnecessary CI runs.

**Recommended workflow:**
```bash
# Use Council Orchestrator for Protocol 101 compliance
cat > command.json << 'EOF'
{
  "command_type": "git_operations",
  "git_operations": {
    "files_to_add": ["file1.py", "file2.md"],
    "commit_message": "feat: add feature",
    "push_after_commit": false
  }
}
EOF
python3 council_orchestrator/app/main.py command.json

# Push once when ready
git push origin feature/my-feature

# Create PR (mark as draft if still WIP)
gh pr create --draft --title "WIP: My Feature"
```

## Step 1: Enable GitHub Actions (done)

GitHub Actions should be enabled by default, but verify:

1. Go to **Settings** → **Actions** → **General**
2. Under "Actions permissions", select:
   - ✅ **Allow all actions and reusable workflows**
3. Under "Workflow permissions", select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
4. Click **Save**

## Step 2: Enable Security Features

1. Go to **Settings** → **Code security and analysis** (Sidebar under "Security").
2. Under the **Advanced Security** section, **Enable** the following:
   - **Dependency graph** (Should be enabled by default)
   - **Dependabot alerts**
   - **Dependabot security updates**
     - *Optional:* Enable **Grouped security updates** to reduce noise.
   - **Secret Protection** -> **Push protection** (Block commits that contain supported secrets).
   - **Private vulnerability reporting** (Optional).

## Step 3: Configure CodeQL Analysis

**Eligibility:**
- **Public repositories:** Free for everyone.
- **Private repositories:** Requires GitHub Advanced Security (GHAS) license.

**Setup Instructions:**
1. Still in **Code security and analysis**, scroll down to **Code scanning** / **CodeQL analysis**.
2. Click **Set up** (or "Configure").
3. Choose **Default** setup (Recommended).
   - GitHub will automatically detect languages (Python).
   - It will create a dynamic workflow without you needing to commit a YAML file.
   - Click **Enable CodeQL**.

*(If "Default" is not available, use the existing `.github/workflows/ci.yml` which includes CodeQL for Python).*

## Step 4: Create Development Branch

Before setting up branch protection, create a `dev` branch for integration testing:

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Create dev branch from main
git checkout -b dev
git push -u origin dev

# Return to your working branch
git checkout -
```

## Step 5: Configure Branch Protection Rules

### 5.1 Protect the `main` Branch

1. Go to **Settings** → **Branches**
2. Click **Add branch protection rule**
3. **Branch name pattern:** `main`
4. Enable:
   - ✅ **Require a pull request before merging**
   - ❌ **Require approvals** - UNCHECK (not needed for solo dev, check for teams)
   - ✅ **Require status checks to pass before merging**
     - ✅ **Require branches to be up to date before merging**
     - **Add required status checks:**
       - `Protocol 101 Manifest Verification` (from CI Pipeline)
       - `Python Linting` (from CI Pipeline)
       - `Test Council Orchestrator` (from CI Pipeline)
       - `Security Scanning` (from CI Pipeline)
   - ✅ **Require conversation resolution before merging** (optional but good practice)
   - ✅ **Do not allow bypassing the above settings**
5. Click **Create**

**Result:** All changes to `main` must:
- Come from `dev` via PR
- Pass CI pipeline (linting, tests)

### 5.2 Protect the `dev` Branch

1. Click **Add branch protection rule** again
2. **Branch name pattern:** `dev`
3. Enable:
   - ✅ **Require a pull request before merging** (forces PR from feature branches)
   - ❌ **Require approvals** - UNCHECK (allows you to merge your own PRs)
   - ✅ **Require status checks to pass before merging**
     - ✅ **Require branches to be up to date before merging**
     - **Add required status checks:**
       - `Protocol 101 Manifest Verification`
       - `Python Linting`
       - `Test Council Orchestrator`
   - ❌ **Do not allow bypassing** - UNCHECK (gives you flexibility on dev)
4. Click **Create**

**Result:** Feature branches must:
- Create PR to `dev` (not directly to `main`)
- Pass CI checks before merging

## Step 6: Configure Notifications

Set up notifications for security alerts:

1. Click on your **profile icon** (top right) → **Settings**
2. In the left sidebar, click **Notifications**
3. Scroll down to the **System** section
4. Enable the following:
   - ✅ **Dependabot alerts: New vulnerabilities** - "When you're given access to Dependabot alerts automatically receive notifications when a new vulnerability is found in one of your dependencies."
   - ✅ **Dependabot alerts: Email digest** - "Email a regular summary of Dependabot alerts for up to 10 of your repositories."
   - ✅ **Security campaign emails** - "Receive email notifications about security campaigns in repositories where you have access to security alerts."

**Result:** You'll now receive email notifications whenever security issues are detected in your repositories.

## Step 7: Verify Everything Works

### 7.1 Test CI Pipeline

```bash
# Create a test branch
git checkout -b test/ci-pipeline

# Make a small change
echo "# Test" >> README.md

# Commit and push
git add README.md
git commit -m "test: verify CI pipeline"
git push origin test/ci-pipeline

# Create a PR on GitHub: test/ci-pipeline -> main
# Verify CI pipeline runs and passes
```

### 7.2 Test Dependabot

Dependabot runs weekly, but you can trigger it manually:

1. Go to **Insights** → **Dependency graph** → **Dependabot**
2. Click **Check for updates**

### 7.3 Test Secret Scanning

If enabled, try pushing a test secret:

```bash
# This should be blocked by Protocol 101 pre-commit hook
echo "OPENAI_API_KEY=sk-test123" > secret.txt
git add secret.txt

# Try to commit (will be blocked - no manifest)
git commit -m "test: secret scanning"
# Blocked by Protocol 101!

# Even if you generate a manifest, secret patterns should be caught
```

## Workflow Files Reference

### `.github/workflows/ci.yml`

**Purpose:** Continuous Integration pipeline

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch

**Jobs:**
1. **Protocol 101 Verification** - Validates commit manifests
2. **ShellCheck** - Lints shell scripts in `plugins/`
3. **Python Linting** - Black and Flake8 checks
4. **Test Council Orchestrator** - Runs pytest for orchestrator
5. **Test Mnemonic Cortex** - Runs pytest for RAG system
6. **Security Scanning** - Trivy vulnerability scanner

### `.github/dependabot.yml`

**Purpose:** Automated dependency updates

**Configuration:**
- **GitHub Actions ecosystem:** Scans workflow files
  - Schedule: Weekly
  - Groups updates
- **Python (pip) ecosystem:** Scans Python dependencies
  - Schedule: Daily (security patches)
  - Groups updates
  - Ignores major version updates for torch/transformers

## Troubleshooting

### Workflows Not Appearing in Actions Tab

**Symptoms:** Actions tab shows "Get started with GitHub Actions" instead of workflows

**Causes:**
1. Workflow files not committed/pushed
2. Workflow files in wrong directory
3. YAML syntax errors
4. GitHub Actions disabled in repo settings

**Solutions:**
```bash
# 1. Verify files are committed
git ls-files .github/workflows/

# 2. Verify files are pushed
git log --oneline --name-only | grep workflows

# 3. Validate YAML syntax
npx js-yaml .github/workflows/ci.yml

# 4. Check repo settings
# Go to Settings → Actions → General → Verify "Allow all actions" is selected
```

## Security Best Practices

1. **Enable all security features:**
   - ✅ Dependabot alerts
   - ✅ Secret scanning
   - ✅ Push protection

2. **Protect main branch:**
   - Require PR reviews
   - Require status checks to pass
   - Prevent force pushes

3. **Use local pre-commit hooks:**
   - Catch secrets before pushing
   - Enforce code quality locally
   - Faster feedback loop

## Related Documentation

- [CI/CD Pipeline Documentation](./overview.md)
- [Git Workflow Guide](./git_workflow.md)
- [How to Commit Guide](./how_to_commit.md)
- [Project Sanctuary Integration Guide](./PROJECT_SANCTUARY_INTEGRATION.md)
- [Protocol 101: The Unbreakable Commit](../../../ADRs/019_protocol_101_unbreakable_commit.md)
- [Council Orchestrator GitOps](../../../ARCHIVE/howto-commit-command.md)

## External Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [Secret Scanning Documentation](https://docs.github.com/en/code-security/secret-scanning)
