# Security Vulnerability Scanning Guide

## Overview

This guide covers how to scan for security vulnerabilities in your dependencies using GitHub CLI and integrate security scanning into your shift-left development process.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Scanning with GitHub CLI](#scanning-with-github-cli)
- [Understanding Dependabot Alerts](#understanding-dependabot-alerts)
- [Shift-Left Security Integration](#shift-left-security-integration)
- [Local Security Scanning](#local-security-scanning)
- [Automated Workflows](#automated-workflows)
- [Best Practices](#best-practices)

## Prerequisites

### Install GitHub CLI

```bash
# macOS
brew install gh

# Authenticate with GitHub
gh auth login
```

### Required Permissions

Ensure your GitHub token has the following scopes:
- `repo` - Full control of private repositories
- `read:org` - Read org and team membership
- `workflow` - Update GitHub Action workflows

## Scanning with GitHub CLI

### Check Authentication Status

```bash
gh auth status
```

### View All Dependabot Alerts

```bash
# List all alerts
gh api repos/OWNER/REPO/dependabot/alerts

# Pretty formatted output
gh api repos/richfrem/ingPoC/dependabot/alerts \
  --jq '.[] | {
    number: .number,
    severity: .security_advisory.severity,
    package: .dependency.package.name,
    summary: .security_advisory.summary,
    patched_version: .security_advisory.vulnerabilities[0].first_patched_version.identifier
  }'
```

### Filter by Severity

```bash
# High severity only
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts \
  --jq '.[] | select(.security_advisory.severity == "high") | {
    package: .dependency.package.name,
    summary: .security_advisory.summary
  }'

# Critical and High severity
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts \
  --jq '.[] | select(.security_advisory.severity == "critical" or .security_advisory.severity == "high")'
```

### Count Open Alerts

```bash
# Total open alerts
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts \
  --jq '[.[] | select(.state == "open")] | length'

# By severity
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts \
  --jq 'group_by(.security_advisory.severity) | map({severity: .[0].security_advisory.severity, count: length})'
```

### Get Detailed Alert Information

```bash
# Get specific alert details
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts/ALERT_NUMBER

# Get fix recommendations
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts \
  --jq '.[] | {
    package: .dependency.package.name,
    current_version: .dependency.package.version,
    patched_version: .security_advisory.vulnerabilities[0].first_patched_version.identifier,
    cvss_score: .security_advisory.cvss.score
  }'
```

## Understanding Dependabot Alerts

### Alert Severity Levels

- **Critical**: Immediate action required (CVSS 9.0-10.0)
- **High**: Should be addressed quickly (CVSS 7.0-8.9)
- **Medium**: Address in normal development cycle (CVSS 4.0-6.9)
- **Low**: Address when convenient (CVSS 0.1-3.9)

### Alert States

- **open**: Vulnerability is present and unresolved
- **dismissed**: Manually dismissed by a user
- **fixed**: Dependency has been updated to a non-vulnerable version

## Shift-Left Security Integration

### Why Shift-Left Security?

Shift-left security means integrating security checks **earlier** in the development process:

✅ **Benefits:**
- Catch vulnerabilities before they reach production
- Reduce cost of fixes (cheaper to fix in development)
- Faster feedback loop for developers
- Prevent vulnerable code from being committed

❌ **Without Shift-Left:**
- Vulnerabilities discovered in production
- Emergency patches and hotfixes
- Potential security incidents
- Higher remediation costs

### Pre-Commit Security Checks

Yes, you **should** integrate security scanning into your pre-commit process! Here's how:

## Local Security Scanning

### 1. NPM Audit (Built-in)

```bash
# Run npm audit
npm audit

# Get JSON output
npm audit --json

# Fix automatically (use with caution)
npm audit fix

# Fix only production dependencies
npm audit fix --production-only

# Dry run to see what would be fixed
npm audit fix --dry-run
```

### 2. Create a Pre-Commit Security Check

Add to your `package.json`:

```json
{
  "scripts": {
    "security:check": "npm audit --audit-level=high",
    "security:fix": "npm audit fix",
    "precommit:security": "npm audit --audit-level=critical --production"
  }
}
```

### 3. Integrate with Husky/lint-staged

Update your `.husky/pre-commit` or create one:

```bash
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

# Run security audit before commit
echo "🔒 Running security audit..."
npm audit --audit-level=high --production

if [ $? -ne 0 ]; then
  echo "❌ Security vulnerabilities found! Please fix before committing."
  echo "Run 'npm audit' for details or 'npm audit fix' to attempt automatic fixes."
  exit 1
fi

# Continue with other pre-commit checks
npx lint-staged
```

### 4. Alternative: Use Snyk CLI

Snyk provides more comprehensive scanning:

```bash
# Install Snyk
npm install -g snyk

# Authenticate
snyk auth

# Test for vulnerabilities
snyk test

# Monitor project (sends results to Snyk dashboard)
snyk monitor

# Test and fail on high severity
snyk test --severity-threshold=high
```

### 5. GitHub CLI Pre-Push Check

Create a script to check before pushing:

```bash
#!/bin/bash
# .git/hooks/pre-push or scripts/pre-push-security.sh

echo "🔍 Checking for Dependabot alerts..."

OPEN_ALERTS=$(gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts \
  --jq '[.[] | select(.state == "open" and (.security_advisory.severity == "critical" or .security_advisory.severity == "high"))] | length')

if [ "$OPEN_ALERTS" -gt 0 ]; then
  echo "⚠️  Warning: $OPEN_ALERTS critical/high severity alerts found in GitHub!"
  echo "Run: gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts --jq '.[] | select(.state == \"open\")' for details"

  read -p "Continue with push? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi
```

## Automated Workflows

### Recommended Shift-Left Security Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Development Workflow                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Local Development                                        │
│     └─> npm audit (manual check)                            │
│                                                               │
│  2. Pre-Commit Hook                                          │
│     └─> npm audit --audit-level=high                        │
│     └─> Fail on critical/high vulnerabilities               │
│                                                               │
│  3. Pre-Push Hook (Optional)                                 │
│     └─> Check GitHub Dependabot alerts via CLI              │
│     └─> Warn on open critical/high alerts                   │
│                                                               │
│  4. CI/CD Pipeline                                           │
│     └─> npm audit in GitHub Actions                         │
│     └─> Dependabot auto-updates                             │
│     └─> SAST/DAST scanning                                  │
│                                                               │
│  5. Production Monitoring                                    │
│     └─> Continuous Dependabot monitoring                    │
│     └─> Security alerts via GitHub                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Add to package.json Scripts

```json
{
  "scripts": {
    "security:audit": "npm audit",
    "security:audit:ci": "npm audit --audit-level=moderate --production",
    "security:fix": "npm audit fix",
    "security:check:github": "gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts --jq '.[] | select(.state == \"open\")'",
    "precommit": "npm run security:audit && lint-staged"
  }
}
```

### GitHub Actions Workflow

Create `.github/workflows/security-scan.yml`:

```yaml
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  security-audit:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run npm audit
        run: npm audit --audit-level=moderate
        continue-on-error: true

      - name: Check Dependabot alerts
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh api repos/${{ github.repository }}/dependabot/alerts \
            --jq '.[] | select(.state == "open") | {severity: .security_advisory.severity, package: .dependency.package.name}'
```

## Best Practices

### 1. **Regular Scanning**
- Run `npm audit` before every commit
- Check Dependabot alerts weekly
- Review security advisories for your dependencies

### 2. **Prioritize Fixes**
- **Critical/High**: Fix immediately
- **Medium**: Fix within sprint
- **Low**: Fix during maintenance windows

### 3. **Keep Dependencies Updated**
```bash
# Check for outdated packages
npm outdated

# Update to latest within semver range
npm update

# Update to latest (breaking changes possible)
npm install package@latest
```

### 4. **Use Dependabot Auto-Updates**

Enable in `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "your-team"
    labels:
      - "dependencies"
      - "security"
```

### 5. **Monitor Production**
- Enable GitHub security alerts
- Set up Slack/email notifications for new vulnerabilities
- Use GitHub Security Advisory Database

### 6. **Document Exceptions**
If you must dismiss an alert:
```bash
# Dismiss with reason
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts/ALERT_NUMBER \
  -X PATCH \
  -f state=dismissed \
  -f dismissed_reason=no_bandwidth \
  -f dismissed_comment="Will address in Q2 security sprint"
```

## Quick Reference Commands

```bash
# Check auth
gh auth status

# List all open alerts
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts --jq '.[] | select(.state == "open")'

# Count by severity
gh api repos/richfrem/quantum-diamond-forge/dependabot/alerts --jq 'group_by(.security_advisory.severity) | map({severity: .[0].security_advisory.severity, count: length})'

# Local audit
npm audit

# Fix vulnerabilities
npm audit fix

# Production-only audit
npm audit --production

# Fail on high severity
npm audit --audit-level=high
```

## Recommended Pre-Commit Setup

**Balanced Approach** (Recommended for this project):

```bash
# Add to package.json
"scripts": {
  "precommit:security": "npm audit --audit-level=high --production"
}
```

**Why this approach?**
- ✅ Catches critical and high severity issues
- ✅ Only checks production dependencies (dev deps less critical)
- ✅ Fast enough for pre-commit hook
- ✅ Prevents vulnerable code from being committed
- ❌ Won't block on medium/low severity issues

## Conclusion

**Should you scan before pushing?**

**Yes!** Implement a **layered security approach**:

1. **Pre-commit**: Fast `npm audit` for critical/high severity
2. **Pre-push**: Optional GitHub Dependabot check (can be slow)
3. **CI/CD**: Comprehensive scanning in GitHub Actions
4. **Continuous**: Dependabot monitoring and auto-PRs

This shift-left approach catches vulnerabilities early, reduces security debt, and maintains a secure codebase without significantly slowing down development.

---

**Last Updated**: 2025-11-21
**Maintained By**: Development Team
