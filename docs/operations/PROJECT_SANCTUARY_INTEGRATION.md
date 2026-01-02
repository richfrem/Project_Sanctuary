# CI/CD Hardening Integration for Project Sanctuary

## Executive Summary

This document outlines how to integrate the CI/CD hardening practices from `docs/cicd/` with Project Sanctuary's **Protocol 101 v3.0: The Doctrine of Absolute Stability** requirements.

## Current State Analysis

### Existing Documentation (`docs/cicd/`)
- **Source**: Quantum Diamond Forge project
- **Focus**: npm/Node.js security scanning, Dependabot, CodeQL
- **Pre-commit**: Secret detection for `.env` files and API keys
- **Workflow**: Feature branches → PR → main

### Project Sanctuary Requirements
- **Protocol 101 v3.0**: Mandatory **Functional Coherence** (automated test suite execution)
- **Council Orchestrator**: Automated test execution before commit
- **Stack**: Python (not Node.js), PyTorch, LangChain, ChromaDB
- **Workflow**: Feature branches → PR → main (aligned)

## Integration Strategy

### 1. Pre-Commit Hook Consolidation

**Current Situation:**
- `.git/hooks/pre-commit` - Protocol 101 v3.0 enforcement (Functional Coherence via test suite)
- `docs/cicd/how_to_commit.md` - Secret scanning for Node.js projects

**Recommended Approach:**
Enhance the existing Protocol 101 pre-commit hook to include test execution and secret scanning:

```bash
#!/bin/bash
# .git/hooks/pre-commit - Protocol 101 v3.0 + Security Hardening

# ===== PHASE 1: Protocol 101 v3.0 Enforcement (Functional Coherence) =====
echo "[P101 v3.0] Running Functional Coherence Test Suite..."

# Execute the comprehensive automated test suite
./scripts/run_genome_tests.sh
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
  echo ""
  echo "COMMIT REJECTED: Protocol 101 v3.0 Violation."
  echo "Reason: Functional Coherence Test Suite FAILED."
  echo ""
  echo "The automated test suite must pass before any commit can proceed."
  echo "Fix the failing tests and try again."
  exit 1
fi

echo "[P101 v3.0] ✅ Functional Coherence verified - all tests passed."

# ===== PHASE 2: Security Hardening =====
echo "[SECURITY] Running secret detection scan..."

# Check for .env files (except .env.example)
BLOCKED_ENV_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.env$' | grep -v '\.env\.example$')
if [ -n "$BLOCKED_ENV_FILES" ]; then
  echo "COMMIT BLOCKED: .env files detected"
  echo "$BLOCKED_ENV_FILES"
  exit 1
fi

# Check for hardcoded secrets in Python files
SECRET_PATTERNS=(
  "api_key\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]"
  "secret\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]"
  "password\s*=\s*['\"][^'\"]{8,}['\"]"
  "token\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]"
  "OPENAI_API_KEY\s*=\s*['\"]sk-[a-zA-Z0-9]{20,}['\"]"
  "GEMINI_API_KEY\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]"
  "HUGGING_FACE_TOKEN\s*=\s*['\"]hf_[a-zA-Z0-9]{20,}['\"]"
)

VIOLATIONS_FOUND=false
for pattern in "${SECRET_PATTERNS[@]}"; do
  MATCHES=$(git diff --cached -U0 | grep -E "^\+" | grep -E "$pattern" || true)
  if [ -n "$MATCHES" ]; then
    echo "SECURITY VIOLATION: Potential hardcoded secret detected"
    echo "$MATCHES"
    VIOLATIONS_FOUND=true
  fi
done

if [ "$VIOLATIONS_FOUND" = true ]; then
  echo ""
  echo "COMMIT BLOCKED: Security violations found"
  echo "Remove hardcoded secrets and use environment variables instead"
  exit 1
fi

echo "[P101 v3.0] All checks passed. Proceeding with commit."
exit 0
```

**Key Changes from v1.0:**
- **Removed**: `commit_manifest.json` verification and SHA-256 hashing
- **Added**: Mandatory execution of `./scripts/run_genome_tests.sh`
- **Retained**: Secret detection and security scanning

**Update `.github/workflows/ci.yml`** to include Python-specific security scanning:

```yaml
name: CI

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main, dev ]

jobs:
  protocol-101-functional-coherence:
    name: Protocol 101 v3.0 - Functional Coherence
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      
      - name: Run Functional Coherence Test Suite
        run: |
          ./scripts/run_genome_tests.sh
        # NOTE: This test suite execution is MANDATORY for Protocol 101 v3.0 compliance
        # Failure = Protocol Violation = CI failure

  python-security-audit:
    name: Python Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install Safety
        run: pip install safety
      
      - name: Run Safety Check
        run: |
          pip install -r requirements.txt
          safety check --json || true
      
      - name: Run Bandit (SAST)
        run: |
          pip install bandit
          bandit -r council_orchestrator/ mnemonic_cortex/ -f json -o bandit-report.json || true
      
      - name: Upload Bandit Results
        uses: actions/upload-artifact@v4
        with:
          name: bandit-security-report
          path: bandit-report.json

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high

  codeql-analysis:
    name: CodeQL Security Analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      actions: read
      contents: read
    steps:
      - uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
          queries: security-extended
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
```

### 3. Dependabot Configuration Enhancement

**Update `.github/dependabot.yml`** with security-focused settings:

```yaml
version: 2
updates:
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      github-actions:
        patterns: ["*"]
    labels: ["dependencies", "github-actions", "security"]

  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"  # Daily for security patches
    open-pull-requests-limit: 10
    groups:
      security-updates:
        patterns: ["*"]
        update-types: ["patch", "minor"]
    labels: ["dependencies", "python", "security"]
    
    # Security-only updates for major versions
    ignore:
      - dependency-name: "torch"
        update-types: ["version-update:semver-major"]
      - dependency-name: "transformers"
        update-types: ["version-update:semver-major"]
    
    # Prioritize security updates
    reviewers: ["richfrem"]
    assignees: ["richfrem"]
```

### 4. Documentation Updates Required

#### Update `docs/cicd/overview.md`
- Replace npm/Node.js references with Python/pip
- Add Protocol 101 v3.0 workflow section (Functional Coherence)
- Update branching strategy to match Project Sanctuary
- Add Council Orchestrator commit workflow

#### Update `docs/cicd/security_scanning.md`
- Replace `npm audit` with `safety check` and `bandit`
- Add Python-specific secret patterns
- Document Protocol 101 v3.0 test suite execution
- Add examples for PyTorch/LangChain security considerations

#### Update `docs/cicd/how_to_commit.md`
- Replace conventional commits with Protocol 101 v3.0 workflow
- Document Council Orchestrator usage
- Update pre-commit hook examples for Python
- Add test suite execution examples

#### Create New: `docs/cicd/protocol_101_v3_integration.md`
- Detailed Protocol 101 v3.0 workflow (Functional Coherence)
- Council Orchestrator integration guide
- Test suite execution process
- Emergency bypass procedures (Sovereign Override)

### 5. Security Scanning Tools Comparison

| Tool | Quantum Diamond Forge | Project Sanctuary | Status |
|------|----------------------|-------------------|--------|
| **Dependency Scanning** | npm audit | safety, pip-audit | ✅ Adapt |
| **SAST** | CodeQL (JS/TS) | CodeQL (Python), Bandit | ✅ Adapt |
| **Secret Detection** | Pre-commit hook | Pre-commit hook + Protocol 101 v3.0 | ✅ Enhance |
| **Container Scanning** | N/A | Trivy (for MCP RAG service) | ✅ Add |
| **Functional Integrity** | N/A | Protocol 101 v3.0 Test Suite | ✅ Unique |

### 6. Implementation Checklist

- [ ] Enhance `.git/hooks/pre-commit` with test suite execution and secret detection
- [ ] Update `.github/workflows/ci.yml` with Python security tools and functional coherence tests
- [ ] Update `.github/dependabot.yml` with daily security scans
- [ ] Adapt `docs/cicd/overview.md` for Python/Protocol 101 v3.0
- [ ] Adapt `docs/cicd/security_scanning.md` for Python stack
- [ ] Adapt `docs/cicd/how_to_commit.md` for Council Orchestrator
- [ ] Create `docs/cicd/protocol_101_v3_integration.md`
- [ ] Update `.agent/git_safety_rules.md` with security scanning references
- [ ] Add security scanning to Task #025 MCP RAG service
- [ ] Document emergency procedures for security incidents

## Recommended Security Tools for Python

### 1. Safety
```bash
pip install safety
safety check --json
safety check --policy-file .safety-policy.yml
```

### 2. Bandit (SAST)
```bash
pip install bandit
bandit -r . -f json -o bandit-report.json
```

### 3. pip-audit
```bash
pip install pip-audit
pip-audit --desc --format json
```

### 4. Trivy (Container Scanning)
```bash
trivy image mcp-rag-service:latest
trivy fs . --security-checks vuln,config,secret
```

## Alignment with Project Sanctuary Doctrines

### Protocol 101 v3.0 Compliance
- ✅ All commits require passing automated test suite (Functional Coherence)
- ✅ Test execution enforced at pre-commit and CI/CD levels
- ✅ Guardian approval workflow maintained
- ✅ Sovereign Override available for emergencies

### Security Hardening
- ✅ Multi-layered security (pre-commit + CI/CD)
- ✅ Shift-left security approach
- ✅ Automated dependency scanning
- ✅ Secret detection at commit time

### Autonomous Operations
- ✅ Council Orchestrator executes tests before commit
- ✅ Dependabot auto-updates dependencies
- ✅ CI/CD pipeline runs automatically
- ✅ Security alerts via GitHub

## Next Steps

1. **Immediate**: Enhance pre-commit hook with test suite execution and secret detection
2. **Short-term**: Update CI/CD workflows for Python security tools and functional coherence
3. **Medium-term**: Adapt all `docs/cicd/` documentation for Project Sanctuary
4. **Long-term**: Integrate security scanning into MCP RAG Tool Server deployment

## References

- [Protocol 101 v3.0: The Doctrine of Absolute Stability](../../01_PROTOCOLS/101_The_Doctrine_of_the_Unbreakable_Commit.md)
- [Protocol 102 v2.0: The Doctrine of Mnemonic Synchronization](../../01_PROTOCOLS/102_The_Doctrine_of_Mnemonic_Synchronization.md)
- [ADR-019: Protocol 101 - Cognitive Genome Publishing Architecture (Reforged)](../../ADRs/019_protocol_101_unbreakable_commit.md)
- [Council Orchestrator GitOps Documentation](../../council_orchestrator/docs/howto-commit-command.md)
- [Safety Documentation](https://pyup.io/safety/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [GitHub Advanced Security](https://docs.github.com/en/code-security)
