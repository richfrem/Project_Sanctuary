---
description: Run a multi-persona audit loop (Red Team → Architect → QA)
argument-hint: "< bundle.md"
---

# Audit Loop

Run an iterative multi-persona analysis: Security → Architecture → QA.

## Recommended Loop Order
```bash
# 1. Red Team (find vulnerabilities)
cat ${CLAUDE_PLUGIN_ROOT}/personas/security/security-auditor.md \
  | claude -p "ACT AS THE SECURITY AUDITOR. Do NOT use tools. Do NOT search filesystem." \
  < bundle.md > audit_01_security.md

# 2. Architecture Review (validate design)
cat ${CLAUDE_PLUGIN_ROOT}/personas/quality-testing/architect-review.md \
  | claude -p "ACT AS THE ARCHITECT REVIEWER. Focus on complexity, patterns, scalability. Do NOT use tools." \
  < bundle.md > audit_02_architecture.md

# 3. QA Expert (find edge cases)
cat ${CLAUDE_PLUGIN_ROOT}/personas/quality-testing/qa-expert.md \
  | claude -p "ACT AS THE QA EXPERT. Focus on testability and edge cases. Do NOT use tools." \
  < bundle.md > audit_03_qa.md
```

## Best Practice
Run the **Architect loop AFTER the Red Team loop** to ensure security fixes didn't introduce accidental complexity.
