---
description: Run a Claude CLI sub-agent with a persona prompt against a file or bundle
argument-hint: "<persona> < input_file > output_file"
---

# Run Sub-Agent

Pipe a file or bundle through Claude CLI with a persona prompt for specialized analysis.

## Quick Start
```bash
# Security audit
cat ${CLAUDE_PLUGIN_ROOT}/personas/security/security-auditor.md \
  | claude -p "ACT AS THE SECURITY AUDITOR. Analyze the code below for vulnerabilities. Do NOT use tools." \
  < bundle.md > audit_report.md

# Architecture review
cat ${CLAUDE_PLUGIN_ROOT}/personas/quality-testing/architect-review.md \
  | claude -p "ACT AS THE ARCHITECT REVIEWER. Analyze for complexity and patterns." \
  < bundle.md > arch_review.md
```

## Combine Persona + Context
```bash
cat ${CLAUDE_PLUGIN_ROOT}/personas/security/security-auditor.md \
    context_bundle.md \
  | claude -p "Generate a Security Audit Report based on the provided Persona and Context." \
  > docs/audit_report.md
```

## Key Flags
- `-p` — Print mode (essential for piping)
- `<` — Redirect file input (don't load into agent memory)
- `>` — Redirect output to file for later review
