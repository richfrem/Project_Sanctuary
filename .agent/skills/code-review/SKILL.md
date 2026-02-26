---
name: code-review
description: "Multi-perspective code review with confidence scoring. Use when reviewing PRs, auditing code quality, or running /sanctuary-end pre-commit checks. Launches parallel review perspectives (compliance, bugs, history) and filters results by confidence threshold to reduce false positives."
---

# Code Review

Structured code review using multiple perspectives and confidence-based filtering.

## When to Use

- Before committing (`/sanctuary-end` pre-commit)
- PR review requests
- User says "review this code" or "audit these changes"
- Post-implementation quality gate

## Review Perspectives

Launch these review angles independently, then merge findings:

### 1. Policy Compliance
Check against project conventions:
- `.agent/rules/03_TECHNICAL/coding_conventions_policy.md`
- File headers present (Python/JS/C# standards)
- Type hints on function signatures
- Docstrings on non-trivial functions
- Import organization

### 2. Bug Detection
Focus on changes only (not pre-existing issues):
- Unhandled error paths
- Missing null/undefined checks
- Resource leaks (file handles, connections)
- Race conditions in async code
- Off-by-one errors
- Hardcoded secrets or credentials

### 3. Historical Context
Use git blame/log to understand:
- Was this code recently refactored? (fragile area)
- Does the change break established patterns?
- Is this a known problematic area?

### 4. Zero Trust Compliance
Project Sanctuary specific:
- No direct commits to `main`
- No `git push` without explicit approval
- State-changing operations gated by HITL
- No inline Mermaid (ADR 085)

## Confidence Scoring

Rate each finding 0-100:

| Score | Meaning | Action |
|-------|---------|--------|
| 0-25 | Probably false positive | Skip |
| 26-50 | Might be real, minor | Note only |
| 51-79 | Likely real, worth flagging | Include in review |
| **80-100** | **Confident, actionable** | **Must address** |

**Only report findings ≥ 50.** This prevents noise.

## False Positive Filters

Do NOT flag:
- Pre-existing issues not introduced in this change
- Style issues that linters catch
- Pedantic nitpicks
- Code with explicit `# noqa` or suppression comments
- Test fixtures with intentionally "wrong" data

## Output Format

```markdown
## Code Review: [branch/PR name]

**Files reviewed:** N files, M lines changed

### Issues (confidence ≥ 80)
1. **[Category]** Description
   `path/to/file.py:L42` — explanation and suggestion

### Observations (confidence 50-79)
1. **[Category]** Description — worth considering

### Clean Areas
- [List what looks good — positive reinforcement]
```

## Integration with Workflow

```
Implementation → /spec-kitty.review → Code Review → /sanctuary-end
                                           ↑
                                    This skill runs here
```
