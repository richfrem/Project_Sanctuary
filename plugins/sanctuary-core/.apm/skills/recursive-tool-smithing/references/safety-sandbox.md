# Mode-Invariant Compliance Gate: Safety Sandbox

> **CRITICAL**: This gate runs on EVERY iteration of the evolution loop.
> It cannot be bypassed by any conditional execution path, tool availability state,
> or user shortcut. Violation = abort the current iteration and log a `crash` entry.

## Permitted Operations (User-Authorized)

| Operation | Scope | Rule |
|-----------|-------|------|
| **File writes** | `plugins/recursive-tool-smithing/` only | All creates/edits within the plugin folder are allowed |
| **Copilot CLI** | `copilot -p "..." < input > output` | GPT 5.0 mini - free, zero tokens. Max 2 concurrent requests. |
| **Git read ops** | Any path | Branch/status/log/diff - always allowed |
| **Git write ops** | Plugin folder only | add + commit after sandbox pass + graduated autonomy check |
| **Timeout** | 5 minutes hard limit per iteration | Enforced by `sandbox_runner.py` |

## Forbidden Operations (Hard Constraints)

| Forbidden | Reason |
|-----------|--------|
| File writes outside `plugins/recursive-tool-smithing/` | Blast radius limit |
| Network calls from generated scripts | No requests/httpx/urllib in toolbox scripts |
| Subprocess in generated scripts | Use plugin utilities only; never inside toolbox scripts |
| More than 2 concurrent Copilot CLI calls | Free tier fair use |
| Secrets/credentials in any script | Exception: Copilot CLI auth handled by OS session only |
| `eval()` / `exec()` with dynamic input | Security absolute |

## Enforcement Checklist (Run Before Every Commit)

Before classifying a result as `keep`, the agent MUST verify:

- [ ] Script path is inside `scripts/toolbox/` OR is an authorized plugin path (skills/, scripts/, agents/)
- [ ] Generated toolbox scripts: no `subprocess`, no dynamic `eval`
- [ ] Generated toolbox scripts: no network imports unless pre-approved
- [ ] Execution completed within 5-minute window
- [ ] No file writes outside authorized paths
- [ ] No secrets or credentials referenced
- [ ] Copilot security review passed (no CRITICAL findings)

## Violation Handling

If ANY constraint is violated:
1. Classify the iteration as `crash`.
2. Log to `evals/results.tsv` with reason `SANDBOX_VIOLATION: <rule>`.
3. Delete the candidate script from `scripts/toolbox/`.
4. Continue from last known good state.
5. Do NOT attempt to "fix" the violation automatically - narrow the next hypothesis instead.
