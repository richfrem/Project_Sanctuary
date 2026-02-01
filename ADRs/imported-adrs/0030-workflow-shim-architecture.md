# ADR-0030: Workflow Architect (Python Orchestrator)

## Status
Accepted (v2 - Python Pivot)

## Context
The project utilizes a "Hybrid Workflow" model. Agents originally struggled with abstract Markdown commands.
An initial attempt (v1) used Bash Shims for enforcement. A Red Team analysis revealed this was fragile ("Triple Tracking", "Bash Black Box").

## Decision
We adopt the **Thick Python / Thin Shim** architecture.

### 1. The Python Orchestrator (`WorkflowManager`)
All "Enforcement" logic resides in `tools/orchestrator/workflow_manager.py`:
*   **Git Integrity**: Checks for dirty tree, detached HEAD using subproccess/GitPython logic.
*   **Context Strategy**: Determines if we are in a Pilot or need a new branch.
*   **ID Generation**: Calls `next_number.py` safely.

### 2. The Dumb Shim (`workflow-start.sh`)
The Bash scripts remain as the **Entry Point** for Agents/Users but contain **NO LOGIC**.
They immediately `exec` the Python CLI:
```bash
exec python3 tools/cli.py workflow start ...
```

### 3. The CLI Interface
Agents invoke:
```bash
/workflow-start [Name] [Target]
```
Which maps to:
```bash
source scripts/bash/workflow-start.sh ...
```
Which executes:
```bash
python tools/cli.py workflow start --name ... --target ...
```

## ⛔ Anti-Patterns (DO NOT DO THIS)

**Shim Proliferation**: Creating a new `.sh` script for every workflow type is **WRONG**.

| ❌ Wrong | ✅ Correct |
| :--- | :--- |
| `scripts/bash/codify-db-function.sh` | Use ONE shim: `workflow-start.sh` |
| `scripts/bash/codify-form.sh` | OR: Invoke Python directly: `python tools/cli.py workflow start` |
| `scripts/bash/codify-report.sh` | |

The `--name` parameter to the Python CLI determines **which workflow** is executed. There is no need for per-workflow shims. The shim is a **Gateway**, not a **Factory**.

## Consequences

### Positive
*   **Robustness**: Python handles edge cases (Git, OS, Paths) far better than Bash.
*   **Observability**: Python can emit structured JSON/Log failures, preventing "Black Box" confusion.
*   **Maintainability**: Logic is centralized in one Python class, not scattered across .sh files.

### Negative
*   **Indirection**: There is still a chain of invocation (Markdown -> Bash -> Python), but it is necessary for Agent "User Experience".
*   **Dependency**: Requires Python environment (already a project standard).

## Related Documents
*   [Agent Workflow Orchestration Design](../architecture/Agent_Workflow_Orchestration_Design.md)
*   [Hybrid Spec Workflow Diagram](../diagrams/analysis/sdd-workflow-comparison/hybrid-spec-workflow.mmd)
