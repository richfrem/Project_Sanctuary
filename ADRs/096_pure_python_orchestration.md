# ADR-096: Pure Python Orchestration

## Status
Deferred (Pending ADR-036 validation)

## Context
ADR-036 (v2) introduced a "Thick Python / Thin Shim" architecture where `.sh` files acted as dumb wrappers around `cli.py workflow start`. 
While this solved the fragility of Bash logic, it retained "Triple Tracking" (Markdown -> Bash -> Python) as a legacy artifact of the pilot.

## Decision
We will remove the Shim Layer entirely.

1.  **Delete** all `scripts/bash/codify-*.sh` and `scripts/bash/sanctuary-start.sh`.
2.  **Update** all 27+ `.agent/workflows/*.md` files to invoke the Python CLI directly:
    *   Old: `source scripts/bash/sanctuary-start.sh ...`
    *   New: `python tools/cli.py workflow start ...`

> [!IMPORTANT]
> **Single Entry Point**: The `--name` argument to the Python CLI determines which workflow template is used. There is ONE orchestration command, not one per workflow type. See ADR-036 Anti-Patterns section.

## Consequences

### Positive
*   **Simplicity**: Eliminates an entire file layer (`.sh`).
*   **Truth**: The Workflow Markdown points directly to the logic executor (`cli.py`).
*   **Platform Agnostic**: Removes dependency on `source` (Bash) semantics, easier for PowerShell/Windows adoption if needed.

### Negative
*   **Migration Cost**: Requires updating 27+ files.
*   **Verbosity**: `python tools/cli.py workflow start` is longer than `source script.sh`.
