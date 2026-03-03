# ADR-096: Pure Python Orchestration

## Status
Superseded by **Pure Plugin Architecture** (2026-03-03). `tools/cli.py` is now fully deprecated.
This ADR documents a transitional step; the current architecture uses Python scripts inside `plugins/` directly.

## Context
ADR-036 (v2) introduced a "Thick Python / Thin Shim" architecture where `.sh` files acted as dumb wrappers around `cli.py workflow start`. 
While this solved the fragility of Bash logic, it retained "Triple Tracking" (Markdown -> Bash -> Python) as a legacy artifact of the pilot.

## Decision
We will remove the Shim Layer entirely.

1.  **Delete** all `scripts/bash/codify-*.sh` and `scripts/bash/sanctuary-start.sh`.
2.  **[HISTORICAL]** Workflows were updated to invoke the Python CLI directly:
    *   Old: `source scripts/bash/sanctuary-start.sh ...`
    *   Intermediate: `python tools/cli.py workflow start ...` [NOW ALSO DEPRECATED]
    *   **Current**: Use plugin scripts in `plugins/sanctuary-guardian/` via slash commands (`/sanctuary-*`)

> [!IMPORTANT]
> **[SUPERSEDED]** `tools/cli.py` is now deprecated. The single entry point is the plugin ecosystem under `plugins/sanctuary-guardian/`. Use `/sanctuary-*` slash commands which map to plugin scripts.

## Consequences

### Positive
*   **Simplicity**: Eliminates an entire file layer (`.sh`).
*   **Truth**: The Workflow Markdown points directly to the logic executor (`cli.py`).
*   **Platform Agnostic**: Removes dependency on `source` (Bash) semantics, easier for PowerShell/Windows adoption if needed.

### Negative
*   **Migration Cost**: Requires updating 27+ files.
*   **Verbosity**: `python tools/cli.py workflow start` is longer than `source script.sh`.
