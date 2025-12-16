# External Codebase Integration Strategy (Vendoring)

**Decision Matrix Score:** 92% (highest among all options)

**Status:** APPROVED (Post-Red Team Audit)
**Date:** 2025-12-16
**Author:** Antigravity & User

---

## Context
We are integrating **IBM ContextForge** as the foundation for our 13th server, the **MCP Gateway**.
-   **Impact:** This external codebase is **~1.2M tokens** (~1000 files).
-   **The Conflict:** Integrating this codebase increases the total token count by **12x**.
    -   **Strict Option (Protocol 001):** Commit the code (Vendoring). Risk: Flooding the AI's context window.
    -   **Efficient Option (PyPI/Submodule):** Hide the code. Risk: Violating Protocol 001 (Fragile build).

### Technical Target Profile (IBM ContextForge)
To understand the "enemy" we are containing, here is the profile of the target codebase:
-   **Language:** Python (requires pip dependency management).
-   **Structure:** The repo root contains top-level directories `mcpgateway/` (source), `tests/`, `docs/`, `examples/`, and `scripts/`.
-   **Integration Path:** We are mapping the upstream root `.` to our local path `mcp_servers/gateway/`.
-   **Risk Vector:** The `mcpgateway/` folder alone contains ~800k tokens of implementation details that are irrelevant to the daily operation of the Council agents.

## Options Considered

### 1. Git Submodule
- **Mechanism:** Links `mcp_servers/gateway` to a specific commit in the upstream repo.
- **Cons:** Violates Protocol 001 (Unbreakable) because the build is not self-contained. Requires network access to `github.com` during checkout.

### 2. Vendoring (Git Subtree) - **PROPOSED**
- **Mechanism:** Commits the squashed upstream code directly into `mcp_servers/gateway`.
- **Pros:** Fully Unbreakable, Traceable, Patchable.
- **Cons:** Repo Bloat (~1000 files), Noise.

### 3. Package Dependency (PyPI)
- **Mechanism:** `pip install mcp-context-forge`
- **Cons:** Unpatchable, Opacity (code hidden in `site-packages`).

## Decision
We will **Vendor** the IBM ContextForge codebase using **Git Subtree** to satisfy Protocol 001 (Unbreakable Build).

**Justification:** While it increases repository size, it is the *only* option that satisfies **Protocol 001 (Unbreakable)**, **Zero Setup**, and **Patchability** simultaneously.

## Safeguards (Red Team Mandates)
To mitigate the "Context Flood" risk (~1.2M tokens), we have implemented a **Triple-Layer Defense**:

1.  **Layer 1 (The Filter):** `capture_code_snapshot.js` uses **Normalized Path Logic** (converting all separators to `/`) to strictly block `mcpgateway/` and `tests/`. This prevents Windows backslash bugs from bypassing the filter.
2.  **Layer 2 (The Circuit Breaker):** The snapshot script throws a fatal error if file counts in the gateway exceed safety thresholds (>500 files). This prevents accidental ingestion if the folder structure changes.
3.  **Layer 3 (The Silencer):** `.gitattributes` and `.ignore` explicitly mark these files as `linguist-vendored` to prevent diff/search noise and editor lag.

## Update Protocol (Vendor Refresh)
Updates are NOT routine. They are "Vendor Refresh Events."
1.  Check out a new branch `refresh/gateway-upstream`.
2.  Run `git subtree pull --prefix mcp_servers/gateway https://github.com/IBM/mcp-context-forge.git main --squash`.
3.  Manually verify `podman-compose.yml` and patches.
4.  Run `node scripts/capture_code_snapshot.js` to verify the **Circuit Breaker** holds.

## Consequences
**Positive:**
-   **Unbreakable Build:** No dependency on external `git clone` or internet during build/deploy.
-   **Traceability:** Unlike a raw copy-paste, `git subtree` retains metadata allowing for cleaner merges.
-   **Debugging:** Source code is available locally for inspection and logging, unlike a PyPI package.

**Negative:**
-   **Repository Size:** Adds ~1000 files to the repo history.
-   **Noise:** `git status` and search are noisier (Mitigated by Layer 3 Safeguards).

## Implementation Details
-   **Execution Command:** `git subtree add --prefix mcp_servers/gateway https://github.com/IBM/mcp-context-forge.git main --squash`
-   **Version Tracking:** A file `mcp_servers/gateway/VENDOR_INFO.md` will track the original commit hash.
-   **IDE Config:** Vendored directories added to `.vscode/settings.json` (`search.exclude` and `files.watcherExclude`).

## References
-   **Host Repostory (Project Sanctuary):** `https://github.com/richfrem/Project_Sanctuary`
-   **Target Repository (IBM ContextForge):** `https://github.com/IBM/mcp-context-forge`
