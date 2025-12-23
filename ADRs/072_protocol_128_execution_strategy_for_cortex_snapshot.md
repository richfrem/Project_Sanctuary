# Protocol 128 Execution Strategy for Cortex Snapshot

**Status:** PROPOSED
**Date:** 2025-12-23
**Author:** Antigravity


---

## Context

The `cortex_capture_snapshot` tool is a critical component of Protocol 128 (Cognitive Continuity), responsible for generating `audit` and `seal` packets. The implementation relies on `scripts/capture_code_snapshot.py`, a mature Node.js utility that handles file traversal, `.gitignore` parsing, token counting, and complex "Awakening Seed" generation.

The `sanctuary_cortex` service, which hosts this tool, is deployed as a Docker container based on `python:3.11`.
**Problem:** The container environment currently lacks the Node.js runtime required to execute the snapshot script. This creates an "Environment Impedance Mismatch" where the Python service cannot successfuly invoke its dependency.

## Decision

We need to formally select an execution strategy to reconcile the Python Service / Node Script mismatch.

**Option A: Hybrid Runtime (Recommended for Velocity)**
Update `mcp_servers/gateway/clusters/sanctuary_cortex/Dockerfile` to install `nodejs` and `npm`. This allows the Python service to shell out (`subprocess.run`) to the existing, proven JS script.

**Option B: Native Python Port (Recommended for Purity)**
Rewrite the logic of `capture_code_snapshot.py` into a native Python module (`mcp_servers.rag_cortex.utils.snapshot_engine`). This eliminates the Node dependency but requires significant porting effort, especially for the legacy "Forging" and argument parsing logic.

**Option C: Sidecar / Service**
Deploy the snapshot tool as a standalone Node.js MCP server or sidecar container. This is deemed likely excessive for a file-system utility.

## Consequences

**Option A (Hybrid):**
*   **Positive:** Immediate enablement of verifying Protocol 128; zero regression risk for the snapshot logic.
*   **Negative:** Increases Docker image size (~50-100MB); introduces polyglot maintenance burden in a single container.

**Option B (Port):**
*   **Positive:** Homogeneous Python environment; better error handling integration with Cortex.
*   **Negative:** Significant development effort (estimated 1-2 days) to port complex "Awakening" and "Token counting" logic; strict parity testing required.

**Option C (Sidecar):**
*   **Positive:** Strict isolation of runtimes.
*   **Negative:** Disproportionate infrastructure complexity for a localized file-system utility.
