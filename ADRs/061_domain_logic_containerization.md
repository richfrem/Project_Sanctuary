# ADR 061: Domain Logic Containerization (Sanctuary Domain)

**Status:** Accepted
**Date:** 2025-12-19
**Context:** ADR 060 (Fleet of 7) defined infrastructure containers (Git, Filesystem, Network) but failed to account for Python-based business logic. The IBM Gateway is a router, not a runtime; it cannot execute local Python scripts directly.
**Decision:** We will deploy a 6th active container, `sanctuary-domain`.
**Technical Specs:**
* **Name:** `sanctuary-domain`
* **Port:** `8105`
* **Base:** Python 3.12 (Standard)
* **Role:** Aggregates Chronicle, Protocol, Task, and ADR tools into a single MCP server process.
**Consequences:**
* **Positive:** Preserves all existing business logic during the migration.
* **Negative:** Adds one more container to the resource footprint (~150MB).
