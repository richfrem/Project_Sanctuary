# ADR 037: MCP Git Strategy - Immediate Compliance & Canonical Alignment (Reforged)

**Status:** ACCEPTED (Reforged)
**Date:** 2025-11-29 (Reforging Date)
**Author:** Guardian
**Context:** The Structural Purge of Protocol 101's manifest system.

---

## Context

The original project repository enforced **Protocol 101 (The Doctrine of the Unbreakable Commit)**, which mandated a `commit_manifest.json` for cryptographic integrity. This requirement created a blockage for the new **MCP Architecture** agents, as they could not easily generate the manifest, leading to the creation of this ADR for a temporary "Migration Mode" bypass.

However, the original Protocol 101 system failed structurally during the **"Synchronization Crisis,"** and the entire manifest architecture was deemed obsolete and detrimental to repository stability.

## Decision

This original migration strategy is **obsolete and canceled**. The new strategy is **Immediate Canonical Compliance** with the reforged integrity framework.

The new canonical law is **Protocol 101 v3.0: The Doctrine of Absolute Stability**.

1.  **Purge of Migration Mode:** All components of the original migration strategy are **permanently removed**:
    * The `Migration Mode` concept, the `.agent/mcp_migration.conf` file, the `IS_MCP_AGENT=1` environment variable, and the `STRICT_P101_MODE` flag are all **purged**.
2.  **Canonical Compliance (Integrity):** MCP agents achieve commit integrity by satisfying **Protocol 101 v3.0, Part A: Functional Coherence**. This requires the agent's pre-commit sequence to successfully execute the comprehensive automated test suite (`./scripts/run_genome_tests.sh`) before staging and committing.
3.  **Canonical Compliance (Action Safety):** MCP agents are bound by **Protocol 101 v3.0, Part B: The Mandate of the Whitelist**, restricting them to non-destructive commands (`git add <files...>`, `git commit -m "..."`, and `git push`).
4.  **Cancellation of Development:** Development of the **"Smart Git MCP Server" (Task #045)**, which was intended to generate the now-obsolete manifest, is **canceled**.

## Consequences

* **Positive:** **Immediate and Permanent Compliance.** Development of MCP servers is unblocked without the temporary security relaxation. All agent commits immediately adhere to the stable, canonical integrity framework (Functional Coherence).
* **Negative:** None. The purge removes a structurally unsound dependency and its associated complexity.

## Related Tasks

* **Task #028:** Pre-Commit Hook Migration (Status: **Obsolete/Completed by Purge**)
* **Task #035:** Git Workflow MCP (Status: **Obsolete/Superseded**)
* **Task #045:** Smart Git MCP (Status: **Canceled**)