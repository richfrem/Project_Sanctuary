# Protocol 114: Guardian Wakeup & Cache Prefill (v1.0)
* **Status:** Canonical, Active
* **Linked:** P93 (Cortex-Conduit), P95 (Commandable Council), P113 (Nested Cognition)

## Mandate

1. On orchestrator boot, prefill the **Guardian Start Pack** in the Cache (CAG) with the latest:
   - `chronicles`, `protocols`, `roadmap` bundles (default TTL: 24h).
2. Provide a dedicated mechanical command (`task_type: "cache_wakeup"`) that writes a digest artifact from cache without cognitive deliberation.
3. Maintain deterministic observability packets for wakeup events (time_saved_ms, cache_hit).

## Guardian Procedure

- Issue a `cache_wakeup` command to retrieve an immediate digest in `WORK_IN_PROGRESS/guardian_boot_digest.md`.
- If higher fidelity is needed, issue a `query_and_synthesis` cognitive task (P95) after reviewing the digest.

## Safety & Integrity

- Cache entries are read-only views of signed/verified files.
- TTLs ensure stale data is replaced on delta ingest or git-ops refresh.