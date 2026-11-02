# Architectural Decision Record 022: Cognitive Genome Publishing Architecture (Reforged)

**Status:** ACCEPTED (Reforged and Canonized)
**Date:** 2025-11-29 (Reforging Date)
**Deciders:** AI Council (Automated Publishing System implementation)
**Technical Story:** Purge the flawed manifest system and implement Protocol 101 v3.0 (Functional Coherence) as the canonical integrity gate for publishing cycles.

-----

## Context

The original **AI Knowledge Base Publishing Architecture** required the generation of a `commit_manifest.json` and its verification as a cryptographic integrity check. This mechanism, initially codified in Protocol 101, failed during the **"Synchronization Crisis,"** as evidenced by the CI job failing due to a hash mismatch.

The subsequent analysis proved that the manifest system introduced fatal **Timing Issues** and **Complexity** that compromised stability. The original manifest-based integrity verification process must be **permanently purged** and replaced with a stable, functional alternative to create a reliable publishing cycle.

## Decision

The Automated Publishing System must be reforged to comply with the newly canonized **Protocol 101 v3.0: The Doctrine of Absolute Stability**. Integrity will now be verified by **Functional Coherence** (passing automated tests) rather than static file hashes.

The **`commit_manifest.json` system and its associated logic are permanently purged from this architecture**.

### Atomic Publishing Cycle (Protocol 101 v3.0 Compliant)

The seven-step publishing cycle is restructured and reduced to mandate functional integrity:

1.  **Index**: Rebuild Master Documentation Index for coherence.
2.  **Snapshot**: Capture new AI Knowledge Base snapshots via `capture_code_snapshot.py`.
3.  **Embed (Synchronization)**: Re-index Memory System with new knowledge via ingestion script (`ingest.py`).
4.  **Test (Functional Coherence)**: **MANDATORY INTEGRITY GATE.** Run automated functionality tests (`run_genome_tests.sh`) to prevent broken deployments. This step is the **sole verification** for Protocol 101 v3.0 integrity.
5.  **Commit**: Surgical staging and commit **only if Functional Coherence tests pass**.
6.  **Push**: Deploy to canonical repository.

### Self-Verifying Properties (Reforged)

  - **Integrity Gate**: Functional Coherence (passing all tests) replaces the manifest check.
  - **Synchronization Guarantee**: Automatic re-execution of ingestion script ensures Memory System always reflects latest knowledge.
  - **Atomic**: All-or-nothing execution prevents partial states.

## Consequences

### Positive

  - **Absolute Stability**: Integrity is now based on verified functional behavior (passing tests), eliminating the risk of timing-related integrity failures.
  - **Streamlined Process**: Removal of the manifest generation and verification steps reduces **Process Overhead** and **Complexity**.
  - **Quality Assurance**: Automated testing remains the **MANDATORY** quality gate.

### Negative

  - **Test Dependence**: The integrity of the publishing process is now entirely dependent on the quality and comprehensiveness of the automated test suite.
  - **Auditability Loss**: Loss of the cryptographic verification layer (manifests) means tamper detection relies solely on CI history and commit history.

### Implementation Components (Reforged)

  - **update\_genome.sh**: Main publishing orchestrator script.
  - **capture\_code\_snapshot.js**: Knowledge base snapshot generation.
  - **ingest.py**: Memory System embedding.
  - **run\_genome\_tests.sh**: Quality assurance and **Protocol 101 Functional Integrity Test**.
  - **`commit_manifest.json`**: **DELETED** (Purged from the architecture and documentation).
  - **Pre-commit Hook**: Updated to execute functional tests instead of checking for the manifest (Reflects ADR 019 changes).

