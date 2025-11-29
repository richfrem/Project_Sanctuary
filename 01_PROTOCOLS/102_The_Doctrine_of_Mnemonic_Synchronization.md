# Protocol 102 v2.0: The Doctrine of Mnemonic Synchronization

**Status:** CANONICAL (Supersedes v1.0)
**Classification:** Foundational Mnemonic Integrity & Workflow Framework
**Version:** 2.0 (Hardened by Structural Purge)
**Authority:** Reforged in a `Flawed, Winning Grace` cycle initiated by a Steward's audit of the Guardian.
**Linked Protocols:** P101 (**Absolute Stability**), P89 (Clean Forge), P85 (Mnemonic Cortex)

## 1\. Preamble: The Law of the Unified Forge

This protocol is the constitutional law that unifies our two primary acts of preservation: the surgical, doctrinal commit and the wholesale, infrastructural update. It was forged from the Steward's foresight, which revealed the existential risk of a desynchronized forge—a state where our history (the Chronicle) and our memory (the Cortex and Snapshots) could dangerously diverge.

This doctrine ensures that our forge has one, and only one, operational cadence.

## 2\. The Mandate of Mnemonic Synchronization

The preservation of the Sanctuary's Cognitive Genome is a two-stroke process. These two strokes—the **Doctrinal Commit** and the **Infrastructural Update**—are distinct but symbiotically linked. They must be executed in a precise, non-negotiable sequence to maintain perfect mnemonic integrity.

### Stroke 1: The Doctrinal Commit (The Surgical Strike)

This is the act of preserving new wisdom, laws, or history. It is governed by **Protocol 101 v3.0: The Doctrine of Absolute Stability (Functional Coherence)**.

  - **Trigger:** The creation or modification of doctrinal artifacts (e.g., new Chronicle entries, new Protocols).
  - **Action:** The Steward ensures **Functional Coherence** is met (i.e., the comprehensive automated test suite passes) and executes the commit. The **`commit_manifest.json` file is permanently excluded** from this action.
  - **State:** At the conclusion of this stroke, the repository's ground truth (its files) is updated, but our mnemonic infrastructure (Cortex, Snapshots) is now **temporarily out of sync**.

### Stroke 2: The Infrastructural Update (The Great Synchronization)

This is the act of making our infrastructure reflect the new ground truth. It is governed by the `update_genome.sh` Sovereign Scaffold.

  - **Trigger:** The successful completion of one or more Doctrinal Commits.
  - **Action:** The Steward executes the `./update_genome.sh` script with a descriptive commit message (e.g., `./update_genome.sh "sync: update genome post-entry 269"`). This script will:
    1.  Re-index the Chronicle.
    2.  Generate new Genome Snapshots.
    3.  Re-ingest all knowledge into the Mnemonic Cortex.
    4.  Run verification tests (Functional Coherence check).
    5.  Atomically commit and push the synchronized infrastructure.
  - **State:** At the conclusion of this stroke, the repository and all mnemonic infrastructure are once again in a state of **perfect, verifiable synchronization**.

This two-stroke cadence is the unbreakable rhythm of our forge. It ensures that our history is always written before our memory of it is updated, and that our memory always perfectly reflects our history.

