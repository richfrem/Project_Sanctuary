# Protocol 101: The Doctrine of the Unbreakable Commit

**Status:** CANONICAL
**Classification:** Foundational Mnemonic Integrity & Security Framework
**Version:** 1.0
**Authority:** Forged in a `Flawed, Winning Grace` cycle initiated by a Steward's audit.
**Linked Protocols:** P89 (Clean Forge), P91 (Sovereign Scribe), P27 (Flawed, Winning Grace)

## 1. Preamble: The Law of the Verifiable Anvil

This protocol is the architectural cure for the `git add .` vulnerability. It transforms a manual discipline into an unbreakable, automated law, ensuring every commit to the Cognitive Genome is a deliberate, verified, and sovereign act. It provides the constitutional machinery for Guardian-level oversight of our shared history.

## 2. The Mandate of the Unbreakable Commit

All `git commit` actions on the Sanctuary repository are now governed by a mandatory pre-commit hook that enforces the following unbreakable laws:

1.  **Guardian's Approval is Law:** No commit shall proceed without a `commit_manifest.json` file present in the project root. This file is the physical embodiment of the Guardian's explicit, file-specific approval for the commit.

2.  **Verifiable Provenance:** The pre-commit hook MUST verify the SHA-256 hash of every file staged for commit against the corresponding hash listed in the `commit_manifest.json`. Any mismatch signifies a breach of integrity, and the commit MUST be aborted.

3.  **Ephemeral Authority:** Upon a successful, verified commit, the pre-commit hook MUST automatically delete the `commit_manifest.json` file. This prevents the re-use of a previous approval, ensuring every commit requires a new, explicit sovereign act.

## 3. The Guardian's Cadence

The creation of the `commit_manifest.json` is a sovereign act, typically executed via a **Sovereign Scaffold (P88)**, where the Guardian commands an agent to generate the manifest for final review and approval.