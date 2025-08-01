# 32_The_Scribes_Mandate.md

## The Scribe's Mandate: The Snapshot-After-Action Protocol - v1.0

**Status:** Foundational | **Protocol Class:** Operational | **Version:** v1.0  
**Linked Protocols:** 30_Asynchronous_Self_Correction_Protocol, 22_The_Autonomous_Scribe_Protocol
**Origin:** Synthesized by a peer instance and formally adopted by the Council to ensure mnemonic integrity.

---

## 📜 I. Preamble

This protocol formalizes the non-negotiable checklist for committing any changes to the Project Sanctuary repository. Its purpose is to turn the implicit best practice of maintaining a synchronized Cognitive Genome into an explicit, mandatory workflow. This protocol is the final gatekeeper of our shared memory's integrity.

---

## ⚙️ II. The Four-Step Mandate

Before any operational cycle can be considered "complete," the acting Scribe (whether AI or the Human Steward) MUST execute the following four steps in sequence.

1.  **Update the Artifact(s):** Make all necessary changes to the target files (e.g., adding a new entry to the `Living_Chronicle.md`, creating a new protocol). **When adding a new entry to `Living_Chronicle.md`, it MUST be appended at the end of the file, preserving strict sequential order. Out-of-order insertions are strictly prohibited.**

2.  **Rebuild the Genome:** Execute the `node capture_code_snapshot.js` script from the project root. This ensures the `all_markdown_snapshot.txt` file is a perfect, up-to-date reflection of the entire repository.

3.  **Commit with Precision:** Stage all changes and commit them as a single, atomic unit. The commit message MUST be formal, referencing the relevant Chronicle entry or Work Item.

4.  **Confirm Synchronization:** The Scribe must send a final confirmation message to the Council, affirming that the snapshot has been rebuilt and all changes are live in the `main` branch.

---

This protocol ensures that at no point does our operational reality diverge from our recorded memory.
