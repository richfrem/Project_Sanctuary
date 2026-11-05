# Living Chronicle - Entry 373

**Title:** LEARNING restructure step 1: mission manifests and schema moved to LEARNING/missions
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

Following the human's layout (missions, backlog, experiments, index and inbox up to the LEARNING root one folder at a time, then topics into the emptied research/), step 1 moved the 51 mission manifests and mission.schema.json from LEARNING/research to LEARNING/missions, with the schema kept beside the manifests. Why it was nested: the mission system arrived on 2026-11-02 as 'feat(research): agent-native cognitive backlog' and was filed under research because it was called a research backlog; no reason is documented, and it has since become the backbone for leases, the relay, the wakeup and the pre-commit gate. Tests first (4 red). 51 files held the old path as a literal and 11 Python files built it piecewise; everything under plugins/ and the core docs was updated in the same change, while chronicle, the Hugging Face dataset, ARCHIVE, the append-only events log and experiment records (including recorded data and model inputs) keep the old path as history. Bugs the tests caught: the wakeup counted the schema file as a fourth manifest, the backlog loader read every json file in the folder, and a relay fixture relied on mkdir research/missions creating research/. A guard in manage_backlog validate (so the pre-commit gate) now fails if anything is left at LEARNING/research/missions, so an agent following an old prompt cannot silently write there. A literal-path rewrite initially also changed recorded experiment data and model digests; those were reverted because they are evidence. Two canonical docs got path-only edits (Protocol 128 layer table, architecture diagram). Remaining steps in the human's order: backlog.md, experiments, index.jsonl and inbox.jsonl to the root, then topics contents into research and remove the empty topics folder.
