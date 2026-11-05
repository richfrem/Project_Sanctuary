# Living Chronicle - Entry 368

**Title:** Fifteen new missions (036-050) committed after passing the schema validator; the bootstrapper now validates mission manifests
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

MISSION-036..050 were proposed by astra on the web from a bundle of key files and documented by agy; they existed only as untracked files in the main checkout. At the human's request they were copied byte-for-byte (hashes verified, fingerprint stable for minutes before copying) into a worktree, validated against LEARNING/research/mission.schema.json and the manage_backlog invariants (50 of 50 pass), and committed with the regenerated index and backlog (identical to the compile already in the main checkout). Authorship of the missions is astra and agy; they were not edited. Bootstrapper: guardian_wakeup.py now prints one line for the manifests (PASS with a count, or WARN naming the first problems; it never turns Layer 1 red or stops the boot); bootstrapping.md, AGENTS.md (bootstrap step 2 and rule 7) and .agent/learning/bootstrap_manifest.json know about the schema and both validator commands. Open finding, not applied: 48 related_missions links from the new missions into existing ones are one-way (e.g. 037 -> 035, 036 -> 015/033/031); the creation checklist asks for reciprocal links, validate does not enforce them, and adding them means editing about two dozen existing manifests including MISSION-035 which is under a lease. Process notes: the new test file was written first and failed 5 of 6 before the change; my first checkout attempts were blocked by my own broad test-folder target and the MISSION-035 lease, so targets were narrowed instead of forcing.
