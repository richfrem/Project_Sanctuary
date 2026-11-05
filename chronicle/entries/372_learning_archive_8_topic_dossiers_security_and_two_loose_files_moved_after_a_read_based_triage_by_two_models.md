# Living Chronicle - Entry 372

**Title:** LEARNING archive: 8 topic dossiers, security and two loose files moved after a read-based triage by two models
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

After the navigation fixes (chronicle 371), the 24 topics/ dossiers were triaged by reading, not by name: a digest per topic (title, status and date lines, opening paragraph, who cites it) went to agy gemini-3.8-flash-low and Haiku 5.5 with one rubric, and every disagreement was checked against the repository. agy matched my final calls on 17 of 24 and Haiku on 15; both under-archived (2 and 1 of my 8) and neither was usable as-is, because the digest lacked facts such as Protocol 131 citing the DRQ dossier. Corrections found by reading: the astra_architectural_review citation was a chart label and it is KEEP only because it is recent; soul_persistence hits were the concept, so it is HISTORICAL; ADR 075 and ADR 079 already live in ARCHIVE/ADRs so their drafting-history topics belong beside them. Moved with git mv to ARCHIVE/LEARNING_pre_2026-11/ (53 files, about 280 KB, with an old-to-new map and restore instructions): scholarly_discourse, openclaw_moltbook_ecosystem, safe_agent_zero (the Moltbook episode), documentation_link_remediation, protocol_128_remediation, liquid_neural_networks, forge_v5_evolution, knowledge_preservation_red_team, security/, and two loose files. research/2026-02-12 became topics/arxiv_2602_deep_dive. topics/README.md marks the 17 remaining dossiers KEEP (6) or HISTORICAL (11). Guard tests: every topic has a status row, every moved item is mapped, archived files are tracked, no kept document links into moved paths. ARCHIVE matches a .gitignore rule and its existing exception is ineffective, so the move relies on git mv. The two model runs are the first same-task pair for MISSION-051: Haiku 57,228 tokens, 0.015 USD, 28 s versus agy 23,564 tokens, 0.022 USD, 13 s (n=1). Next: move research/ up to LEARNING/missions and LEARNING/experiments with every plugins/ reference updated in the same PR.
