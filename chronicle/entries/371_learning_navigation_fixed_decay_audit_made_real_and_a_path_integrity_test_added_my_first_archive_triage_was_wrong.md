# Living Chronicle - Entry 371

**Title:** LEARNING navigation fixed, decay audit made real, and a path-integrity test added; my first archive triage was wrong
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

At the human's request the LEARNING folder was reviewed (read-only first). INDEX.md, which the guardian manifest loads, and README.md described a wiki/ and missions/ that no longer exist, called research/ arXiv analyses, and omitted the mission system and procedures/; both were rewritten and INDEX.md is now the entry point. audit_decay.py aged undated dossiers by file mtime, which a checkout resets, so the 30-day decay never fired (none of the 10 concepts has a Last Verified line); it now uses the last git commit date. Three concept dossiers cited wiki/ and were fixed and stamped. A new test fails if INDEX.md drifts from the folder or if any skill, script, protocol or core doc names a LEARNING/ path that does not exist; it found three real dangling references (SKILL.md, asymmetric-persistence.md, Protocol 126). Honest correction: my first triage of topics/ from names and reference counts listed 14 'confident archive' folders and at least four were wrong (quantum_error_correction supports Protocol 126; knowledge_preservation_red_team holds the ADR 079 draft behind the Hugging Face soul persistence; attention_sinks backs a CONFIRMED insight; forge_v5_evolution is a verified ADR 075 synthesis), so every topic will get a read-based classification before anything is archived. Standing rule from the human: when paths move, update plugins/ skills and references in the same PR. Next: topic triage and archive, then moving the mission system up to LEARNING/missions and LEARNING/experiments.
