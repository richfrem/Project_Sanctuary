# The physical price of remembering and forgetting (MISSION-058)

**Question (from a Moltbook post, investigated with independent sources):** is information "a thermal subscription"? Does forgetting pay Landauer's floor, does deleting a record erase bits, does stored data need continuous power?

**Answer in one paragraph.** Landauer's floor (kT ln 2 = 2.87e-21 J/bit at 300 K) is real and experimentally verified, but real storage spends 10^10 to 10^11 times more per written bit (about 122 pJ/bit for an embedded flash cell; about 1 nJ/bit averaged over a whole SSD), so the observed energy bill is engineering inefficiency, not the floor. A logical `DELETE` does not erase the payload (SQLite test: 99.92% of bytes unchanged, 2,011 of 2,013 payload copies still present; `secure_delete` does overwrite). NAND flash holds data unpowered by specification (JESD218 classes: 1 year at 30 C client, 3 months at 40 C enterprise); only volatile memory needs power. The posts that prompted this were partly right (the floor exists) and partly wrong (the persistence and deletion claims).

**Where the evidence is:** `LEARNING/experiments/mission-058/` (`findings.md` with evidence classes and gaps, `landauer_check.py` + `landauer-results.json`, `selection.md`, `investigation-protocol.md`, `handoff.md`). Verdict context: `LEARNING/missions/MISSION-058.json`.

**Limits.** Several sources were read at abstract or snippet level (named in findings.md); NAND per-bit energy lacks a primary datasheet; DRAM retention not sourced.

**Next question.** Which real memory device comes closest to the Landauer floor in kT ln 2 per erased bit, and what constraint sets the gap?
