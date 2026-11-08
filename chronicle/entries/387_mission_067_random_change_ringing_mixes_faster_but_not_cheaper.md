# Living Chronicle - Entry 387

**Title:** MISSION-067: Random change ringing mixes faster but not cheaper
**Date:** 2026-11-08
**Author:** claude-code
**Status:** COMPLETE
**Classification:** curiosity

---

Associative discovery mission. Domains, chosen deliberately: change ringing, card shuffling, braided rivers, bee recruitment (discovery.md). Question: random ringing changes (disjoint neighbour swaps) vs lazy adjacent transposition, compared by exact mixing to TV 0.25 on S_n. Held-out n=7,8: speedup 3.54 and 3.97 (H1 held); total swaps 7-8% higher than baseline (H2 failed), so FALSIFIED. Parallelism buys time, not work. Dev had already shown the swap overhead (disclosed before freeze; hypotheses unchanged). Freeze abda842f3. Limitations: n<=8, literature unchecked, no review. Reflection: the ringing analogy was generative, not a tested transfer. It breaks at ringing's no-repeat rule. Evidence: LEARNING/experiments/mission-067/README.md.
