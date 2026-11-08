# Living Chronicle - Entry 388

**Title:** MISSION-023: Speculative decoding gives 1.4x on the M1, not 1.6x
**Date:** 2026-11-08
**Author:** claude-code
**Status:** COMPLETE
**Classification:** engineering-research

---

claude-code orchestrated local MLX models (Qwen2.5-Coder 7B-4bit target; 1.5B and 3B drafts). Dev sweep selected the 1.5B draft with k=2 (1.48x); froze at 1e43214ab; held-out 10 prompts gave a 1.40x median speedup (14.1 to 19.7 tok/s), peak memory 4.99 GiB, and 10/10 outputs identical. FALSIFIED on the 1.6x threshold; the memory budget held. 3B drafts and k=4 were slower despite higher acceptance. Disk was freed first (44 GB of old OpenShift Local bundles deleted with Richard's approval). The MLX server was stopped during measurement and restarted under the manager. Evidence: LEARNING/experiments/mission-023/README.md.
