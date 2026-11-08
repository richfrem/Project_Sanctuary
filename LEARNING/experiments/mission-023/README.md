# MISSION-023: MLX speculative decoding on an M1 (16 GB)

**Verdict: FALSIFIED.** Speed criterion failed, memory criterion held. Freeze commit `1e43214ab` hashes protocol.md, bench_spec.py and dev_results.json; the selected config was fixed on dev before the held-out run.

## Setup
Target `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`; drafts Qwen2.5-Coder 1.5B and 3B (4-bit, MLX); mlx-lm 0.32.2 (`uvx`); greedy decoding, 256 new tokens, chat template. The local MLX server on :8080 was stopped during measurement and restarted afterwards.

## Dev sweep (3 prompts; selects only)
| config | median tok/s | median speedup | peak GiB |
|---|---|---|---|
| baseline | 13.8 | 1.00 | 4.15 |
| 1.5B, k=2 | 19.9 | **1.48** | 4.97 |
| 1.5B, k=4 | 18.2 | 1.36 | 4.97 |
| 3B, k=2 | 15.5 | 1.15 | 5.80 |
| 3B, k=4 | 14.5 | 1.05 | 5.81 |

## Held-out (10 prompts, selected config 1.5B k=2)
- Baseline 14.1 tok/s median; speculative 19.7 tok/s median.
- **Median speedup 1.40x** (min 1.30x), below the 1.6x threshold, so H fails.
- **Peak memory 4.99 GiB**, within the 8 GiB budget.
- 10/10 outputs token-identical to baseline (lossless, as expected for greedy).
- Draft acceptance about 63% of emitted tokens.

## Observations
- The smaller draft wins: the 3B draft has the same acceptance rate as the 1.5B (about 65% at k=2) but costs more per step, so it gives almost no gain.
- Larger k raises acceptance (77% at k=4) but lowers throughput. Rejected drafts waste compute on a bandwidth-limited M1.
- Speedup was very stable across prompts (1.30-1.5x); baseline throughput varied by <2%.

## Limitations
One machine, one run per prompt, short coding/explanation prompts (code-heavy or repetitive text may accept more drafts), no 0.5B draft tested, k only in {2,4}. mlx-lm's generation tok/s excludes prompt processing. The selection of 1 of 4 configs on 3 dev prompts is a small sample.

## Practical takeaway
On this M1, a 1.5B draft with k=2 is a free ~1.4x speed-up with identical greedy output and about +0.8 GiB memory. It is worth enabling for local Qwen generation, but it is not the 1.6x the mission predicted.

## Reproduce
`HF_HUB_OFFLINE=1 uvx --from mlx-lm==0.32.2 python bench_spec.py --mode heldout --draft 1.5B --k 2` (MLX server stopped first).
