# MISSION-023 protocol (frozen before held-out run)

Hypothesis (manifest, unchanged): a 1.5B or 3B draft model alongside Qwen2.5-Coder-7B-Instruct-4bit via MLX speculative decoding increases generation throughput by >= 1.6x while staying within an 8 GiB memory budget (M1, 16 GB unified memory).

- Instruments: `bench_spec.py`, mlx-lm 0.32.2 via `uvx --from mlx-lm==0.32.2`; models `mlx-community/Qwen2.5-Coder-{7B,3B,1.5B}-Instruct-4bit` (HF cache snapshots recorded in results).
- Decoding: greedy, max 256 new tokens, chat template; one warmup per model set; the MLX server on :8080 is stopped during measurement and restored afterwards.
- Metric: per-prompt speedup = gen_tps(spec) / gen_tps(baseline) using mlx-lm's reported generation tokens/s; primary = median over prompts. Memory = max over prompts of `mx.get_peak_memory()` (target + draft + KV), in GiB.
- Dev (3 prompts): sweep draft in {1.5B, 3B} x num_draft_tokens in {2, 4}; select the single config with the highest dev median speedup. Dev selects; it does not decide.
- Held-out (10 prompts, fixed in `bench_spec.py` before any run): baseline vs the one selected config, one run.
- Verdict: CORROBORATED if held-out median speedup >= 1.6 AND max peak <= 8.0 GiB; else FALSIFIED. UNASSESSED if the run fails.
- Descriptive only: identical-output count (greedy speculative decoding should be lossless; mismatches indicate numeric divergence), draft acceptance fraction, min speedup, wall time.
- Limits: one machine, one run per prompt (no repeat variance), short code/explanation prompts, mlx-lm's tps excludes prompt processing.
- Budget: < 30 min of local compute; no cloud model dispatch. Orchestrated by claude-code; the subject models run locally.

## Dev selection (recorded before freeze; criteria unchanged)
Dev medians: 1.5B k2 1.48x (peak 4.97 GiB), 1.5B k4 1.36x, 3B k2 1.15x, 3B k4 1.05x; baseline 13.8 tok/s, 4.15 GiB; all outputs identical to baseline. Selected config: **1.5B, num_draft_tokens=2**. Its dev median is already below 1.6x; the threshold is kept as written.
