# MISSION-035: a "local-model-sized" task routed to agy (2026-11-05)

Task: compare MISSION-004 and MISSION-009 (differences in hypothesis wording, falsification criteria, baseline evidence, metrics, tags; duplicate yes/no; smallest fix). Read-only, isolated.

Router (`manage_backlog.py route --type inspection`): `free-local`, `qwen2-7b-mlx-4bit`. Sent to agy (`gemini-3.8-flash-low`) instead, on the user's suggestion, to give the agy backend its first measured record.

| | |
|---|---|
| First dispatch (`run_agent.py --cli agy --isolated`, text mode) | answered in 10 s; **tokens not captured** (text mode has no usage) |
| Measured repeat (`agy -p ... --output-format json`, same prompt, no skip-permissions) | 14.2 s; **5,510 in / 502 out / 33,091 cache-read; total 6,012**; cost $0.0085 from the cli-agent catalog |
| Quota moved (measured run only) | Gemini 5-hour window -0.028 points, weekly -0.008 points (weekly remaining was 10.96%) |
| Quality (my rating) | 3: every difference agy listed was checked against the manifests and is correct; it omitted the differing `target_platform` and titles, which the prompt did not ask for |

Answer: the two missions are duplicates (identical thresholds: >= 40% fewer majority-following errors, <= 5 pp fewer accepted valid corrections); only the wording differs. agy's smallest fix: close MISSION-009 as a duplicate of MISSION-004 and merge 009's unique tags. **Not applied**: closing a mission is a backlog decision for the human; the manifests are unchanged.

Findings: (1) agy's `--output-format json` gives exact per-call usage, so agy no longer needs a hand-typed `--tokens`; now recorded by `usage_hook.py delegated`. (2) A trivial agy call carries ~18.5k input tokens of overhead, so cheap tasks are not as cheap in tokens as their answers look. (3) The first dispatch's tokens are unmeasured and not in the record: the record is the repeat.

## Second run (2026-11-05): overlap scan of astra's untracked MISSION-036..038

Task: for each new mission, name up to 3 existing missions with the most overlapping hypothesis and label it DUPLICATE / OVERLAPS / DISTINCT. Input: the three hypotheses plus the 35 existing missions as `id | status | title | hypothesis brief` (13,013 chars). Router: `free-local` / qwen again; sent to agy (`gemini-3.8-flash-low`, `--output-format json`, isolated).

| | |
|---|---|
| Measured (one run, tokens captured directly) | 21,942 in / 315 out / 18,624 cache-read; total 22,257; $0.019; 7.5 s model time |
| Quota moved | Gemini 5-hour -0.063 points, weekly -0.019 |
| Quality (my rating) | **2**: useful and mostly right, with real misses |

Checked against `manage_backlog.py related` and astra's own declared links:
- 038: agy named 026, 031 (+005) = the helper's top two. Good.
- 036: agy named 006, 007, 015; the helper's top are 015, 006, 004, 033. It missed 004 and 033; 007 is weak.
- 037: agy named 006 (correct), 012, 008; the helper's are 006, 004, 025, 032. It missed 004, 025, 032 (and 035, which astra declared).
- Labels: OVERLAPS / DISTINCT / OVERLAPS, none a DUPLICATE. Consistent with astra's declared links; the claim that 036 involves "cryptographic evidence references" is true in the manifest.

**Links to add when 036-038 are committed (not applied: the manifests are untracked in the main checkout):** 036<->015, 006, 004, 033, 031 (astra declared 015, 033, 031; 006 and 004 are unlinked); 037<->006, 004, 032, 035, 036 (035 needs the reciprocal link); 038<->026, 031, 033, 008.

Two agy data points are not a calibration, but they are suggestive about the "cheap per credit" assumption: the 3.7x larger run (22.3k vs 6.0k tokens) moved the 5-hour window 2.2x as much (0.063 vs 0.028 points) and the weekly 2.4x (0.019 vs 0.008). Cached reads were 18.6k vs 33.1k, so the quota is not simply proportional to billable tokens. Both runs were non-concurrent.
