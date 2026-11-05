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
