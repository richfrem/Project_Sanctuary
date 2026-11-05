# Process branch: related-mission scan (pre-registration)

Branch/lease: `PROCESS-related-mission-scan` (no mission manifest; a process change requested by the human on 2026-11-04 after MISSION-035's links to 002/016/028/033 were found only by a manual scan).

## Hypothesis
A deterministic stdlib helper, `manage_backlog.py related`, ranking all other missions by TF-IDF cosine similarity of title+hypothesis plus tag overlap, would have surfaced the links a human later judged relevant for MISSION-035 without being told them.

## Ground truth (fixed before implementing)
Human-judged related set for MISSION-035 (from the merged manifest links): {MISSION-002, MISSION-016, MISSION-028, MISSION-033, MISSION-031}. Judged noise from the earlier keyword scan: {MISSION-012, MISSION-023, MISSION-029}.

## Predictions
1. recall@8 >= 0.8 (at least 4 of the 5 related missions in the top 8).
2. At most 1 of the 3 noise missions in the top 5.
3. Runtime < 1 s over the current ~35 manifests.
4. `related` also works for an undrafted idea via `--text`, and flags link state (linked / one-way / unlinked) per candidate.

## Falsified if
recall@8 < 0.8, or 2+ noise missions in the top 5, or runtime >= 1 s. A miss is reported with the ranks observed, not tuned away silently (a single re-weighting pass is allowed only if disclosed as post-hoc).

## Not claimed
Similarity is lexical. It cannot prove two missions are semantically related and does not replace the human/agent judgement of the links it suggests.

## Results (2026-11-04, live manifests, `related --id MISSION-035`)
| Prediction | Result |
|---|---|
| recall@8 >= 0.8 | **MISSED: 3/5 (0.6)**. Ranks: 016 #1, 033 #2, 002 #5, 028 #9, 031 not returned |
| noise (012, 023, 029) in top 5 <= 1 | Met: 0 of 3 in the top 10 |
| runtime < 1 s | Met: 0.056 s |
| `--text` mode and link state | Met (unit tests; live run shows linked / one-way / unlinked) |

Post-hoc, disclosed: the default `--top` was raised from 8 to 10 after seeing 028 at rank 9, which gives recall@10 = 4/5 (0.8). That is a display-size choice, not a pass of the registered criterion, which stays a miss. No re-weighting was done. n = 1 target, and the ground-truth set came from the same author's later judgement, so this is weak evidence.

Why 031 is missed: its relation to MISSION-035 is mechanical (usage snapshots attach to the lease lifecycle that 031 built), not lexical. This is why the checklist adds a manual "what does this plug into" question, and why the helper is described as an aid, not a replacement for judgement.

Also found: MISSION-035's PR #299 shipped the flight-board's IN_PROGRESS marker (status left IN_PROGRESS on main for an unclaimed mission). Restored to PROPOSED on this branch; defect captured as inbox item SEED-20261104-8f0725.
