# Merge prep: regenerate shared generated files (MISSION-034)

Every mission rewrites the same shared files, so parallel branches conflicted on them in 60 of 60 real merges (MISSION-031).
Do not hand-merge them. After merging a branch, run:

```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py regenerate   # `check` = report only
```

| File | How it is handled |
|---|---|
| `.agent/learning/events.jsonl`, `.agent/coordination/leases.jsonl` | Append-only logs: `merge=union` in `.gitattributes` keeps both sides' lines (no conflict). |
| `LEARNING/research/index.jsonl`, `backlog.md` | Rebuilt from `missions/*.json` by `manage_backlog.py compile`. |
| `.agent/learning/cognitive_ledger.json` | Rebuilt from the `CONTRIBUTION_ASSESSED` events, using the same arithmetic as `score_contribution.py`. Events are folded in (timestamp, id) order, so line order after a union merge does not matter. |
| `chronicle/ledger.json` | Rebuilt from `chronicle/entries/*.md` (`chronicle_manager.py ledger`; `create` also refreshes it). Entries whose filename lacks a 3-digit prefix (e.g. `264b_...`) are outside the numbering scheme and not listed. |

After it runs, `git add` the three regenerated files (git still marks them conflicted until you add them).

**It reports, it never hides.** If the stored ledger claims more than its events imply (an assessment edited into the
aggregate with no event), the tool prints `DRIFT`, leaves the file unchanged and exits 3. Resolve by logging the missing
event or reverting the edit; do not edit the aggregate to match. Exit 3 persists until that is done.

**Reconstructed events.** If an aggregate was edited by hand and its event is lost, log an event with `"reconstructed": true`
and a `reconstruction_basis` (the commit that changed the aggregate). State what git shows and mark the rest unknown.
Example: `reconstructed-antigravity-MISSION-024` in `events.jsonl`.

Evidence: `LEARNING/research/experiments/mission-034/README.md`.
