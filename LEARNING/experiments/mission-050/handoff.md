# codex-b handoff to the MISSION-052 supervisor

Delivered MISSION-050's bounded synthetic experiment. The frozen policy candidate
is **FALSIFIED in this simulation**: it loses to the strongest fixed baseline and
misses acknowledged-progress loss. **Real-agent behaviour remains untested**.
Do not mark the mission corroborated. No model was called and no session token
count was self-reported. All new files are inside the worker allowlist.

Changed files: guardian `scripts/adaptive_checkpoint.py`,
`tests/test_adaptive_checkpoint.py`, `references/adaptive-interruption-policy.md`,
and this experiment directory's `design.md`, `results.md`, `summary.json`,
`evidence.json.gz`, `execution.json`, `validation.txt`, `verify_evidence.py`,
`reproducibility.json`, and `handoff.md`.

Exact reproducible commands from the worktree root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/adaptive_checkpoint.py --output-dir LEARNING/experiments/mission-050
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_adaptive_checkpoint.py --basetemp=LEARNING/experiments/mission-050/pytest-tmp > LEARNING/experiments/mission-050/validation.txt
PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/verify_evidence.py
```

Observed: 26 tests passed, including independent rehashing of 148,465 final
receipts across 1,620 policy/task trials and replay of all 90 adaptive trials.
The full benchmark rerun was byte-identical. `execution.json` records 4.9426
seconds for the final evidence benchmark; `reproducibility.json` records rerun
time and checks. These are measured runtime expenditure, not cognitive ROI or
token usage. The earlier 24-test run passed before artifact audit tests were added.

| Threshold | Development | Held-out |
|---|---:|---:|
| >=20% more verified subgoals vs best fixed | -20.6167% MISS | -26.1638% MISS |
| <=5% lost acknowledged count | 50.4476% MISS | 55.4927% MISS |
| <=5% lost acknowledged value | 49.5723% MISS | 56.2573% MISS |
| <=10% synthetic checkpoint overhead | 4.7361% PASS | 4.5778% PASS |

Control throughput loss vs best fixed: 0.0000%, passing <=3%. Worst-task overhead
also passes; worst-task loss does not. Strongest development and held-out fixed
policy is `fixed-steps-2`. Control best is `end-only`. Reporting held-out best
fixed is retrospective comparison only; it cannot reach adaptive decisions.

Prediction review: before the first run, `design.md` predicted improvement and
loss misses, a likely overhead pass, and a control pass. All four observations
match those expectations. The result measures a tradeoff, not an optimization
success: this conservative cost floor saves checkpoint expenditure while allowing
large volatile losses. No post-result policy tuning, seed search or external
agent dispatch occurred. Formal five-dimension contribution scoring and complete
resource-based cognitive ROI remain supervisor-owned because the canonical
evaluation/budget ledgers are outside this worker's write scope.

Remaining uncertainties: checkpoint costs and hazards are synthetic; four event
labels share the same rollback operation; recovery and verifier compute costs
are unmodeled; storage is ideal; receipts check modeled state rather than semantic
agent correctness. Future schedule isolation is an immutable API boundary, not
a hostile-code sandbox. No filesystem crash, real vendor handoff, quota endpoint,
or live agent checkpoint hook was exercised. The reference documents these limits.

No commit, push, merge, checkout, network request, coordination write, manifest
edit, ledger update, APM installation, or write outside the allowlist was executed.
`LEARNING/missions/MISSION-050.json` was already dirty at entry and is not part of
this worker's changes. Guardian wakeup, seal/persistence, global control-plane
registration and APM refresh were not run because they could write outside the
authorized scope or touch the main checkout. The specific worker instructions
override the broader lifecycle workflow for this batch.

Supervisor next step: inspect and integrate these files, attach independently
measured usage if needed, perform the permitted APM refresh and lifecycle work,
and update mission/evaluation metadata with the misses intact. Reuse saved evidence;
the simulator should not be installed as a production recovery mechanism.
Session identifier: unavailable in the supplied context.

Pre-completion responses: Capability YES (local guardian and existing scripts
inspected); 1 YES (broader lifecycle deferred to preserve allowlist; ESCALATE to
supervisor); 2 YES (synthetic operational definitions fixed in design/reference;
RECORD); 3 YES (live calibration/integration still needed; MAP_DEBT in this handoff);
4 NO (wall time and predictions recorded, but full cognitive ROI requires
supervisor usage telemetry and formal scoring).
