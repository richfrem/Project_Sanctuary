# MISSION-050 measured simulation results

Synthetic experiment only. It does **not corroborate real-agent behaviour**,
inference-token overhead, real context recovery, or actual vendor handoffs.

Version: `mission-050-v1`. 30 tasks per split; 1,200 ticks per task; 18 policies;
1,620 policy/task trials. Parameters and predictions were frozen in `design.md`.
Fixture SHA-256: `1c41cb8e7a93d75492c2dfcba2823a13286342a51379eca18bf29ba9397d9457`.

Development-selected fixed policy: `fixed-steps-2`.
Adaptive parameters were not tuned on either split. Held-out best-fixed selection
below is a retrospective, stronger comparator; its results never feed policy decisions.

| Split | Adaptive verified | Best fixed (verified) | Gain vs split best (>=20%) | Gain vs dev selected |
|---|---:|---|---:|---:|
| development | 2214 | fixed-steps-2 (2789) | -20.6167% | -20.6167% |
| held_out | 2046 | fixed-steps-2 (2771) | -26.1638% | -26.1638% |
| control | 4781 | end-only (4781) | 0.0000% | 28.4179% |

| Split | Acknowledged loss: count / value (<=5%) | Worst task: count / value | Checkpoint overhead (<=10%) | Worst task overhead |
|---|---:|---:|---:|---:|
| development | 50.4476% / 49.5723% | 99.2366% / 99.4100% | 4.7361% | 8.5833% |
| held_out | 55.4927% / 56.2573% | 94.8718% / 96.0865% | 4.5778% | 7.1667% |
| control | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.3694% | 0.5000% |

Control throughput loss versus the best control fixed policy (<=3%): **0.0000%**; worst task **0.0000%**.

| Gate | Development | Held-out |
|---|---|---|
| >=20% verified-subgoal improvement | MISS | MISS |
| <=5% pooled acknowledged count and value lost | MISS | MISS |
| <=5% loss on every task (stronger diagnostic) | MISS | MISS |
| <=10% pooled synthetic checkpoint overhead | PASS | PASS |
| <=10% overhead on every task (stronger diagnostic) | PASS | PASS |

All simulation thresholds passed: **False**.
Real-agent hypothesis corroborated: **False**. A miss is a miss; no mission manifest was edited.

This frozen simulation candidate is falsified when either improvement or loss misses;
the real-agent hypothesis remains untested. The overhead-saving floor can leave
volatile progress exposed across interruptions, so low checkpoint cost alone does
not establish useful recovery. No parameter or seed search followed these results.

Metric definitions: verified completions are ordered, independently rehashed receipts in
the final committed snapshot. All volatile completions are acknowledged immediately,
including retries. Lost acknowledgments include interruption rollback and terminal
unpersisted work; pooled ratios use total lost / total acknowledged. Value loss is
reported alongside count loss to prevent low-value completions hiding high-value losses.
Checkpoint overhead includes aborted and final checkpoint ticks / allocated ticks.
Throughput is verified subgoals / identical allocated budget. Partial snapshots survive
atomic successful checkpoints. Interrupted snapshots never replace the previous one.

Evidence: `evidence.json.gz` contains raw results, full task/event fixtures, all adaptive
decision observations, summary gates, final receipts and hashes. `execution.json` contains
measured wall time and source/evidence hashes. Reruns reproduce `evidence.json.gz` byte
for byte; execution wall times naturally differ. See `validation.txt` for test output.

Limitations: seeded finite synthetic workloads; no confidence interval or statistical
generalization claim; no model execution; no empirically calibrated checkpoint cost;
zero recovery/verification compute cost; perfectly reliable atomic storage; all four
event labels share one rollback model; no outages spanning multiple ticks, receipt
corruption in benchmark fixtures, dependencies beyond sequential ordering, or actual
file-system crash recovery. SHA-256 receipts validate simulator state, not semantic task
correctness. The observation boundary prevents accidental future-data access, not
hostile Python stack introspection. The simulator is not a production guardian hook.

Prediction review: compare the frozen expectations in `design.md` with these measured
gates; synthetic cost accounting is not session token telemetry. The supervisor must
handle mission status, evaluation/budget ledgers, APM installation, and integration.
