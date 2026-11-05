#!/usr/bin/env python3
"""Deterministic, offline checkpoint simulation; never a production recovery hook.

Only the runner receives schedules. AdaptivePolicy receives a frozen value-only
observation: measured historical hazard, progress value/rate, checkpoint cost.
Resource ticks are synthetic units, not estimates of inference tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time


VERSION = "mission-050-v1"
TIME_INTERVALS = (8, 16, 24, 32, 48, 64, 96, 128, 256, 512)
STEP_INTERVALS = (1, 2, 4, 8, 16, 32)
EVENT_KINDS = ("context_reset", "worker_termination", "quota_denial", "vendor_handoff")


@dataclass(frozen=True, slots=True)
class Subgoal:
    """A sequential unit of work with an independently checkable receipt."""

    work: int
    value: int


@dataclass(frozen=True, slots=True)
class Task:
    """Exogenous workload; not provided to policy code."""

    seed: int
    goals: tuple[Subgoal, ...]
    checkpoint_cost: int
    budget: int


@dataclass(frozen=True, slots=True)
class Event:
    """An interruption at the end of a one-indexed resource tick."""

    tick: int
    kind: str


@dataclass(frozen=True, slots=True)
class AdaptiveObservation:
    """Entire adaptive-policy input; contains no runner or future references."""

    interruption_rate: float
    pending_value: float
    value_rate: float
    checkpoint_cost: int

    def __post_init__(self) -> None:
        for value in (self.interruption_rate, self.pending_value, self.value_rate):
            if not math.isfinite(value) or value < 0:
                raise ValueError("observed values must be finite and nonnegative")
        if self.checkpoint_cost < 1:
            raise ValueError("checkpoint cost must be positive")


@dataclass(frozen=True, slots=True)
class FixedObservation:
    """Fixed policies receive elapsed time and completed progress only."""

    elapsed_since_checkpoint: int
    completed_since_checkpoint: int


@dataclass(frozen=True, slots=True)
class AdaptivePolicy:
    """A stateless policy using measured history only; no tuning on held-out data."""

    name: str = "adaptive"

    def should_checkpoint(self, observed: AdaptiveObservation) -> bool:
        """Balance measured hazard and cost, with a 10% overhead target."""
        if observed.interruption_rate == 0 or observed.value_rate == 0:
            return False
        threshold = max(
            math.sqrt(
                2 * observed.checkpoint_cost * observed.value_rate
                / observed.interruption_rate
            ),
            9 * observed.checkpoint_cost * observed.value_rate,
        )
        return observed.pending_value >= threshold


@dataclass(frozen=True, slots=True)
class FixedPolicy:
    """Fixed wall-time, completed-subgoal, or end-only checkpoint policy."""

    mode: str
    interval: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("time", "steps", "end"):
            raise ValueError("unknown fixed policy mode")
        if self.mode != "end" and self.interval < 1:
            raise ValueError("fixed intervals must be positive")

    @property
    def name(self) -> str:
        return "end-only" if self.mode == "end" else f"fixed-{self.mode}-{self.interval}"

    def should_checkpoint(self, observed: FixedObservation) -> bool:
        """Apply a fixed threshold without schedule access."""
        if self.mode == "time":
            return observed.elapsed_since_checkpoint >= self.interval
        if self.mode == "steps":
            return observed.completed_since_checkpoint >= self.interval
        return False


def receipt(task: Task, index: int) -> str:
    """Compute a canonical completion proof from workload facts."""
    goal = task.goals[index]
    payload = f"{VERSION}:{task.seed}:{index}:{goal.work}:{goal.value}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def verify_receipts(task: Task, receipts: tuple[tuple[int, str], ...]) -> bool:
    """Independently verify ordered receipts rather than trusting a count."""
    return len(receipts) <= len(task.goals) and all(
        index == expected and digest == receipt(task, expected)
        for expected, (index, digest) in enumerate(receipts)
    )


def make_task(seed: int, budget: int = 1200, goal_count: int = 300) -> Task:
    """Generate a workload independently of any interruption schedule."""
    rng = random.Random(seed)
    goals = tuple(Subgoal(rng.randint(3, 12), rng.randint(1, 5)) for _ in range(goal_count))
    return Task(seed, goals, rng.randint(2, 6), budget)


def make_schedule(seed: int, budget: int, profile: int) -> tuple[Event, ...]:
    """Freeze exogenous events before any policy executes."""
    if profile not in range(6):
        raise ValueError("profile must be in 0..5")
    rng = random.Random(seed)
    events = []
    for tick in range(1, budget + 1):
        if profile < 3:
            rate = (0.01, 0.025, 0.05)[profile]
        elif profile == 3:
            rate = 0.003 if tick <= budget // 2 else 0.06
        elif profile == 4:
            rate = 0.06 if tick <= budget // 2 else 0.003
        else:
            rate = 0.07 if (tick // 100) % 2 else 0.002
        if rng.random() < rate:
            events.append(Event(tick, EVENT_KINDS[len(events) % len(EVENT_KINDS)]))
    return tuple(events)


def run_trial(
    task: Task,
    events: tuple[Event, ...],
    policy: AdaptivePolicy | FixedPolicy,
    *,
    trace: bool = False,
) -> dict:
    """Run one budget with atomic, metered checkpoints and verified recovery.

    Interruptions happen after a work tick (including its acknowledgment) or
    checkpoint tick, before a checkpoint commit. The previous snapshot survives.
    Only completed receipts are acknowledged. Snapshots also preserve partial
    work; lost partial work is not miscounted as an acknowledged completion.
    """
    if task.budget < 1 or task.checkpoint_cost < 1 or not task.goals:
        raise ValueError("positive budget, checkpoint cost, and nonempty goals required")
    if any(goal.work < 1 or goal.value < 1 for goal in task.goals):
        raise ValueError("subgoal work and value must be positive")
    event_by_tick = {event.tick: event for event in events}
    if len(event_by_tick) != len(events) or any(
        event.tick < 1 or event.tick > task.budget or event.kind not in EVENT_KINDS
        for event in events
    ):
        raise ValueError("events must have unique in-budget ticks and known kinds")
    prefix_value = [0]
    for goal in task.goals:
        prefix_value.append(prefix_value[-1] + goal.value)
    completed = durable_completed = partial = durable_partial = 0
    receipts: list[tuple[int, str]] = []
    durable_receipts: tuple[tuple[int, str], ...] = ()
    elapsed = checkpoint_ticks = work_ticks = interrupted = 0
    checkpoint_attempts = checkpoint_commits = last_checkpoint = 0
    checkpoint_remaining = 0
    ack_count = ack_value = lost_count = lost_value = lost_partial_work = 0
    earned_value = 0.0
    decisions = []

    def progress_value(count: int, progress: int) -> float:
        if count == len(task.goals):
            return float(prefix_value[count])
        goal = task.goals[count]
        return prefix_value[count] + goal.value * progress / goal.work

    while elapsed < task.budget:
        pending = (completed, partial) != (durable_completed, durable_partial)
        if checkpoint_remaining == 0:
            remaining = task.budget - elapsed
            # Common budget-aware final flush: no free post-budget persistence.
            final_flush = pending and (
                remaining <= task.checkpoint_cost or completed == len(task.goals)
            )
            if isinstance(policy, AdaptivePolicy):
                observed = AdaptiveObservation(
                    interrupted / elapsed if elapsed else 0.0,
                    max(0.0, progress_value(completed, partial)
                        - progress_value(durable_completed, durable_partial)),
                    earned_value / work_ticks if work_ticks else 0.0,
                    task.checkpoint_cost,
                )
            else:
                observed = FixedObservation(elapsed - last_checkpoint,
                                            completed - durable_completed)
            requested = policy.should_checkpoint(observed)
            if trace:
                decisions.append({"tick": elapsed, "observation": asdict(observed),
                                  "requested": requested, "final_flush": final_flush})
            if pending and (requested or final_flush):
                if remaining < task.checkpoint_cost:
                    break
                checkpoint_remaining = task.checkpoint_cost
                checkpoint_attempts += 1
            elif completed == len(task.goals):
                break

        elapsed += 1
        if checkpoint_remaining:
            checkpoint_ticks += 1
            checkpoint_remaining -= 1
            committing = checkpoint_remaining == 0
        else:
            work_ticks += 1
            goal = task.goals[completed]
            partial += 1
            earned_value += goal.value / goal.work
            committing = False
            if partial == goal.work:
                receipts.append((completed, receipt(task, completed)))
                ack_count += 1
                ack_value += goal.value
                completed += 1
                partial = 0

        if elapsed in event_by_tick:
            interrupted += 1
            lost_count += completed - durable_completed
            lost_value += prefix_value[completed] - prefix_value[durable_completed]
            # Count only uncompleted work; completed losses are recorded above.
            lost_partial_work += partial - durable_partial if completed == durable_completed else partial
            if not verify_receipts(task, durable_receipts):
                raise ValueError("corrupt durable snapshot during recovery")
            completed, partial = durable_completed, durable_partial
            receipts = list(durable_receipts)
            checkpoint_remaining = 0
        elif committing:
            candidate = tuple(receipts)
            if not verify_receipts(task, candidate):
                raise ValueError("checkpoint receipt verification failed")
            durable_completed, durable_partial = completed, partial
            durable_receipts = candidate
            checkpoint_commits += 1
            last_checkpoint = elapsed

    lost_count += completed - durable_completed
    lost_value += prefix_value[completed] - prefix_value[durable_completed]
    if not verify_receipts(task, durable_receipts):
        raise ValueError("final independent verification failed")
    if ack_count != durable_completed + lost_count or ack_value != prefix_value[durable_completed] + lost_value:
        raise AssertionError("acknowledgment conservation violated")
    result = {
        "task_seed": task.seed, "policy": policy.name, "budget": task.budget,
        "elapsed": elapsed, "work_ticks": work_ticks, "checkpoint_ticks": checkpoint_ticks,
        "checkpoint_attempts": checkpoint_attempts, "checkpoint_commits": checkpoint_commits,
        "observed_interruptions": interrupted, "verified_subgoals": len(durable_receipts),
        "verified_value": prefix_value[durable_completed], "acknowledged_subgoals": ack_count,
        "acknowledged_value": ack_value, "lost_acknowledged_subgoals": lost_count,
        "lost_acknowledged_value": lost_value, "lost_partial_work": lost_partial_work,
        "loss_count_fraction": lost_count / ack_count if ack_count else 0.0,
        "loss_value_fraction": lost_value / ack_value if ack_value else 0.0,
        "overhead_fraction": checkpoint_ticks / task.budget,
        "verified_receipts": durable_receipts,
        "verified_receipts_sha256": digest_json(durable_receipts),
    }
    if trace:
        result["decisions"] = decisions
    return result


def digest_json(value: object) -> str:
    """Hash a canonical JSON value for reproducibility checks."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixed_family() -> tuple[FixedPolicy, ...]:
    """Return the preregistered baseline grid, including end-only persistence."""
    return (FixedPolicy("end"), *(FixedPolicy("time", n) for n in TIME_INTERVALS),
            *(FixedPolicy("steps", n) for n in STEP_INTERVALS))


def summarize(trials: list[dict]) -> dict:
    """Pool denominators rather than average per-task percentages."""
    keys = ("budget", "checkpoint_ticks", "verified_subgoals", "verified_value",
            "acknowledged_subgoals", "acknowledged_value", "lost_acknowledged_subgoals",
            "lost_acknowledged_value", "observed_interruptions")
    totals = {key: sum(trial[key] for trial in trials) for key in keys}
    totals.update({
        "loss_count_fraction": totals["lost_acknowledged_subgoals"] / totals["acknowledged_subgoals"]
        if totals["acknowledged_subgoals"] else 0.0,
        "loss_value_fraction": totals["lost_acknowledged_value"] / totals["acknowledged_value"]
        if totals["acknowledged_value"] else 0.0,
        "overhead_fraction": totals["checkpoint_ticks"] / totals["budget"],
        "worst_task_loss_count_fraction": max(t["loss_count_fraction"] for t in trials),
        "worst_task_loss_value_fraction": max(t["loss_value_fraction"] for t in trials),
        "worst_task_overhead_fraction": max(t["overhead_fraction"] for t in trials),
    })
    return totals


def best_fixed(summaries: dict) -> str:
    """Select by verified count, value, cost, then stable lexical name."""
    names = [name for name in summaries if name != "adaptive"]
    return min(names, key=lambda name: (-summaries[name]["verified_subgoals"],
                                       -summaries[name]["verified_value"],
                                       summaries[name]["checkpoint_ticks"], name))


def compare(adaptive: dict, baseline: dict) -> dict:
    """Compute gates without rounding borderline values before comparison."""
    denominator = baseline["verified_subgoals"]
    gain = adaptive["verified_subgoals"] / denominator - 1 if denominator else None
    return {
        "verified_subgoal_gain": gain,
        "improvement_pass": denominator > 0 and adaptive["verified_subgoals"] * 100 >= denominator * 120,
        "pooled_loss_pass": max(adaptive["loss_count_fraction"], adaptive["loss_value_fraction"]) <= 0.05,
        "every_task_loss_pass": max(adaptive["worst_task_loss_count_fraction"],
                                    adaptive["worst_task_loss_value_fraction"]) <= 0.05,
        "pooled_overhead_pass": adaptive["overhead_fraction"] <= 0.10,
        "every_task_overhead_pass": adaptive["worst_task_overhead_fraction"] <= 0.10,
    }


def benchmark() -> dict:
    """Evaluate frozen development/held-out splits; never tune adaptive policy."""
    policies = (*fixed_family(), AdaptivePolicy())
    fixtures = {}
    for split, task_seed, schedule_seed in (("development", 10000, 20000),
                                             ("held_out", 30000, 40000)):
        fixtures[split] = [
            {"task": asdict(task := make_task(task_seed + i)),
             "events": [asdict(event) for event in make_schedule(schedule_seed + i, task.budget, i % 6)],
             "profile": i % 6, "schedule_seed": schedule_seed + i}
            for i in range(30)
        ]
    fixtures["control"] = [{**fixture, "events": [], "schedule_seed": None}
                           for fixture in fixtures["held_out"]]
    splits = {}
    # Development completes before held-out evaluation; selection is frozen here.
    selected_fixed = None
    for split in ("development", "held_out", "control"):
        trials = []
        for fixture in fixtures[split]:
            data = fixture["task"]
            task = Task(data["seed"], tuple(Subgoal(**goal) for goal in data["goals"]),
                        data["checkpoint_cost"], data["budget"])
            events = tuple(Event(**event) for event in fixture["events"])
            for policy in policies:
                trials.append(run_trial(task, events, policy, trace=isinstance(policy, AdaptivePolicy)))
        summaries = {policy.name: summarize([t for t in trials if t["policy"] == policy.name])
                     for policy in policies}
        winner = best_fixed(summaries)
        if split == "development":
            selected_fixed = winner
        splits[split] = {"summaries": summaries, "trials": trials,
                         "best_fixed_retrospective": winner,
                         "vs_development_selected": compare(summaries["adaptive"], summaries[selected_fixed]),
                         "vs_split_best_fixed": compare(summaries["adaptive"], summaries[winner])}
    control = splits["control"]
    control_best = control["summaries"][control["best_fixed_retrospective"]]["verified_subgoals"]
    control_loss = 1 - control["summaries"]["adaptive"]["verified_subgoals"] / control_best
    control["throughput_loss_fraction"] = control_loss
    control["throughput_pass"] = control_loss <= 0.03
    best_control_trials = {t["task_seed"]: t for t in control["trials"]
                           if t["policy"] == control["best_fixed_retrospective"]}
    control["worst_task_throughput_loss_fraction"] = max(
        1 - t["verified_subgoals"] / best_control_trials[t["task_seed"]]["verified_subgoals"]
        for t in control["trials"] if t["policy"] == "adaptive"
    )
    control["every_task_throughput_pass"] = control["worst_task_throughput_loss_fraction"] <= 0.03
    all_gates = [control["throughput_pass"], control["every_task_throughput_pass"]]
    for split in ("development", "held_out"):
        all_gates.extend(value for key, value in splits[split]["vs_split_best_fixed"].items()
                         if key.endswith("_pass"))
    return {"version": VERSION, "units": "synthetic resource ticks, not inference tokens",
            "tasks_per_split": 30, "selected_fixed_on_development": selected_fixed,
            "fixture_sha256": digest_json(fixtures), "fixtures": fixtures, "splits": splits,
            "simulation_all_thresholds_pass": all(all_gates),
            "real_agent_hypothesis_corroborated": False}


def render_report(result: dict) -> str:
    """Render measured thresholds and explicit simulation limitations."""
    lines = ["# MISSION-050 measured simulation results", "",
             "Synthetic experiment only. It does **not corroborate real-agent behaviour**,",
             "inference-token overhead, real context recovery, or actual vendor handoffs.", "",
             f"Version: `{VERSION}`. 30 tasks per split; 1,200 ticks per task; 18 policies;",
             "1,620 policy/task trials. Parameters and predictions were frozen in `design.md`.",
             f"Fixture SHA-256: `{result['fixture_sha256']}`.", "",
             f"Development-selected fixed policy: `{result['selected_fixed_on_development']}`.",
             "Adaptive parameters were not tuned on either split. Held-out best-fixed selection",
             "below is a retrospective, stronger comparator; its results never feed policy decisions.", "",
             "| Split | Adaptive verified | Best fixed (verified) | Gain vs split best (>=20%) | Gain vs dev selected |",
             "|---|---:|---|---:|---:|"]
    for split in ("development", "held_out", "control"):
        data = result["splits"][split]
        summary = data["summaries"]
        best = data["best_fixed_retrospective"]
        gain = data["vs_split_best_fixed"]["verified_subgoal_gain"]
        dev_gain = data["vs_development_selected"]["verified_subgoal_gain"]
        lines.append(f"| {split} | {summary['adaptive']['verified_subgoals']} | {best} ({summary[best]['verified_subgoals']}) | {gain:.4%} | {dev_gain:.4%} |")
    lines += ["", "| Split | Acknowledged loss: count / value (<=5%) | Worst task: count / value | Checkpoint overhead (<=10%) | Worst task overhead |",
              "|---|---:|---:|---:|---:|"]
    for split in ("development", "held_out", "control"):
        a = result["splits"][split]["summaries"]["adaptive"]
        lines.append(f"| {split} | {a['loss_count_fraction']:.4%} / {a['loss_value_fraction']:.4%} | {a['worst_task_loss_count_fraction']:.4%} / {a['worst_task_loss_value_fraction']:.4%} | {a['overhead_fraction']:.4%} | {a['worst_task_overhead_fraction']:.4%} |")
    control = result["splits"]["control"]
    lines += ["", f"Control throughput loss versus the best control fixed policy (<=3%): **{control['throughput_loss_fraction']:.4%}**; worst task **{control['worst_task_throughput_loss_fraction']:.4%}**.", "",
              "| Gate | Development | Held-out |", "|---|---|---|"]
    labels = {"improvement_pass": ">=20% verified-subgoal improvement",
              "pooled_loss_pass": "<=5% pooled acknowledged count and value lost",
              "every_task_loss_pass": "<=5% loss on every task (stronger diagnostic)",
              "pooled_overhead_pass": "<=10% pooled synthetic checkpoint overhead",
              "every_task_overhead_pass": "<=10% overhead on every task (stronger diagnostic)"}
    for gate, label in labels.items():
        outcomes = ["PASS" if result["splits"][s]["vs_split_best_fixed"][gate] else "MISS"
                    for s in ("development", "held_out")]
        lines.append(f"| {label} | {' | '.join(outcomes)} |")
    lines += ["", f"All simulation thresholds passed: **{result['simulation_all_thresholds_pass']}**.",
              "Real-agent hypothesis corroborated: **False**. A miss is a miss; no mission manifest was edited.", "",
              "This frozen simulation candidate is falsified when either improvement or loss misses;",
              "the real-agent hypothesis remains untested. The overhead-saving floor can leave",
              "volatile progress exposed across interruptions, so low checkpoint cost alone does",
              "not establish useful recovery. No parameter or seed search followed these results.", "",
              "Metric definitions: verified completions are ordered, independently rehashed receipts in",
              "the final committed snapshot. All volatile completions are acknowledged immediately,",
              "including retries. Lost acknowledgments include interruption rollback and terminal",
              "unpersisted work; pooled ratios use total lost / total acknowledged. Value loss is",
              "reported alongside count loss to prevent low-value completions hiding high-value losses.",
              "Checkpoint overhead includes aborted and final checkpoint ticks / allocated ticks.",
              "Throughput is verified subgoals / identical allocated budget. Partial snapshots survive",
              "atomic successful checkpoints. Interrupted snapshots never replace the previous one.", "",
              "Evidence: `evidence.json.gz` contains raw results, full task/event fixtures, all adaptive",
              "decision observations, summary gates, final receipts and hashes. `execution.json` contains",
              "measured wall time and source/evidence hashes. Reruns reproduce `evidence.json.gz` byte",
              "for byte; execution wall times naturally differ. See `validation.txt` for test output.", "",
              "Limitations: seeded finite synthetic workloads; no confidence interval or statistical",
              "generalization claim; no model execution; no empirically calibrated checkpoint cost;",
              "zero recovery/verification compute cost; perfectly reliable atomic storage; all four",
              "event labels share one rollback model; no outages spanning multiple ticks, receipt",
              "corruption in benchmark fixtures, dependencies beyond sequential ordering, or actual",
              "file-system crash recovery. SHA-256 receipts validate simulator state, not semantic task",
              "correctness. The observation boundary prevents accidental future-data access, not",
              "hostile Python stack introspection. The simulator is not a production guardian hook.", "",
              "Prediction review: compare the frozen expectations in `design.md` with these measured",
              "gates; synthetic cost accounting is not session token telemetry. The supervisor must",
              "handle mission status, evaluation/budget ledgers, APM installation, and integration.", ""]
    return "\n".join(lines)


def main() -> None:
    """Write deterministic benchmark evidence only to an explicit output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = benchmark()
    elapsed = time.perf_counter() - started
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    evidence = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    compressed = gzip.compress(evidence.encode(), mtime=0)
    (output / "evidence.json.gz").write_bytes(compressed)
    summary = {key: value for key, value in result.items() if key not in ("fixtures", "splits")}
    summary["splits"] = {split: {key: value for key, value in data.items() if key != "trials"}
                         for split, data in result["splits"].items()}
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "results.md").write_text(render_report(result), encoding="utf-8")
    execution = {"benchmark_wall_seconds": elapsed,
                 "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
                 "compressed_evidence_sha256": hashlib.sha256(compressed).hexdigest(),
                 "session_tokens": "not self-reported; supervisor-owned telemetry"}
    (output / "execution.json").write_text(json.dumps(execution, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "benchmark_wall_seconds": elapsed,
                      "simulation_all_thresholds_pass": result["simulation_all_thresholds_pass"],
                      "selected_fixed": result["selected_fixed_on_development"]}, sort_keys=True))


if __name__ == "__main__":
    main()
