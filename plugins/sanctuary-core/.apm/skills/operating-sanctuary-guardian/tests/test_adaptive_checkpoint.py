"""MISSION-050: causal policy isolation, metered atomic recovery, and honest gates.

Run with PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider
plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_adaptive_checkpoint.py
--basetemp=LEARNING/experiments/mission-050/pytest-tmp
"""

import ast
from dataclasses import FrozenInstanceError, fields
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import adaptive_checkpoint as ac


def tiny_task(budget=12, cost=2, work=2, value=3):
    """Construct a hand-computable workload independent of random fixtures."""
    return ac.Task(7, tuple(ac.Subgoal(work, value) for _ in range(20)), cost, budget)


def test_receipts_reject_corruption_reordering_duplicate_and_excess():
    task = tiny_task()
    valid = tuple((i, ac.receipt(task, i)) for i in range(2))
    assert ac.verify_receipts(task, valid)
    assert not ac.verify_receipts(task, ((0, "forged"),))
    assert not ac.verify_receipts(task, tuple(reversed(valid)))
    assert not ac.verify_receipts(task, (valid[0], valid[0]))
    assert not ac.verify_receipts(task, valid * 11)


def test_end_only_final_flush_is_metered_and_independently_verified():
    result = ac.run_trial(tiny_task(), (), ac.FixedPolicy("end"))
    # Work at t=1..10; metered final flush at t=11..12.
    assert result["verified_subgoals"] == 5
    assert result["checkpoint_ticks"] == 2
    assert result["work_ticks"] == 10
    assert result["lost_acknowledged_subgoals"] == 0
    assert result["acknowledged_subgoals"] == 5
    assert result["elapsed"] == 12
    assert result["verified_receipts_sha256"] == ac.digest_json(
        tuple((i, ac.receipt(tiny_task(), i)) for i in range(5))
    )


def test_interruption_on_checkpoint_commit_tick_keeps_previous_snapshot():
    task = tiny_task(budget=8)
    policy = ac.FixedPolicy("steps", 1)
    clean = ac.run_trial(task, (), policy)
    crashed = ac.run_trial(task, (ac.Event(4, "worker_termination"),), policy)
    assert clean["verified_subgoals"] == 2
    assert crashed["verified_subgoals"] == 1
    assert crashed["lost_acknowledged_subgoals"] == 1
    assert crashed["checkpoint_attempts"] == 2
    assert crashed["checkpoint_commits"] == 1
    assert crashed["checkpoint_ticks"] == 4


def test_interruption_mid_checkpoint_is_charged_and_retry_is_not_free():
    result = ac.run_trial(tiny_task(budget=8), (ac.Event(3, "context_reset"),),
                          ac.FixedPolicy("steps", 1))
    assert result["checkpoint_ticks"] == 3
    assert result["checkpoint_attempts"] == 2
    assert result["checkpoint_commits"] == 1
    assert result["lost_acknowledged_subgoals"] == 1


def test_saved_partial_work_survives_interruption():
    task = tiny_task(budget=12, cost=1, work=6)
    result = ac.run_trial(task, (ac.Event(5, "quota_denial"),), ac.FixedPolicy("time", 3))
    # t1..3 partial work saved at t4, interruption t5 rolls back one work tick.
    assert result["lost_partial_work"] == 1
    assert result["verified_subgoals"] == 1
    assert result["lost_acknowledged_subgoals"] == 0


def test_no_free_final_save_when_budget_cannot_afford_checkpoint():
    result = ac.run_trial(tiny_task(budget=3, cost=4, work=1), (), ac.FixedPolicy("end"))
    assert result["verified_subgoals"] == 0
    assert result["lost_acknowledged_subgoals"] == 1
    assert result["checkpoint_ticks"] == 0
    assert result["elapsed"] == 1


def test_retries_obey_acknowledgment_conservation():
    events = (ac.Event(2, "context_reset"), ac.Event(4, "vendor_handoff"))
    result = ac.run_trial(tiny_task(budget=10), events, ac.FixedPolicy("end"))
    assert result["acknowledged_subgoals"] == (
        result["verified_subgoals"] + result["lost_acknowledged_subgoals"]
    )
    assert result["acknowledged_value"] == (
        result["verified_value"] + result["lost_acknowledged_value"]
    )
    assert result["lost_acknowledged_subgoals"] >= 2


def test_adaptive_input_boundary_is_immutable_value_only():
    assert [field.name for field in fields(ac.AdaptiveObservation)] == [
        "interruption_rate", "pending_value", "value_rate", "checkpoint_cost"
    ]
    observed = ac.AdaptiveObservation(0.1, 50, 1, 2)
    with pytest.raises(FrozenInstanceError):
        observed.pending_value = 0
    with pytest.raises((AttributeError, TypeError)):
        observed.schedule = (1, 2, 3)
    assert not hasattr(observed, "__dict__")
    assert list(inspect.signature(ac.AdaptivePolicy.should_checkpoint).parameters) == ["self", "observed"]
    # The policy's AST can read only observation fields, math.sqrt and constants.
    tree = ast.parse(inspect.getsource(ac))
    policy_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AdaptivePolicy")
    attributes = {(n.value.id, n.attr) for n in ast.walk(policy_node)
                  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    assert attributes <= {("observed", f.name) for f in fields(ac.AdaptiveObservation)} | {("math", "sqrt")}
    loads = {n.id for n in ast.walk(policy_node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert loads <= {"dataclass", "str", "bool", "AdaptiveObservation", "observed", "math", "max", "threshold"}


def test_identical_pasts_produce_identical_decisions_despite_different_futures():
    task = tiny_task(budget=100)
    common = (ac.Event(8, "context_reset"),)
    a = ac.run_trial(task, common, ac.AdaptivePolicy(), trace=True)
    b = ac.run_trial(task, (*common, ac.Event(60, "vendor_handoff")), ac.AdaptivePolicy(), trace=True)
    assert [d for d in a["decisions"] if d["tick"] < 60] == [d for d in b["decisions"] if d["tick"] < 60]


def test_no_observed_interruptions_means_no_adaptive_request():
    result = ac.run_trial(ac.make_task(30000), (), ac.AdaptivePolicy(), trace=True)
    assert all(not decision["requested"] for decision in result["decisions"])
    assert result["checkpoint_attempts"] == 1
    assert result["observed_interruptions"] == 0
    assert result["overhead_fraction"] <= 0.01


def test_adaptive_responds_to_hazard_value_and_cost():
    policy = ac.AdaptivePolicy()
    assert not policy.should_checkpoint(ac.AdaptiveObservation(0, 100, 1, 2))
    assert not policy.should_checkpoint(ac.AdaptiveObservation(0.1, 17, 1, 2))
    assert policy.should_checkpoint(ac.AdaptiveObservation(0.1, 18, 1, 2))
    assert not policy.should_checkpoint(ac.AdaptiveObservation(0.1, 18, 1, 3))
    assert not policy.should_checkpoint(ac.AdaptiveObservation(0.0001, 18, 1, 2))


@pytest.mark.parametrize("observation", [(-1, 1, 1, 1), (0, float("nan"), 1, 1),
                                        (0, 1, float("inf"), 1), (0, 1, 1, 0)])
def test_invalid_observations_are_rejected(observation):
    with pytest.raises(ValueError):
        ac.AdaptiveObservation(*observation)


@pytest.mark.parametrize("events", [(ac.Event(0, "context_reset"),),
                                     (ac.Event(13, "context_reset"),),
                                     (ac.Event(2, "unknown"),),
                                     (ac.Event(2, "context_reset"), ac.Event(2, "quota_denial"))])
def test_invalid_schedules_are_rejected(events):
    with pytest.raises(ValueError):
        ac.run_trial(tiny_task(), events, ac.AdaptivePolicy())


def test_all_event_labels_have_the_declared_same_rollback_semantics():
    results = [ac.run_trial(tiny_task(), (ac.Event(4, kind),), ac.FixedPolicy("steps", 1))
               for kind in ac.EVENT_KINDS]
    assert all(result == results[0] for result in results)


def test_fixed_policy_family_includes_step_time_and_end_baselines():
    assert len(ac.fixed_family()) == 17
    assert {p.mode for p in ac.fixed_family()} == {"steps", "time", "end"}
    assert ac.FixedPolicy("time", 8).should_checkpoint(ac.FixedObservation(8, 0))
    assert not ac.FixedPolicy("time", 8).should_checkpoint(ac.FixedObservation(7, 100))
    assert ac.FixedPolicy("steps", 2).should_checkpoint(ac.FixedObservation(0, 2))
    assert not ac.FixedPolicy("end").should_checkpoint(ac.FixedObservation(10000, 10000))


def test_deterministic_seeded_trials_and_full_budget_accounting():
    task = ac.make_task(10000)
    events = ac.make_schedule(20000, task.budget, 5)
    first = ac.run_trial(task, events, ac.AdaptivePolicy(), trace=True)
    assert first == ac.run_trial(ac.make_task(10000), ac.make_schedule(20000, task.budget, 5),
                                 ac.AdaptivePolicy(), trace=True)
    assert first["work_ticks"] + first["checkpoint_ticks"] == first["elapsed"]
    assert first["elapsed"] <= task.budget
    assert first["checkpoint_ticks"] <= first["checkpoint_attempts"] * task.checkpoint_cost


def test_pooled_metrics_and_gates_do_not_hide_high_value_losses():
    trial = ac.run_trial(tiny_task(), (), ac.FixedPolicy("end"))
    summary = ac.summarize([trial, trial])
    assert summary["verified_subgoals"] == 2 * trial["verified_subgoals"]
    assert summary["loss_count_fraction"] == trial["loss_count_fraction"]
    baseline = {"verified_subgoals": 100}
    adaptive = dict(summary, verified_subgoals=120, loss_count_fraction=0.01,
                    loss_value_fraction=0.051, overhead_fraction=0.10)
    gates = ac.compare(adaptive, baseline)
    assert gates["improvement_pass"]  # Exactly 20%, without float subtraction error.
    assert gates["pooled_overhead_pass"]
    assert not gates["pooled_loss_pass"]
    assert not ac.compare(adaptive, {"verified_subgoals": 0})["improvement_pass"]


def test_best_fixed_ignores_adaptive_and_breaks_ties_by_value_then_cost():
    summaries = {
        "adaptive": {"verified_subgoals": 999},
        "fixed-a": {"verified_subgoals": 5, "verified_value": 8, "checkpoint_ticks": 5},
        "fixed-b": {"verified_subgoals": 5, "verified_value": 9, "checkpoint_ticks": 8},
        "fixed-c": {"verified_subgoals": 5, "verified_value": 9, "checkpoint_ticks": 7},
    }
    assert ac.best_fixed(summaries) == "fixed-c"


def test_recorded_evidence_has_disjoint_splits_and_independently_valid_receipts():
    """Audit persisted evidence without trusting its claimed receipt count or hash."""
    root = Path(__file__).resolve().parents[6]
    artifact = root / "LEARNING/experiments/mission-050/evidence.json.gz"
    if not artifact.exists():
        pytest.skip("run the frozen benchmark to generate evidence first")
    with gzip.open(artifact, "rt", encoding="utf-8") as stream:
        evidence = json.load(stream)
    assert evidence["fixture_sha256"] == ac.digest_json(evidence["fixtures"])
    dev_seeds = {f["task"]["seed"] for f in evidence["fixtures"]["development"]}
    held_seeds = {f["task"]["seed"] for f in evidence["fixtures"]["held_out"]}
    assert len(dev_seeds) == len(held_seeds) == 30
    assert dev_seeds.isdisjoint(held_seeds)
    assert {f["schedule_seed"] for f in evidence["fixtures"]["development"]}.isdisjoint(
        {f["schedule_seed"] for f in evidence["fixtures"]["held_out"]}
    )
    assert all(not f["events"] for f in evidence["fixtures"]["control"])
    assert [f["task"] for f in evidence["fixtures"]["control"]] == [
        f["task"] for f in evidence["fixtures"]["held_out"]
    ]
    for split, data in evidence["splits"].items():
        assert len(data["trials"]) == 30 * 18
        fixtures = {f["task"]["seed"]: f for f in evidence["fixtures"][split]}
        for trial in data["trials"]:
            task = fixtures[trial["task_seed"]]["task"]
            receipts = trial["verified_receipts"]
            assert len(receipts) == trial["verified_subgoals"]
            verified_value = 0
            for expected, (index, digest) in enumerate(receipts):
                assert index == expected
                goal = task["goals"][index]
                payload = f"{evidence['version']}:{task['seed']}:{index}:{goal['work']}:{goal['value']}"
                assert digest == hashlib.sha256(payload.encode("ascii")).hexdigest()
                verified_value += goal["value"]
            assert verified_value == trial["verified_value"]
            assert ac.digest_json(receipts) == trial["verified_receipts_sha256"]
            assert trial["acknowledged_subgoals"] == len(receipts) + trial["lost_acknowledged_subgoals"]
            assert trial["acknowledged_value"] == verified_value + trial["lost_acknowledged_value"]
            assert trial["work_ticks"] + trial["checkpoint_ticks"] == trial["elapsed"] <= trial["budget"]
        for name, summary in data["summaries"].items():
            assert ac.summarize([t for t in data["trials"] if t["policy"] == name]) == summary
        winner = ac.best_fixed(data["summaries"])
        assert data["best_fixed_retrospective"] == winner
        assert data["vs_split_best_fixed"] == ac.compare(data["summaries"]["adaptive"],
                                                        data["summaries"][winner])
    assert evidence["selected_fixed_on_development"] == ac.best_fixed(
        evidence["splits"]["development"]["summaries"]
    )
    assert evidence["real_agent_hypothesis_corroborated"] is False


def test_recorded_adaptive_trials_replay_exactly_from_frozen_fixtures():
    """Replay every adaptive trial, not only a favorable or randomly chosen case."""
    root = Path(__file__).resolve().parents[6]
    artifact = root / "LEARNING/experiments/mission-050/evidence.json.gz"
    if not artifact.exists():
        pytest.skip("run the frozen benchmark to generate evidence first")
    with gzip.open(artifact, "rt", encoding="utf-8") as stream:
        evidence = json.load(stream)
    for split, fixtures in evidence["fixtures"].items():
        trials = {t["task_seed"]: t for t in evidence["splits"][split]["trials"]
                  if t["policy"] == "adaptive"}
        for fixture in fixtures:
            task_data = fixture["task"]
            task = ac.Task(task_data["seed"], tuple(ac.Subgoal(**g) for g in task_data["goals"]),
                           task_data["checkpoint_cost"], task_data["budget"])
            result = ac.run_trial(task, tuple(ac.Event(**e) for e in fixture["events"]),
                                  ac.AdaptivePolicy(), trace=True)
            # JSON round-trip normalizes tuples into persisted lists.
            assert json.loads(json.dumps(result)) == trials[task.seed]


def test_v2_prior_requests_checkpoint_without_observed_interruption():
    """Pseudo-count prior fixes cold start while preserving v1 replay semantics."""
    task = tiny_task(budget=100, cost=1, work=2, value=3)
    v1 = ac.run_trial(task, (), ac.AdaptivePolicy(), trace=True)
    v2 = ac.run_trial(task, (), ac.AdaptivePolicyV2(), trace=True)
    assert not any(d['requested'] for d in v1['decisions'])
    assert any(d['requested'] and not d['final_flush'] for d in v2['decisions'])
    assert v2['decisions'][0]['observation']['interruption_rate'] == 1 / 40
    assert v2['observed_interruptions'] == 0
    assert v2['checkpoint_attempts'] > v1['checkpoint_attempts']


def test_v2_identical_past_has_identical_decisions_with_prior():
    task = tiny_task(budget=100)
    common = (ac.Event(8, 'context_reset'),)
    a = ac.run_trial(task, common, ac.AdaptivePolicyV2(), trace=True)
    b = ac.run_trial(task, (*common, ac.Event(60, 'vendor_handoff')), ac.AdaptivePolicyV2(), trace=True)
    assert [d for d in a['decisions'] if d['tick'] < 60] == [d for d in b['decisions'] if d['tick'] < 60]


@pytest.mark.parametrize('prior', [(0, 40), (1, 0), (-1, 40), (float('nan'), 40)])
def test_v2_rejects_invalid_pseudocount_prior(prior):
    with pytest.raises(ValueError):
        ac.run_trial(tiny_task(), (), ac.AdaptivePolicyV2(), hazard_prior=prior)
