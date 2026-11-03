#!/usr/bin/env python3
"""
test_model_routing.py - Pytest suite for cheapest-viable model routing (MISSION-016)
===================================================================================

Purpose:
    Verifies recommend_model_routing defaults to the cheap/local tier, keeps
    cost weights <= 0.2 for exploratory/formatting/inspection work, gates
    frontier escalation on failures >= 2, populates model_routing during
    capture and compile, exposes the `route` CLI, and checks that
    evaluate_dispatch returns DOWNGRADE_REQUIRED (failures < 2) or APPROVED
    (failures >= 2) for frontier dispatches.

Key Input Dependencies:
    manage_backlog.py and ../../evaluating-agent-work/scripts/evaluate_dispatch.py
    (sibling skill). Uses pytest tmp_path roots; does not touch LEARNING/ or
    .agent/ in the real repository.

Usage:
    python3 -m pytest plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/test_model_routing.py -q

Function Index:
    _load_dispatch_module, _dispatch_args, test_* (see below)
"""

import sys
import json
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
DISPATCH_PATH = SCRIPTS_DIR.parents[1] / "evaluating-agent-work" / "scripts" / "evaluate_dispatch.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import manage_backlog as mb  # noqa: E402

CHEAP_OR_LOCAL = {"gemini-3.8-flash", "claude-haiku-5.5", "qwen2-7b-mlx-4bit"}
EXPLORATORY_TYPES = ["exploration", "formatting", "inspection", "test", "verification", "summarization"]
HYPOTHESIS = "File-based JSON writes in the procedure store are prone to corruption under concurrent execution"
FALSIFICATION = "Ten simultaneous processes write to the same procedure file without losing count or corrupting JSON"


def _load_dispatch_module() -> ModuleType:
    """Imports evaluate_dispatch.py from the sibling evaluating-agent-work skill."""
    spec = importlib.util.spec_from_file_location("evaluate_dispatch", DISPATCH_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dispatch = _load_dispatch_module()


def _capture_main_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture, argv: List[str]) -> Dict[str, Any]:
    """Runs evaluate_dispatch.main in JSON mode with its project root redirected to tmp_path; returns the record."""
    monkeypatch.setattr(dispatch, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluate_dispatch.py", *argv, "--format", "json"])
    dispatch.main()
    return json.loads(capsys.readouterr().out)


def _route_cli(*args: str) -> subprocess.CompletedProcess:
    """Invokes `manage_backlog.py route` as a subprocess."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "manage_backlog.py"), "route", *args],
        capture_output=True, text=True, check=False,
    )


# --- Default routing to the cheap / local tier -------------------------------

def test_default_routing_is_cheap_gemini_flash() -> None:
    result = mb.recommend_model_routing("exploration", "map the memory layer modules")
    assert result["cheapest_viable_tier"] == "cheap"
    assert result["recommended_model"] == "gemini-3.8-flash"
    assert result["cost_weight"] == 0.1


def test_local_inspection_routes_to_free_local_mlx() -> None:
    result = mb.recommend_model_routing("inspection", "verify the schema file")
    assert result["cheapest_viable_tier"] == "free-local"
    assert result["recommended_model"] == "qwen2-7b-mlx-4bit"
    assert result["cost_weight"] == 0.05
    assert result["cli_target"] == "mlx-lm"


def test_local_detected_from_description_alone() -> None:
    result = mb.recommend_model_routing("general", "run the offline lint pass")
    assert result["cheapest_viable_tier"] == "free-local"


def test_fallbacks_are_only_cheap_or_local_models_below_escalation() -> None:
    for task_type in EXPLORATORY_TYPES:
        result = mb.recommend_model_routing(task_type, "", failures=0)
        assert result["recommended_model"] in CHEAP_OR_LOCAL
        assert set(result["fallback_models"]) <= CHEAP_OR_LOCAL
        assert result["recommended_model"] not in result["fallback_models"]


# --- Cost weights stay <= 0.2 for exploratory / formatting / inspection ------

@pytest.mark.parametrize("task_type", EXPLORATORY_TYPES)
@pytest.mark.parametrize("failures", [0, 1])
def test_cost_weight_at_most_0_2_for_cheap_work(task_type: str, failures: int) -> None:
    result = mb.recommend_model_routing(task_type, "", failures=failures)
    assert result["cost_weight"] <= 0.2
    assert result["cheapest_viable_tier"] in {"cheap", "free-local"}


def test_routing_weights_match_evaluate_dispatch_cost_tiers() -> None:
    for model, weight in mb.ROUTING_WEIGHTS.items():
        assert dispatch.get_model_tier(model)["weight"] == weight, model


# --- Escalation gate ----------------------------------------------------------

@pytest.mark.parametrize("failures", [0, 1])
def test_frontier_blocked_below_two_failures(failures: int) -> None:
    result = mb.recommend_model_routing("architecture", "design the routing layer", failures=failures)
    assert result["cheapest_viable_tier"] != "frontier"
    assert result["cost_weight"] <= 0.2


@pytest.mark.parametrize("failures", [2, 5])
def test_frontier_permitted_at_two_or_more_failures(failures: int) -> None:
    result = mb.recommend_model_routing("architecture", "design the routing layer", failures=failures)
    assert result["cheapest_viable_tier"] == "frontier"
    assert result["recommended_model"] in {"claude-opus", "gpt-6-astra"}
    assert result["cost_weight"] == 4.0


def test_negative_failures_are_clamped_to_zero() -> None:
    result = mb.recommend_model_routing("exploration", "", failures=-3)
    assert result["cheapest_viable_tier"] == "cheap"


# --- BacklogManager auto-population ------------------------------------------

def test_capture_inbox_attaches_model_routing(tmp_path: Path) -> None:
    manager = mb.BacklogManager(root_dir=tmp_path)
    entry = manager.capture_inbox("scan the docs for stale links", tags=["inspection"])
    assert entry["model_routing"]["cheapest_viable_tier"] == "free-local"
    assert entry["model_routing"]["recommended_model"] == "qwen2-7b-mlx-4bit"


def test_compile_populates_missing_model_routing(tmp_path: Path) -> None:
    manager = mb.BacklogManager(root_dir=tmp_path)
    manager.save_mission({
        "id": "MISSION-901", "title": "Stale link sweep", "status": "PROPOSED",
        "hypothesis": "Stale links can be enumerated with a local scan of the docs tree",
        "falsification_criteria": "A scan misses a broken link present in the docs tree",
        "task_type": "inspection",
        "scores": {"strategic_leverage": 3, "epistemic_depth": 2, "cost_weight": 0.05, "operational_friction": 1},
    })
    manager.compile()
    stored = manager.get_mission("MISSION-901")
    assert stored is not None
    routing = stored["model_routing"]
    assert routing["recommended_model"] == "qwen2-7b-mlx-4bit"
    assert routing["cheapest_viable_tier"] == "free-local"
    assert routing["fallback_models"]
    index_row = json.loads((tmp_path / "LEARNING" / "research" / "index.jsonl").read_text().splitlines()[0])
    assert index_row["recommended_model"] == "qwen2-7b-mlx-4bit"


def test_compile_preserves_existing_routing_values(tmp_path: Path) -> None:
    manager = mb.BacklogManager(root_dir=tmp_path)
    manager.save_mission({
        "id": "MISSION-902", "title": "Custom routed", "status": "PROPOSED",
        "hypothesis": "A hand-chosen model should survive compilation untouched",
        "falsification_criteria": "Compilation overwrites the hand-chosen recommended model",
        "model_routing": {"recommended_model": "claude-haiku-5.5", "cheapest_viable_tier": "cheap"},
        "scores": {"strategic_leverage": 2, "epistemic_depth": 2, "cost_weight": 0.2, "operational_friction": 1},
    })
    manager.compile()
    routing = manager.get_mission("MISSION-902")["model_routing"]
    assert routing["recommended_model"] == "claude-haiku-5.5"
    assert routing["cost_weight"] == 0.2
    assert "fallback_models" in routing and "cli_target" in routing


# --- `route` CLI --------------------------------------------------------------

def test_cli_route_returns_valid_json() -> None:
    proc = _route_cli("Summarize the changelog", "--type", "summarization", "--format", "json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["cheapest_viable_tier"] == "cheap"
    assert data["recommended_model"] == "gemini-3.8-flash"


def test_cli_route_defaults_to_json() -> None:
    proc = _route_cli("Check the lint status")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["cheapest_viable_tier"] == "free-local"


def test_cli_route_escalates_with_failures() -> None:
    proc = _route_cli("Diagnose the deadlock", "--failures", "2", "--format", "json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["cheapest_viable_tier"] == "frontier"
    assert data["cost_weight"] == 4.0


def test_cli_route_text_output() -> None:
    proc = _route_cli("Format the CSV", "--type", "formatting", "--format", "text")
    assert proc.returncode == 0, proc.stderr
    assert "Recommended model:  gemini-3.8-flash" in proc.stdout
    assert "Tier:" in proc.stdout


# --- evaluate_dispatch escalation verdicts -----------------------------------

def test_evaluate_frontier_downgrade_required_below_two_failures() -> None:
    result = dispatch.evaluate_investment(HYPOTHESIS, FALSIFICATION, dispatch.get_model_tier("claude-opus"), 1, failures=1)
    assert result["status"] == "DOWNGRADE_REQUIRED"
    assert any("failures < 2" in r for r in result["recommendations"])


def test_evaluate_frontier_approved_at_two_failures() -> None:
    result = dispatch.evaluate_investment(HYPOTHESIS, FALSIFICATION, dispatch.get_model_tier("gpt-6-astra"), 1, failures=2)
    assert result["status"] == "APPROVED"
    assert "Frontier escalation approved after 2 failed attempts." in result["recommendations"]


def test_evaluate_falsifiability_failure_not_overridden_by_escalation() -> None:
    result = dispatch.evaluate_investment(HYPOTHESIS, "vague", dispatch.get_model_tier("claude-opus"), 1, failures=3)
    assert result["status"] == "NEEDS_REFINEMENT"


def test_cli_dispatch_downgrade_required_when_failures_below_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    record = _capture_main_json(monkeypatch, tmp_path, capsys, [
        "--task", "Architecture review", "--hypothesis", HYPOTHESIS,
        "--falsification-criteria", FALSIFICATION, "--model", "claude-opus",
        "--planned-turns", "1", "--failures", "0",
    ])
    assert record["investment_decision"] == "DOWNGRADE_REQUIRED"


def test_cli_dispatch_approved_when_failures_at_least_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    record = _capture_main_json(monkeypatch, tmp_path, capsys, [
        "--task", "Architecture review", "--hypothesis", HYPOTHESIS,
        "--falsification-criteria", FALSIFICATION, "--model", "claude-opus",
        "--planned-turns", "1", "--failures", "2",
    ])
    assert record["investment_decision"] == "APPROVED"
    assert record["cost_weight"] == 4.0
