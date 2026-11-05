"""MISSION-022 tests: offline unit tests (mock model) + gated live evaluation on local MLX.

Run unit tests:   python3 -m pytest .../operating-sanctuary-guardian/tests/test_local_agent_loop.py
Run live eval:    RUN_LIVE_AGENT_EVAL=1 python3 -m pytest ... -k live -s
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).parent.resolve()
SCRIPTS = HERE.parent / "scripts"
REPO = HERE.parents[5]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))

import local_agent_loop as lal  # noqa: E402
import agent_eval_tasks as aet  # noqa: E402

DENY = ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests",
        "LEARNING/research/experiments"]
RESULTS_PATH = REPO / "LEARNING/research/experiments/mission-022/heldout_results.json"


def scripted(*replies):
    """Mock chat_fn that plays back replies (repeating the last one)."""
    state = {"i": 0, "seen": []}

    def chat(messages):
        state["seen"].append(messages)
        r = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        return {"content": r, "prompt_tokens": 100}
    chat.state = state
    return chat


def call(name, **args):
    return '<scratchpad>x</scratchpad>\n<{"name": "%s", "arguments": %s}' % (name, json.dumps(args))


FINAL = "<scratchpad>x</scratchpad><final_answer>done</final_answer>"


@pytest.fixture
def sb():
    return lal.ReadOnlySandbox(REPO, deny=DENY)


# ---- sandbox -------------------------------------------------------------------------
def test_list_dir_marks_directories(sb):
    out = sb.list_dir("LEARNING")
    assert "research/" in out.splitlines() and "README.md" in out.splitlines()


@pytest.mark.parametrize("arg", [".", "/", "", "./"])
def test_root_aliases_resolve_to_repo_root(sb, arg):
    assert "AGENTS.md" in sb.run("list_dir", {"path": arg})


def test_glob_in_list_dir_gives_actionable_error(sb):
    assert "find_files" in sb.list_dir("*[LC]*")


def test_escape_is_blocked(sb):
    assert "ERROR" in sb.run("read_file", {"path": "../../etc/passwd"})
    assert "ERROR" in sb.run("list_dir", {"path": "/etc"}) or "AGENTS.md" not in sb.run("list_dir", {"path": "/etc"})


def test_deny_list_blocks_reads_and_hides_entries(sb):
    assert "ERROR" in sb.run("read_file", {"path": DENY[0] + "/agent_eval_tasks.py"})
    assert "tests/" not in sb.list_dir("plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian").splitlines()
    assert "agent_eval_tasks" not in sb.find_files("**/agent_eval_tasks.py")
    assert "agent_eval_tasks" not in sb.search_text("held-out task generator")


def test_search_text_and_find_files(sb):
    assert "MISSION-021.json" in sb.search_text("Local Tool Use", "LEARNING/missions")
    assert "LEARNING/missions/MISSION-001.json" in sb.find_files("LEARNING/missions/*.json")


def test_search_text_rejects_glob_and_missing_paths(sb):
    assert "ERROR" in sb.search_text("x", "LEARNING/missions/*.json")
    assert "ERROR: path not found" in sb.search_text("x", "no/such/dir")


def test_unknown_tool_and_bad_args_are_observations_not_crashes(sb):
    assert "unknown tool" in sb.run("rm_rf", {})
    assert "bad arguments" in sb.run("read_file", {"nope": 1})


def test_read_file_truncates(sb):
    small = lal.ReadOnlySandbox(REPO, max_chars=50)
    assert "[truncated" in small.read_file("AGENTS.md")


# ---- loop guards ---------------------------------------------------------------------
def test_tool_then_final(sb):
    r = lal.run_agent("t", sb, chat_fn=scripted(call("list_dir", path="context"), FINAL))
    assert r.status == "final" and r.turns == 2 and r.unassisted and r.final_answer == "done"


def test_bare_json_and_tagged_calls_both_parse():
    assert lal.extract_calls('<{"name": "list_dir", "arguments": {"path": "a"}}')[0]["arguments"] == {"path": "a"}
    assert lal.extract_calls('<tool_call>{"name": "list_dir", "arguments": {"path": "b"}}</tool_call>')[0]["arguments"] == {"path": "b"}
    assert lal.extract_calls("no call here {not json}") == []


def test_tag_per_tool_calls_are_accepted():
    """A model that writes <search_text>query</search_text> instead of the JSON form must not burn turns on parse errors."""
    assert lal.extract_calls("<scratchpad>x</scratchpad>\n<search_text>MISSION-021 verdict</search_text>") == [
        {"name": "search_text", "arguments": {"query": "MISSION-021 verdict"}}]
    assert lal.extract_calls("<read_file>LEARNING/missions/MISSION-021.json</read_file>")[0]["arguments"] == {"path": "LEARNING/missions/MISSION-021.json"}
    assert lal.extract_calls('<find_files>{"pattern": "**/*.md"}</find_files>')[0]["arguments"] == {"pattern": "**/*.md"}
    assert lal.extract_calls("<find_files>{not json}</find_files>") == [] and lal.extract_calls("<list_dir></list_dir>") == []


def test_malformed_reply_feedback_shows_an_example(sb):
    chat = scripted("<unknown>x</unknown>", FINAL)
    lal.run_agent("t", sb, chat_fn=chat)
    assert '"name": "search_text"' in chat.state["seen"][1][-1]["content"]


def test_turn_ceiling_is_hard_capped_at_five(sb):
    chat = scripted(*[call("list_dir", path=f"d{i}") for i in range(50)])
    r = lal.run_agent("t", sb, chat_fn=chat, max_turns=99)
    assert r.turns == lal.HARD_MAX_TURNS == 5 and r.status == "turn_limit"


def test_repeated_call_is_suppressed_not_reexecuted(sb):
    calls = []
    sb.run = lambda n, a: calls.append((n, a)) or "obs"
    chat = scripted(call("list_dir", path="x"), call("list_dir", path="x"), FINAL)
    r = lal.run_agent("t", sb, chat_fn=chat)
    assert len(calls) == 1 and r.repeat_interventions == 1 and not r.unassisted
    assert "already made" in chat.state["seen"][2][-1]["content"]


def test_more_than_four_identical_calls_falsifies(sb):
    r = lal.run_agent("t", sb, chat_fn=scripted(call("list_dir", path="x")), max_turns=5)
    assert r.status == "repetition_falsified" and r.max_identical_calls == 5 and r.final_answer is None


def test_final_turn_is_announced_and_flagged(sb):
    chat = scripted(call("list_dir", path="a"), call("list_dir", path="b"), FINAL)
    r = lal.run_agent("t", sb, chat_fn=chat, max_turns=3)
    assert r.final_turn_nudged and r.final_on_nudged_turn and not r.unassisted
    assert "LAST" in chat.state["seen"][2][-1]["content"]


def test_malformed_reply_is_counted_and_recoverable(sb):
    r = lal.run_agent("t", sb, chat_fn=scripted("I think the answer is 4", FINAL))
    assert r.status == "final" and r.malformed_replies == 1 and not r.unassisted


def test_context_budget_aborts(sb):
    chat = lambda m: {"content": FINAL, "prompt_tokens": 99999}  # noqa: E731
    assert lal.run_agent("t", sb, chat_fn=chat, ctx_budget=1000).status == "context_exceeded"


def test_server_error_is_reported_not_raised(sb):
    def boom(m):
        raise OSError("connection refused")
    r = lal.run_agent("t", sb, chat_fn=boom)
    assert r.status == "error"


# ---- generated tasks + graders -------------------------------------------------------
def test_generated_tasks_are_deterministic_and_stratified():
    a, b = aet.build_tasks(REPO), aet.build_tasks(REPO)
    assert a == b and len(a) == 30
    assert {t["tier"] for t in a} == {"scalar", "locate", "dirset"}


def test_graders_reject_extras_negation_and_wrong_values():
    s = {"type": "set", "expected": ["a.md", "b.md"]}
    assert aet.grade(s, "a.md, b.md", REPO)["correct"]
    assert aet.grade(s, "- `a.md`\n- `b.md`", REPO)["correct"]
    assert not aet.grade(s, "a.md, b.md, c.md", REPO)["correct"]          # extra item
    assert not aet.grade(s, "a.md", REPO)["correct"]                      # missing item
    sc = {"type": "scalar", "field": "status", "expected": "PROPOSED"}
    assert aet.grade(sc, "PROPOSED", REPO)["correct"]
    assert not aet.grade(sc, "PROPOSED or RESOLVED", REPO)["correct"]     # hedge
    assert not aet.grade(sc, None, REPO)["correct"]
    num = {"type": "scalar", "field": "priority_score", "expected": "200.0"}
    assert aet.grade(num, "200", REPO)["correct"] and not aet.grade(num, "400.0", REPO)["correct"]
    loc = {"type": "scalar", "field": "id", "expected": "MISSION-007"}
    assert aet.grade(loc, "MISSION-007", REPO)["correct"] and not aet.grade(loc, "MISSION-007 and MISSION-008", REPO)["correct"]


def test_agent_cannot_reach_answer_key(sb):
    assert "ERROR" in sb.run("read_file", {"path": f"{DENY[0]}/test_local_agent_loop.py"})


# ---- live evaluation (gated) ---------------------------------------------------------
def _mlx_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(os.environ.get("RUN_LIVE_AGENT_EVAL") != "1" or not _mlx_up(),
                    reason="set RUN_LIVE_AGENT_EVAL=1 with the MLX server on :8080 (takes ~10 min)")
def test_live_heldout_discovery_completion_rate():
    """Pre-registered gate: >=85% exactly-correct on 30 generated held-out tasks; no repetition falsification."""
    temp = float(os.environ.get("AGENT_EVAL_TEMP", "0.0"))
    seed = int(os.environ.get("AGENT_EVAL_SEED", "1337"))
    sandbox = lal.ReadOnlySandbox(REPO, deny=DENY)
    rows = []
    for t in aet.build_tasks(REPO, seed=seed):
        res = lal.run_agent(t["prompt"], sandbox, chat_fn=lambda m: lal.mlx_chat(m, temperature=temp), max_turns=5)
        g = aet.grade(t, res.final_answer, REPO)
        rows.append({"id": t["id"], "tier": t["tier"], "expected": t["expected"], "grade": g, **res.to_dict()})
        print(f"{t['id']:>4} correct={g['correct']!s:5} status={res.status:<20} turns={res.turns} "
              f"unassisted={res.unassisted} {g['reason'][:70]}")
    n = len(rows)
    by_tier = {k: sum(r["grade"]["correct"] for r in rows if r["tier"] == k) / sum(r["tier"] == k for r in rows)
               for k in ("scalar", "locate", "dirset")}
    summary = {
        "temperature": temp, "seed": seed, "n": n,
        "correct_rate": sum(r["grade"]["correct"] for r in rows) / n,
        "completion_rate": sum(r["status"] == "final" for r in rows) / n,
        "unassisted_correct_rate": sum(r["grade"]["correct"] and r["unassisted"] for r in rows) / n,
        "correct_within_3_turns": sum(r["grade"]["correct"] and r["turns"] <= 3 for r in rows) / n,
        "repetition_falsified": sum(r["status"] == "repetition_falsified" for r in rows),
        "context_exceeded": sum(r["status"] == "context_exceeded" for r in rows),
        "repeat_interventions": sum(r["repeat_interventions"] for r in rows),
        "tasks_with_final_turn_nudge": sum(r["final_turn_nudged"] for r in rows),
        "by_tier": by_tier, "max_prompt_tokens": max(r["max_prompt_tokens"] for r in rows),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
    }
    out = RESULTS_PATH.with_name(f"heldout_results_seed{seed}_temp{temp}.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    assert summary["repetition_falsified"] == 0
    assert summary["correct_rate"] >= 0.85
