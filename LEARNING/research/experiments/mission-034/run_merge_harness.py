#!/usr/bin/env python3
"""60 two-branch merges, each branch rewriting every shared generated file. Compares plain `git merge`
(baseline) with merge + merge_prep.py regenerate (treatment). Scratch repo only; never touches the real repo."""
import json, shutil, subprocess, sys, tempfile, time, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SK = REPO / "plugins/sanctuary-core/.apm/skills"
sys.path.insert(0, str(SK / "operating-sanctuary-guardian/scripts"))
import merge_prep  # noqa: E402
import score_contribution  # noqa: E402
from manage_backlog import BacklogManager  # noqa: E402

N = 60
GEN = {"LEARNING/research/index.jsonl", "LEARNING/backlog.md", ".agent/learning/cognitive_ledger.json"}


def git(cwd, *a, check=True):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=check)


def seed(union: bool) -> Path:
    root = Path(tempfile.mkdtemp(prefix="m34-"))
    git(root, "init", "-q", "-b", "main"); git(root, "config", "user.email", "t@t"); git(root, "config", "user.name", "t")
    shutil.copytree(REPO / "LEARNING/missions", root / "LEARNING/missions")
    (root / ".agent/learning").mkdir(parents=True); (root / ".agent/coordination").mkdir(parents=True)
    ev = [json.loads(l) for l in (REPO / ".agent/learning/events.jsonl").read_text().splitlines() if l.strip()]
    (root / ".agent/learning/events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    (root / ".agent/coordination/leases.jsonl").write_text('{"seed": true}\n')
    if union:
        (root / ".gitattributes").write_text("".join(f"{p} merge=union\n" for p in merge_prep.UNION_LOGS))
    merge_prep.regenerate(root)  # stable base (also makes the ledger consistent with its events)
    git(root, "add", "-A"); git(root, "commit", "-qm", "base")
    return root


def work(root: Path, tag: str, i: int):
    tpl = json.loads((root / "LEARNING/missions/MISSION-034.json").read_text())
    mid = f"MISSION-{900 + (i if tag == 'a' else 100 + i)}"
    tpl["id"] = mid; tpl["title"] = f"synthetic {tag}{i}"
    (root / f"LEARNING/missions/{mid}.json").write_text(json.dumps(tpl, indent=2))
    BacklogManager(root).compile()
    a = {"event_type": "CONTRIBUTION_ASSESSED", "assessment_id": f"{tag}{i}", "timestamp": f"2026-11-05T00:{i % 60:02d}:{'0' if tag == 'a' else '3'}0+00:00",
         "task_id": mid, "agent_id": f"agent-{tag}", "raw_score": 10 + (i % 5), "cognitive_fitness": 0.7, "notes": "",
         "dimensions": {"useful_outcome": 2, "learning_depth": 2, "verification_rigor": 2, "token_economy": 2, "anti_asch_integrity": 2}}
    with open(root / ".agent/learning/events.jsonl", "a") as f: f.write(json.dumps(a) + "\n")
    score_contribution.update_ledger_projection(root, a)
    with open(root / ".agent/coordination/leases.jsonl", "a") as f: f.write(json.dumps({"lease": f"{tag}{i}"}) + "\n")
    git(root, "add", "-A"); git(root, "commit", "-qm", f"{tag}{i}")


def digest(root, rels):
    return {r: hashlib.sha256((root / r).read_bytes()).hexdigest() for r in rels}


def run(mode: str):
    root = seed(union=(mode == "treatment"))
    res = dict(mode=mode, merges=N, conflicted=0, manual=0, not_identical=0, over_2s=0, times=[], conflict_files={}, marker_left=0, manifests_changed=0)
    for i in range(N):
        git(root, "checkout", "-q", "-b", "A", "main"); work(root, "a", i)
        git(root, "checkout", "-q", "-b", "B", "main"); work(root, "b", i)
        m = git(root, "merge", "--no-edit", "A", check=False)
        conflicted = [l for l in git(root, "diff", "--name-only", "--diff-filter=U").stdout.split() if l]
        if conflicted:
            res["conflicted"] += 1
            for c in conflicted: res["conflict_files"][c] = res["conflict_files"].get(c, 0) + 1
        if mode == "treatment":
            t0 = time.perf_counter(); rep = merge_prep.regenerate(root); dt = time.perf_counter() - t0
            res["times"].append(dt); res["over_2s"] += dt > 2
            res["manifests_changed"] += bool(rep["manifests_changed"])
            if rep["drift"]: res["manual"] += 1
            if set(conflicted) - GEN: res["manual"] += 1  # a conflict outside the regenerable set needs a human
            for c in conflicted:
                if c in GEN: git(root, "add", c)
            # from-scratch regeneration in a copy must equal what merge_prep left
            scratch = Path(tempfile.mkdtemp(prefix="m34s-"))
            shutil.copytree(root / "LEARNING", scratch / "LEARNING"); shutil.copytree(root / ".agent", scratch / ".agent")
            for g in GEN: (scratch / g).unlink(missing_ok=True)
            merge_prep.regenerate(scratch)
            if digest(root, GEN) != digest(scratch, GEN): res["not_identical"] += 1
            shutil.rmtree(scratch)
            if any("<<<<<<<" in (root / g).read_text() for g in GEN): res["marker_left"] += 1
            git(root, "add", "-A"); git(root, "commit", "-qm", "merge", check=False)
        else:
            git(root, "merge", "--abort", check=False)
        git(root, "checkout", "-q", "-f", "main"); git(root, "branch", "-q", "-D", "A", "B")
    ts = res.pop("times")
    if ts: res["regen_seconds"] = dict(mean=round(sum(ts) / len(ts), 3), max=round(max(ts), 3))
    shutil.rmtree(root)
    return res


if __name__ == "__main__":
    out = {"baseline": run("baseline"), "treatment": run("treatment")}
    print(json.dumps(out, indent=2))
