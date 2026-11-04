#!/usr/bin/env python3
"""impact_eval.py - MISSION-031 evaluation of projected impact, overlap detection and the flight board.

Purpose: measure (E1) silent conflicts and false negatives/positives over random mission pairs with the OLD checkout code (loaded from git) as baseline and
    the new code, against brute-force ground truth; (E2) real git merges for pairs predicted disjoint, plus `verify` on deliberately undeclared edits;
    (E3) latency on the real registry and at 200 missions / 20 leases; (E4) the live `suggest` result and the parallelism the registry allows.
Key Input Dependencies: git; scripts/manage_coordination.py and impact.py (new code); the pre-change manage_coordination.py via `git show <rev>:<path>`.
Ground truth is computed by expanding every declared path over a synthetic file tree with an independent glob-to-regex translation, NOT with impact.py.
Key Functions: build_universe(), expand(), random_impact(), run_e1(), run_e2(), run_e3(), run_e4(), main().
Usage: python3 impact_eval.py --baseline-rev HEAD --out LEARNING/research/experiments/mission-031/results.json [--pairs 500] [--git-pairs 60]
"""
import argparse
import importlib.util
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
REPO = HERE.parents[5]
sys.path.insert(0, str(SCRIPTS))
import impact  # noqa: E402
import manage_coordination as mc  # noqa: E402

AREAS = [f"plugins/p{i}/{d}" for i in range(6) for d in ("scripts", "docs", "tests")] + [f"area{i}" for i in range(4)]


def build_universe() -> List[str]:
    """Synthetic repository file tree (no shared-generated files)."""
    files = []
    for a in AREAS:
        for j in range(5):
            files += [f"{a}/f{j}.py", f"{a}/f{j}.md"]
        files += [f"{a}/sub/g0.py", f"{a}/sub/g1.md"]
    files += ["README.md", "Makefile", "docs/top.md", "docs/guide/a.md"]
    return sorted(set(files))


def _glob_regex(pattern: str):
    """Independent glob translation: ** = any depth, * = within a component, ? = one character."""
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
            if pattern[i:i + 1] == "/":
                out += "(?:/)?"
                i += 1
        elif c == "*":
            out += "[^/]*"
            i += 1
        elif c == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(c)
            i += 1
    return re.compile(out)


def expand(path: str, universe: List[str]) -> Set[str]:
    """Files covered by a declared path: a file, everything under a directory, or what a glob matches (or lies under a matched directory)."""
    p = path.strip().strip("/")
    if not any(ch in p for ch in "*?["):
        return {f for f in universe if f == p or f.startswith(p + "/")}
    rx = _glob_regex(p)
    out = set()
    for f in universe:
        parts = f.split("/")
        if any(rx.fullmatch("/".join(parts[:k])) for k in range(1, len(parts) + 1)):
            out.add(f)
    return out


def random_impact(rng: random.Random) -> List[str]:
    kinds = []
    for _ in range(rng.randint(1, 3)):
        a = rng.choice(AREAS)
        kinds.append(rng.choice([
            f"{a}/f{rng.randint(0, 4)}.{rng.choice(['py', 'md'])}", f"{a}/", f"{a}/sub/", f"{a}/*.md", f"{a}/**/*.py",
            f"plugins/p{rng.randint(0, 5)}/**", f"plugins/*/{rng.choice(['docs', 'tests', 'scripts'])}/", f"**/*.{rng.choice(['md', 'py'])}",
            "plugins/", "docs/guide/", "README.md", f"area{rng.randint(0, 3)}/**/g?.md"]))
    return kinds


def load_old_module(rev: str):
    """The pre-change manage_coordination.py exactly as committed at `rev` (baseline)."""
    rel = "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py"
    src = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "old_manage_coordination.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("old_mc", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_manifest(root: Path, mid: str, paths: List[str]) -> None:
    d = root / "LEARNING" / "research" / "missions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{mid}.json").write_text(json.dumps({"id": mid, "title": mid, "status": "PROPOSED", "priority_score": 1.0,
                                               "projected_impact": {"paths": paths, "resources": []}}))


def _reset_board(root: Path) -> None:
    shutil.rmtree(root / ".agent", ignore_errors=True)


def run_e1(pairs: int, baseline_rev: Optional[str], seed: int = 31) -> Dict:
    """Random mission pairs; `baseline_rev=None` skips the old-code baseline (used by the quick test)."""
    rng = random.Random(seed)
    universe = build_universe()
    old = load_old_module(baseline_rev) if baseline_rev else None
    root = Path(tempfile.mkdtemp())
    stats = {"pairs": pairs, "overlapping_truth": 0, "new_false_negatives": 0, "new_false_positives": 0, "new_true_positives": 0,
             "old_silent_conflicts": 0, "manual_targets_needed": 0, "examples_fp": []}
    for n in range(pairs):
        a, b = random_impact(rng), random_impact(rng)
        truth = bool(set().union(*[expand(p, universe) for p in a]) & set().union(*[expand(p, universe) for p in b]))
        stats["overlapping_truth"] += truth
        _write_manifest(root, "M-A", a)
        _write_manifest(root, "M-B", b)
        # new code: both checked out WITHOUT --targets; the manifests supply the locks
        _reset_board(root)
        first = mc.checkout(root, "M-A", "agent-a")
        second = mc.checkout(root, "M-B", "agent-b")
        stats["manual_targets_needed"] += first["status"] != "success"
        blocked = second["status"] == "conflict"
        if truth and not blocked:
            stats["new_false_negatives"] += 1
        elif truth and blocked:
            stats["new_true_positives"] += 1
        elif blocked and not truth:
            stats["new_false_positives"] += 1
            if len(stats["examples_fp"]) < 5:
                stats["examples_fp"].append({"a": a, "b": b})
        # old code: the same flow (no targets on the command line) takes no locks at all
        if old is not None:
            _reset_board(root)
            old.checkout(root, "M-A", "agent-a")
            old_second = old.checkout(root, "M-B", "agent-b")
            if truth and old_second["status"] != "conflict":
                stats["old_silent_conflicts"] += 1
    shutil.rmtree(root, ignore_errors=True)
    nontruth = pairs - stats["overlapping_truth"]
    stats["new_false_positive_rate"] = round(stats["new_false_positives"] / max(1, nontruth), 3)
    stats["old_silent_conflict_rate"] = round(stats["old_silent_conflicts"] / max(1, stats["overlapping_truth"]), 3)
    return stats


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.email=e@e", "-c", "user.name=e", *args], cwd=cwd, capture_output=True, text=True)


def run_e2(n_pairs: int, seed: int = 32) -> Dict:
    rng = random.Random(seed)
    universe = build_universe()
    res = {"pairs": n_pairs, "predicted_disjoint": 0, "predicted_disjoint_real_conflicts": 0, "predicted_overlap": 0, "predicted_overlap_real_conflicts": 0,
           "real_conflicts_total": 0, "real_conflicts_not_predicted": 0, "undeclared_planted": 0, "undeclared_flagged": 0, "declared_edits_unflagged": 0, "generated_file_conflicts": 0}
    for n in range(n_pairs):
        a, b = random_impact(rng), random_impact(rng)
        ea = sorted(set().union(*[expand(p, universe) for p in a]))
        eb = sorted(set().union(*[expand(p, universe) for p in b]))
        if not ea or not eb:
            res["pairs"] -= 1
            continue
        d = Path(tempfile.mkdtemp())
        try:
            _git(d, "init", "-q", "-b", "main")
            for f in universe:
                (d / f).parent.mkdir(parents=True, exist_ok=True)
                (d / f).write_text("\n".join(f"line {i}" for i in range(6)) + "\n")
            (d / "LEARNING/research").mkdir(parents=True, exist_ok=True)
            (d / "LEARNING/research/index.jsonl").write_text("base\n")
            _git(d, "add", "."), _git(d, "commit", "-q", "-m", "base")
            for name, edits, paths in (("A", ea, a), ("B", eb, b)):
                _git(d, "checkout", "-q", "-b", f"feat-{name}", "main")
                _write_manifest(d, f"M-{name}", paths)
                for f in rng.sample(edits, min(len(edits), 4)):
                    (d / f).write_text((d / f).read_text().replace("line 2", f"line 2 edited by {name}"))
                (d / "LEARNING/research/index.jsonl").write_text(f"regenerated by {name}\n")   # every mission regenerates this file
                _git(d, "add", "."), _git(d, "commit", "-q", "-m", name)
                v = mc.verify_impact(d, f"M-{name}", worktree=".", base="main")
                res["declared_edits_unflagged"] += (not v["ok"])
                if name == "A" and n % 3 == 0:                                   # plant an undeclared edit on a third of branches
                    outside = [f for f in universe if f not in edits and not any(impact.paths_overlap(f, p) for p in paths)]
                    if outside:
                        res["undeclared_planted"] += 1
                        target = d / rng.choice(outside)
                        target.write_text(target.read_text() + "sneaky\n")
                        res["undeclared_flagged"] += (not mc.verify_impact(d, f"M-{name}", worktree=".", base="main")["ok"])
                        _git(d, "checkout", "--", ".")
            predicted_overlap = impact.conflicts(impact.normalise_impact({"paths": a}), impact.normalise_impact({"paths": b}))
            _git(d, "checkout", "-q", "main")
            _git(d, "merge", "-q", "--no-edit", "feat-A")
            merged = _git(d, "merge", "-q", "--no-edit", "feat-B")
            conflict_files = _git(d, "diff", "--name-only", "--diff-filter=U").stdout.split() if merged.returncode else []
            real_conflict_in_declared = [f for f in conflict_files if f != "LEARNING/research/index.jsonl"]
            res["generated_file_conflicts"] += ("LEARNING/research/index.jsonl" in conflict_files)
            if predicted_overlap:
                res["predicted_overlap"] += 1
                res["predicted_overlap_real_conflicts"] += bool(real_conflict_in_declared)
            else:
                res["predicted_disjoint"] += 1
                res["predicted_disjoint_real_conflicts"] += bool(real_conflict_in_declared)
            res["real_conflicts_total"] += bool(real_conflict_in_declared)
            res["real_conflicts_not_predicted"] += bool(real_conflict_in_declared) and not predicted_overlap
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return res


def _p95(xs: List[float]) -> float:
    return round(sorted(xs)[max(0, int(len(xs) * 0.95) - 1)], 3)


def run_e3(root: Path) -> Dict:
    cands = mc._open_candidates(root)
    impacts = {c["id"]: c["impact"] for c in cands if c["impact"]}
    leases = [{"mission_id": f"L-{i}", "locked_targets": [f"area{i % 4}/"], "exclusive_resources": []} for i in range(5)]
    reg = []
    for _ in range(100):
        t0 = time.perf_counter()
        impact.overlap_matrix(impacts)
        impact.dispatchable(cands, leases, 3)
        reg.append((time.perf_counter() - t0) * 1000)
    rng = random.Random(3)
    pool = [f"{a}/" for a in AREAS] + [f"area{i}/**/*.md" for i in range(8)]
    big = {f"M-{i}": {"paths": rng.sample(pool, 3), "resources": []} for i in range(200)}
    held = [{"mission_id": f"L-{i}", "locked_targets": rng.sample(pool, 3), "exclusive_resources": []} for i in range(20)]
    per = []
    for imp in list(big.values())[:100]:
        t0 = time.perf_counter()
        any(impact.conflicts(imp, impact.lease_impact(l)) for l in held)
        per.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    impact.overlap_matrix(big)
    matrix_ms = (time.perf_counter() - t0) * 1000
    return {"registry_missions": len(impacts), "registry_matrix_plus_suggest_ms_p95": _p95(reg), "registry_ms_max": round(max(reg), 3),
            "synthetic_200_missions_20_leases_per_mission_check_ms_p95": _p95(per), "synthetic_200_full_matrix_ms": round(matrix_ms, 1)}


def run_e4(root: Path) -> Dict:
    s = mc.suggest(root, 3)
    m = mc.build_matrix(root)
    cands = {c["id"]: c["impact"] for c in mc._open_candidates(root) if c["impact"]}
    mutually_disjoint = all(not impact.conflicts(cands[x], cands[y]) for i, x in enumerate(s["selected"]) for y in s["selected"][i + 1:])
    biggest = impact.dispatchable([{"id": k, "priority": 1, "impact": v} for k, v in cands.items()], [], 99)["selected"]
    return {"suggested": s["selected"], "mutually_disjoint": mutually_disjoint, "registered": len(m["registered"]), "unregistered": len(m["unregistered"]),
            "max_parallel_set_size_greedy": len(biggest)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-rev", default="HEAD")
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--git-pairs", type=int, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {"e1_random_pairs": run_e1(a.pairs, a.baseline_rev), "e2_real_git": run_e2(a.git_pairs), "e3_latency": run_e3(REPO), "e4_live_registry": run_e4(REPO)}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
