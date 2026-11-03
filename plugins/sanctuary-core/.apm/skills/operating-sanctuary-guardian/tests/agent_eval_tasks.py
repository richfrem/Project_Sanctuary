"""Held-out task generator and graders for the MISSION-022 local agent loop evaluation.

Ground truth is COMPUTED from repository state at run time (never hand-written), and this
module lives under a path the agent sandbox denies, so the agent cannot read the answers.

Task tiers (10 each):
  scalar  - read a named mission file, report one field        (1 tool call)
  locate  - find the mission by a unique keyword, report a field (search then read)
  dirset  - list directory entries matching a condition          (exact set equality)
"""
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set

IGNORED = {".git", "__pycache__", ".DS_Store"}
MISSIONS = "LEARNING/research/missions"
MAX_MISSION = 21          # missions >21 are being edited by this experiment; excluded
SCALAR_FIELDS = ["status", "priority_score", "execution_surface"]
STOP = set("the and for with via vs into from that this offline local sovereign".split())


def _missions(repo: Path) -> Dict[str, dict]:
    out = {}
    for f in sorted((repo / MISSIONS).glob("MISSION-*.json")):
        n = int(f.stem.split("-")[1])
        if n <= MAX_MISSION:
            out[f.stem] = json.loads(f.read_text())
    return out


def _entries(d: Path) -> List[Path]:
    return [e for e in sorted(d.iterdir()) if e.name not in IGNORED]


def build_tasks(repo: Path, seed: int = 1337) -> List[dict]:
    rng = random.Random(seed)
    repo = Path(repo)
    ms = _missions(repo)
    ids = sorted(ms)
    tasks: List[dict] = []

    # --- scalar -----------------------------------------------------------------
    for i, mid in enumerate(rng.sample(ids, 10)):
        field = SCALAR_FIELDS[i % len(SCALAR_FIELDS)]
        tasks.append({"id": f"S{i+1}", "tier": "scalar", "type": "scalar", "field": field,
                      "prompt": f"Read {MISSIONS}/{mid}.json and report the value of its \"{field}\" field. Answer with only the value.",
                      "expected": str(ms[mid][field])})

    # --- locate (search for a unique keyword, report the mission id) (keyword unique across the full text of all mission files) --
    texts = {m: (repo / MISSIONS / f"{m}.json").read_text().lower() for m in ids}
    cands = []
    for m in ids:
        for w in set(re.findall(r"[a-z]{7,}", ms[m]["title"].lower())):
            if w not in STOP and sum(w in t for t in texts.values()) == 1:
                cands.append((m, w))
    rng.shuffle(cands)
    seen, picked = set(), []
    for m, w in cands:
        if m not in seen:
            seen.add(m)
            picked.append((m, w))
        if len(picked) == 10:
            break
    for i, (mid, kw) in enumerate(picked):
        tasks.append({"id": f"L{i+1}", "tier": "locate", "type": "scalar", "field": "id",
                      "prompt": f"One mission file under {MISSIONS}/ has a title containing the word \"{kw}\". "
                                f"Find it and report the mission id (the value of its \"id\" field, like MISSION-007). Answer with only the id.",
                      "expected": mid})

    # --- directory sets ---------------------------------------------------------------
    plans = []
    for rel in ["context", ".agent", "chronicle", "LEARNING", "protocols",
                "plugins/sanctuary-core/.apm/skills",
                "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references",
                "plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts",
                "plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts"]:
        d = repo / rel
        if not d.is_dir():
            continue
        es = _entries(d)
        if any(e.name.startswith(".") for e in es):
            continue  # machine-local hidden entries make expected sets environment-dependent
        dirs = [e.name for e in es if e.is_dir()]
        for ext in sorted({e.suffix for e in es if e.is_file() and e.suffix}):
            names = [e.name for e in es if e.is_file() and e.suffix == ext]
            if 2 <= len(names) <= 10:
                plans.append((f"List the names of all files with the \"{ext}\" extension directly inside {rel}/ (not in subdirectories).", names))
        if 2 <= len(dirs) <= 10:
            plans.append((f"List the names of all subdirectories directly inside {rel}/.", dirs))
    top = _entries(repo)
    for letter in sorted({e.name[0] for e in top if e.is_dir() and not e.name.startswith(".")}):
        names = [e.name for e in top if e.is_dir() and e.name[0] == letter]
        if len(names) >= 2:
            plans.append((f"List the names of all top-level directories of the repository whose names start with the exact character \"{letter}\" (case-sensitive).", names))
    rng.shuffle(plans)
    for i, (prompt, names) in enumerate(plans[:10]):
        tasks.append({"id": f"D{i+1}", "tier": "dirset", "type": "set",
                      "prompt": prompt + " Answer with a comma-separated list of names only.",
                      "expected": sorted(names)})
    return tasks


# --- graders ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[`'\"*]", "", s).strip().lower().rstrip(".")


def _vocab(repo: Path, field: str) -> Set[str]:
    return {str(m[field]).lower() for m in _missions(repo).values() if field in m}


def grade(task: dict, final: str, repo: Path) -> dict:
    if final is None:
        return {"correct": False, "reason": "no final answer"}
    if task["type"] == "set":
        items = [re.sub(r"[`'\"*]", "", x).strip().strip("-•").strip() for x in re.split(r"[,\n;]", final)]
        got = {Path(x.rstrip("/")).name for x in items if x}
        exp = set(task["expected"])
        tp = len(got & exp)
        return {"correct": got == exp, "precision": tp / len(got) if got else 0.0,
                "recall": tp / len(exp) if exp else 1.0, "reason": "" if got == exp else f"got={sorted(got)}"}
    ans, exp = _norm(final), _norm(task["expected"])
    if task["field"] == "id":
        ok = exp in ans and len(set(re.findall(r"mission-\d+", ans))) == 1
    else:
        words = re.findall(r"[\w.\-]+", ans)
        try:
            exp_f = float(exp)
            nums = [float(w) for w in words if re.fullmatch(r"\d+(\.\d+)?", w)]
            ok = nums == [exp_f] or (len(nums) >= 1 and exp_f in nums and len(set(nums)) == 1)
        except ValueError:
            others = {v for v in _vocab(repo, task["field"]) if v != exp}
            ok = exp in words and not any(o in words for o in others)
    return {"correct": ok, "reason": "" if ok else f"expected={exp!r} got={ans[:80]!r}"}
