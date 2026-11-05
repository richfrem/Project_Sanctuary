#!/usr/bin/env python3
"""impact.py - Projected impact and overlap analysis for the Sanctuary flight board (MISSION-031).

Purpose: let a mission declare what it will touch (`projected_impact` in its manifest) so the flight board can lock exactly that, refuse
    overlapping missions, and tell an incoming agent which missions can run in parallel without colliding.
Layer: sanctuary-core / operating-sanctuary-guardian (coordination).
Key Input Dependencies: LEARNING/research/missions/*.json (optional `projected_impact`), the board's active leases. Standard library only.
Manifest shape (all keys optional except paths for a registered mission):
    "projected_impact": {"paths": ["dir/", "dir/file.py", "dir/**/*.md"], "resources": ["mlx-server:8099"], "shared_generated": ["extra/file"]}
Conservatism rule: an ambiguous comparison counts as OVERLAP. False positives are allowed (a mission waits); false negatives (a silent collision) are not.
Shared generated files (index.jsonl, backlog.md, ledgers, ...) are rewritten by every mission: they are never lease conflicts; they are reported as
    merge notes ("regenerate after merging, do not hand-merge").
Key Functions: paths_overlap(), lock_set(), load_manifest_impact(), impact_overlap(), overlap_matrix(), dispatchable(), undeclared_changes(), validate_impact().
"""
import fnmatch
import functools
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SHARED_GENERATED = (
    "LEARNING/research/index.jsonl", "LEARNING/research/backlog.md", ".agent/learning/cognitive_ledger.json",
    ".agent/learning/events.jsonl", "chronicle/ledger.json",
)
GLOB_CHARS = set("*?[")
OPEN_STATUSES = {"PROPOSED", "STAGED"}


def norm(path: str) -> str:
    """Normalise a repo-relative path (no leading ./ or /, no duplicate or trailing slashes)."""
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return "/".join(part for part in p.split("/") if part and part != ".")


@functools.lru_cache(maxsize=8192)
def _parts(path: str) -> tuple:
    n = norm(path)
    return tuple(n.split("/")) if n else ()


def _is_glob(component: str) -> bool:
    return any(c in GLOB_CHARS for c in component)


def _component_may_match(a: str, b: str) -> bool:
    """Can two path components denote the same name? Two wildcards are assumed to match (conservative)."""
    if not _is_glob(a) and not _is_glob(b):
        return a == b
    if _is_glob(a) and not _is_glob(b):
        return fnmatch.fnmatchcase(b, a)
    if _is_glob(b) and not _is_glob(a):
        return fnmatch.fnmatchcase(a, b)
    return True


_LEAF_RE = re.compile(r".\.[A-Za-z0-9]+$")


@functools.lru_cache(maxsize=8192)
def _is_leaf(path: str) -> bool:
    """A path that names a file (a non-leading dot extension, no trailing slash) has no children. Extension-less names stay ambiguous (maybe a directory)."""
    raw = (path or "").strip()
    if raw.endswith("/"):
        return False
    parts = _parts(raw)
    return bool(parts) and bool(_LEAF_RE.search(parts[-1]))


def _overlap(p: Sequence[str], q: Sequence[str], p_leaf: bool, q_leaf: bool) -> bool:  # p, q: tuples of path components
    if not p or not q:
        if not p and not q:
            return True                    # equal paths
        # one path is exhausted: it is a directory prefix of the other (overlap) unless it names a file (a file contains nothing)
        return not (p_leaf if not p else q_leaf)
    if p[0] == "**":
        return _overlap(p[1:], q, p_leaf, q_leaf) or _overlap(p, q[1:], p_leaf, q_leaf)
    if q[0] == "**":
        return _overlap(p, q[1:], p_leaf, q_leaf) or _overlap(p[1:], q, p_leaf, q_leaf)
    return _component_may_match(p[0], q[0]) and _overlap(p[1:], q[1:], p_leaf, q_leaf)


def paths_overlap(a: str, b: str) -> bool:
    """True if two paths, directories or globs may share a file (conservative: ambiguity is overlap, but a file has no children)."""
    pa, pb = _parts(a), _parts(b)
    if not pa or not pb:
        return False  # an empty path denotes nothing
    return _overlap(pa, pb, _is_leaf(a), _is_leaf(b))


_GENERATED = frozenset(norm(x) for x in SHARED_GENERATED)


def is_shared_generated(path: str, extra: Iterable[str] = ()) -> bool:
    n = norm(path)
    return n in _GENERATED or any(n == norm(x) for x in extra)


def lock_set(paths: Iterable[str], extra_generated: Iterable[str] = ()) -> List[str]:
    """Paths that should be leased: the declared paths minus shared generated files (those are merge notes, not locks)."""
    out: List[str] = []
    for p in paths:
        if norm(p) and not is_shared_generated(p, extra_generated) and p.strip() not in out:
            out.append(p.strip())
    return out


def load_manifest_impact(root: Path, mission_id: str) -> Optional[Dict[str, List[str]]]:
    """Normalised projected_impact of a mission manifest, or None if the mission has no declaration."""
    manifest = Path(root) / "LEARNING" / "research" / "missions" / f"{mission_id}.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return normalise_impact(data.get("projected_impact"))


def normalise_impact(raw: Any) -> Optional[Dict[str, List[str]]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("paths"), list):
        return None
    paths = [str(x) for x in raw["paths"] if isinstance(x, str) and norm(x)]
    if not paths:
        return None
    return {"paths": paths,
            "resources": [str(x) for x in raw.get("resources", []) if isinstance(x, str) and x.strip()],
            "shared_generated": [str(x) for x in raw.get("shared_generated", []) if isinstance(x, str) and norm(x)]}


def _prepare(imp: Dict[str, List[str]]):
    """Precompute what the pairwise comparison needs: lease-able paths as (components, is_leaf) pairs, and the resource set."""
    locks = lock_set(imp["paths"], imp.get("shared_generated", []))
    return [(_parts(p), _is_leaf(p)) for p in locks], set(imp.get("resources", []))


def _prepared_conflict(a, b) -> bool:
    (la, ra), (lb, rb) = a, b
    if ra & rb:
        return True
    return any(_overlap(p, q, pl, ql) for p, pl in la for q, ql in lb)


def impact_overlap(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> Dict[str, List[Any]]:
    """Overlap between two impacts: conflicting path pairs and shared resources (shared generated files are excluded)."""
    la = lock_set(a["paths"], a.get("shared_generated", []))
    lb = lock_set(b["paths"], b.get("shared_generated", []))
    pairs = [[x, y] for x in la for y in lb if paths_overlap(x, y)]
    resources = sorted(set(a.get("resources", [])) & set(b.get("resources", [])))
    return {"paths": pairs, "resources": resources}


def conflicts(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> bool:
    return _prepared_conflict(_prepare(a), _prepare(b))


def overlap_matrix(impacts: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Adjacency list: mission id -> ids of missions whose impact conflicts with it."""
    ids = sorted(impacts)
    prepared = {i: _prepare(impacts[i]) for i in ids}
    adj: Dict[str, List[str]] = {i: [] for i in ids}
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            if _prepared_conflict(prepared[a], prepared[b]):
                adj[a].append(b)
                adj[b].append(a)
    return adj


def lease_impact(lease: Dict[str, Any]) -> Dict[str, List[str]]:
    return {"paths": list(lease.get("locked_targets", [])), "resources": list(lease.get("exclusive_resources", []))}


def dispatchable(candidates: List[Dict[str, Any]], active_leases: List[Dict[str, Any]], count: int = 3) -> Dict[str, Any]:
    """Greedy by priority: up to `count` registered missions that are free of active leases and mutually disjoint.

    candidates: [{"id", "priority", "impact"|None}]. Returns {"selected": [ids], "blocked": [{id, by}], "unregistered": [ids]}.
    """
    selected: List[Dict[str, Any]] = []
    blocked: List[Dict[str, str]] = []
    unregistered: List[str] = []
    leases = [(l.get("mission_id"), _prepare({"paths": list(l.get("locked_targets", [])), "resources": list(l.get("exclusive_resources", []))}))
              for l in active_leases]
    for c in sorted(candidates, key=lambda c: -float(c.get("priority") or 0)):
        if not c.get("impact"):
            unregistered.append(c["id"])
            continue
        mine = _prepare(c["impact"])
        by = next((mid for mid, prep in leases if _prepared_conflict(mine, prep)), None)
        by = by or next((s["id"] for s in selected if _prepared_conflict(mine, s["prep"])), None)
        if by:
            blocked.append({"id": c["id"], "by": by})
        elif len(selected) < count:
            selected.append({**c, "prep": mine})
    return {"selected": [s["id"] for s in selected], "blocked": blocked, "unregistered": unregistered}


def undeclared_changes(declared: Dict[str, List[str]], changed_files: Iterable[str]) -> List[str]:
    """Files a worktree changed that fall outside the declared impact (shared generated files are allowed)."""
    extra = declared.get("shared_generated", [])
    out = []
    for f in changed_files:
        if is_shared_generated(f, extra):
            continue
        if not any(paths_overlap(f, d) for d in declared["paths"]):
            out.append(f)
    return sorted(set(out))


def validate_impact(raw: Any) -> List[str]:
    """Schema errors for a manifest's projected_impact (empty list if valid or absent)."""
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return ["projected_impact must be an object"]
    errors = []
    if not isinstance(raw.get("paths"), list) or not raw["paths"] or not all(isinstance(x, str) and norm(x) for x in raw["paths"]):
        errors.append("projected_impact.paths must be a non-empty list of repo-relative path strings")
    for key in ("resources", "shared_generated"):
        if key in raw and not (isinstance(raw[key], list) and all(isinstance(x, str) for x in raw[key])):
            errors.append(f"projected_impact.{key} must be a list of strings")
    for p in raw.get("paths", []) if isinstance(raw.get("paths"), list) else []:
        if isinstance(p, str) and (p.startswith("/") or ".." in norm(p).split("/")):
            errors.append(f"projected_impact path must be repo-relative and stay inside the repository: {p}")
    return errors
