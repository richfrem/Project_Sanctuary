#!/usr/bin/env python3
"""manage_backlog.py - Agent-Native Cognitive Backlog & Investment Tracker.
Purpose: Sovereign, agent-first backlog and falsifiable mission tracker for Project Sanctuary. Provides Tier 1 machine indices, append-only inbox capture, structured mission manifests, cheapest-viable model routing (MISSION-016), and deterministic compilation to human Markdown views.
Layer: sanctuary-core / managing-sovereign-memory (Layer 2 Epistemic Backlog & Mission Governance).
Key Input Dependencies: LEARNING/missions/*.json (mission manifests), LEARNING/research/inbox.jsonl (appended by capture), .agent/learning/events.jsonl (optional event log).
Key Outputs: LEARNING/research/index.jsonl (Tier 1 fast index), LEARNING/backlog.md (human view).
Usage:
    python3 manage_backlog.py list [--status STATUS] [--format text|json|jsonl]
    python3 manage_backlog.py route <TASK_DESCRIPTION> [--type TYPE] [--failures N] [--format json|text]
    python3 manage_backlog.py get <MISSION-ID>
    python3 manage_backlog.py capture --idea "..." [--author AGENT] [--tags ...]
    python3 manage_backlog.py related (--id <MISSION-ID> | --text "draft idea" [--tags ...]) [--top N] [--format text|json]
    python3 manage_backlog.py compile
    python3 manage_backlog.py validate
    python3 manage_backlog.py record-result --id <MISSION-ID> --verdict CORROBORATED|FALSIFIED|ABANDONED --evidence "..." [--citation "..."] [--lessons "..."]

Function Index:
    get_project_root, get_research_dir, calculate_priority_score,
    recommend_model_routing, is_free_local_task, run_route,
    BacklogManager (load_missions, get_mission, save_mission, capture_inbox,
    record_result, compile, validate, complete_model_routing), main
"""

import re
import sys
import json
import uuid
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "LEARNING").exists():
            return parent
    return Path.cwd()


def get_research_dir(root: Optional[Path] = None) -> Path:
    base = root or get_project_root()
    return base / "LEARNING" / "research"


def validate_projected_impact(raw: Any) -> List[str]:
    """Schema errors for an optional `projected_impact` declaration (MISSION-031); empty if absent or valid.

    Shape: {"paths": [repo-relative files, directories or globs], "resources": [ids], "shared_generated": [paths]}. A malformed declaration
    would silently under-lock a mission on the flight board, so it is rejected here.
    """
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return ["projected_impact must be an object"]
    errors: List[str] = []
    paths = raw.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p.strip() for p in paths):
        errors.append("projected_impact.paths must be a non-empty list of path strings")
    else:
        for p in paths:
            if p.startswith("/") or ".." in p.replace("\\", "/").split("/"):
                errors.append(f"projected_impact path must be repo-relative and inside the repository: {p}")
    for key in ("resources", "shared_generated"):
        if key in raw and not (isinstance(raw[key], list) and all(isinstance(x, str) for x in raw[key])):
            errors.append(f"projected_impact.{key} must be a list of strings")
    return errors


_STOPWORDS = frozenset(
    "the and for that with from this are was were has have had not but can will would should could into onto over "
    "than then them they their its via per any all each both more most less such only also when where which what "
    "while within without across between about after before during under using used use based new existing mission "
    "missions agent agents".split()
)


def _terms(text: str) -> List[str]:
    """Lowercase word tokens (len >= 3), stopwords removed, plural 's' stripped."""
    out: List[str] = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) < 3 or w in _STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _mission_text(m: Dict[str, Any]) -> str:
    return " ".join([m.get("title", ""), m.get("hypothesis", ""), " ".join(m.get("tags", []))])


def related_missions(missions: List[Dict[str, Any]], target_id: Optional[str] = None,
                     text: Optional[str] = None, tags: Optional[List[str]] = None,
                     top: int = 10) -> List[Dict[str, Any]]:
    """Rank other missions by lexical relatedness to a mission (target_id) or a draft idea (text).

    Score = TF-IDF cosine of title+hypothesis+tags, plus 0.3 x tag Jaccard. Lexical only: it suggests
    candidates for a human/agent to judge; it does not prove a relationship. Missions scoring 0 are omitted.
    Each result carries link state: linked (mutual), one-way, or unlinked.
    """
    import math
    from collections import Counter

    target: Optional[Dict[str, Any]] = None
    if target_id is not None:
        target = next((m for m in missions if m.get("id") == target_id), None)
        if target is None:
            raise KeyError(target_id)
        qtext, qtags = _mission_text(target), set(target.get("tags", []))
    else:
        qtext, qtags = text or "", set(tags or [])

    docs = {m["id"]: Counter(_terms(_mission_text(m))) for m in missions}
    df = Counter(t for c in docs.values() for t in c)
    n = len(missions) + 1
    idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}

    def vec(c: "Counter[str]") -> Dict[str, float]:
        return {t: (1 + math.log(f)) * idf.get(t, math.log(n + 1) + 1.0) for t, f in c.items()}

    def cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        dot = sum(w * b[t] for t, w in a.items() if t in b)
        na = math.sqrt(sum(w * w for w in a.values()))
        nb = math.sqrt(sum(w * w for w in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    qvec = vec(Counter(_terms(qtext)))
    results: List[Dict[str, Any]] = []
    for m in missions:
        if m.get("id") == target_id:
            continue
        mvec = vec(docs[m["id"]])
        mtags = set(m.get("tags", []))
        jac = len(qtags & mtags) / len(qtags | mtags) if (qtags | mtags) else 0.0
        score = cos(qvec, mvec) + 0.3 * jac
        if score <= 0:
            continue
        shared = sorted((t for t in qvec if t in mvec), key=lambda t: -(qvec[t] * mvec[t]))[:5]
        fwd = target is not None and m["id"] in (target.get("related_missions") or {})
        back = target is not None and target_id in (m.get("related_missions") or {})
        link = "linked" if fwd and back else "one-way" if fwd or back else "unlinked"
        results.append({"id": m["id"], "status": m.get("status", ""), "score": round(score, 3),
                        "shared_terms": shared, "link": link, "title": m.get("title", "")})
    results.sort(key=lambda r: (-r["score"], r["id"]))
    return results[:top]


def hypothesis_brief(hypothesis: str, limit: int = 220) -> str:
    """First sentence of a hypothesis, capped at ``limit`` chars, so agents can triage from the index alone."""
    text = " ".join((hypothesis or "").split())
    if not text:
        return ""
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "\u2026"


def build_index_item(rank: int, m: Dict[str, Any]) -> Dict[str, Any]:
    """Single source of truth for one Tier 1 index line (used by `compile` and `list --format jsonl`)."""
    mr = m.get("model_routing", {})
    return {
        "rank": rank,
        "id": m["id"],
        "title": m["title"],
        "status": m.get("status"),
        "priority_score": m.get("priority_score"),
        "execution_surface": m.get("execution_surface", "local"),
        "cheapest_viable_tier": mr.get("cheapest_viable_tier"),
        "recommended_model": mr.get("recommended_model"),
        "verdict": m.get("verdict"),
        "hypothesis_brief": hypothesis_brief(m.get("hypothesis", "")),
        "tags": m.get("tags", []),
        "updated_at": m.get("updated_at"),
        "path": f"LEARNING/missions/{m['id']}.json",
    }


def calculate_priority_score(leverage: int, depth: int, cost: float, friction: int) -> float:
    """Calculates priority score: (leverage * depth) / (cost * friction)."""
    if cost <= 0:
        cost = 0.05
    if friction <= 0:
        friction = 1
    return round((leverage * depth) / (cost * friction), 2)


# Cost weights mirror MODEL_COST_TIERS in evaluate_dispatch.py (kept in sync by tests).
ROUTING_WEIGHTS: Dict[str, float] = {
    "qwen2-7b-mlx-4bit": 0.05,
    "gemini-3.8-flash": 0.1,
    "claude-haiku-5.5": 0.2,
    "claude-opus": 4.0,
    "gpt-6-astra": 4.0,
}
FREE_LOCAL_KEYWORDS = {
    "inspection", "inspect", "test", "tests", "testing", "verify", "verification",
    "validate", "lint", "local", "offline", "mlx",
}
ESCALATION_FAILURE_THRESHOLD = 2


def is_free_local_task(task_type: str, description: str = "") -> bool:
    """Returns True when the task type or description marks it as local inspection/test/verification."""
    tokens = set(re.findall(r"[a-z0-9]+", f"{task_type} {description}".lower()))
    return bool(tokens & FREE_LOCAL_KEYWORDS)


def recommend_model_routing(task_type: str, description: str = "", failures: int = 0) -> Dict[str, Any]:
    """Recommends the cheapest viable model; permits frontier escalation only after repeated failures.

    Args:
        task_type: Task category (e.g. "inspection", "test", "exploration", "formatting").
        description: Free-text task description used to detect local-only work.
        failures: Count of failed attempts so far. Frontier is allowed when >= 2.

    Returns:
        Dict with cheapest_viable_tier, recommended_model, cost_weight, cli_target,
        fallback_models, and frontier_escalation_condition.
    """
    failures = max(0, int(failures))
    if failures >= ESCALATION_FAILURE_THRESHOLD:
        recommended, fallbacks, tier, cli = "claude-opus", ["gpt-6-astra"], "frontier", "cli-agent"
    elif is_free_local_task(task_type, description):
        recommended, fallbacks, tier, cli = "qwen2-7b-mlx-4bit", ["gemini-3.8-flash", "claude-haiku-5.5"], "free-local", "mlx-lm"
    else:
        recommended, fallbacks, tier, cli = "gemini-3.8-flash", ["claude-haiku-5.5", "qwen2-7b-mlx-4bit"], "cheap", "cli-agent"
    return {
        "cheapest_viable_tier": tier,
        "recommended_model": recommended,
        "cost_weight": ROUTING_WEIGHTS[recommended],
        "cli_target": cli,
        "fallback_models": fallbacks,
        "frontier_escalation_condition": (
            f"Escalate to frontier only after {ESCALATION_FAILURE_THRESHOLD} failed cheap-tier attempts"
        ),
    }


def run_route(args: argparse.Namespace) -> None:
    """Prints the routing recommendation for the `route` subcommand as JSON or text."""
    result = recommend_model_routing(args.type, args.description, args.failures)
    if args.format == "json":
        print(json.dumps(result, indent=2))
        return
    print(f"Tier:               {result['cheapest_viable_tier']}")
    print(f"Recommended model:  {result['recommended_model']} (cost weight {result['cost_weight']}x)")
    print(f"Fallback models:    {', '.join(result['fallback_models'])}")
    print(f"CLI target:         {result['cli_target']}")
    print(f"Escalation:         {result['frontier_escalation_condition']}")


class BacklogManager:
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or get_project_root()
        self.research_dir = self.root_dir / "LEARNING" / "research"
        self.missions_dir = self.root_dir / "LEARNING" / "missions"
        self.inbox_file = self.research_dir / "inbox.jsonl"
        self.index_file = self.research_dir / "index.jsonl"
        self.markdown_file = self.root_dir / "LEARNING" / "backlog.md"
        self.events_file = self.root_dir / ".agent" / "learning" / "events.jsonl"

        self.missions_dir.mkdir(parents=True, exist_ok=True)
        self.research_dir.mkdir(parents=True, exist_ok=True)

    def load_missions(self) -> List[Dict[str, Any]]:
        """Loads all mission JSON manifests from missions_dir."""
        missions: List[Dict[str, Any]] = []
        if not self.missions_dir.exists():
            return missions

        for f in sorted(self.missions_dir.glob("MISSION-*.json")):  # not mission.schema.json, which lives beside the manifests
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    missions.append(data)
            except Exception as e:
                print(f"[WARN] Failed to parse {f.name}: {e}", file=sys.stderr)
        return missions

    def get_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        target = self.missions_dir / f"{mission_id}.json"
        if target.exists():
            with open(target, "r", encoding="utf-8") as fp:
                return json.load(fp)
        return None

    def save_mission(self, mission: Dict[str, Any]) -> Path:
        mission_id = mission["id"]
        target = self.missions_dir / f"{mission_id}.json"
        with open(target, "w", encoding="utf-8") as fp:
            json.dump(mission, fp, indent=2)
        return target

    def capture_inbox(self, idea: str, author: str = "agent", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Appends a raw unvetted idea to inbox.jsonl."""
        now = datetime.now(timezone.utc)
        seed_id = f"SEED-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        entry = {
            "id": seed_id,
            "created_at": now.isoformat(),
            "author": author,
            "idea": idea.strip(),
            "tags": tags or [],
            "status": "CAPTURED",
            "model_routing": recommend_model_routing((tags or ["general"])[0], idea)
        }
        with open(self.inbox_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")
        return entry

    def record_result(self, mission_id: str, verdict: str, evidence: str,
                      citation: str = "", lessons: Optional[List[str]] = None,
                      agent: str = "agent") -> Dict[str, Any]:
        """Records an empirical trial verdict, evidence, and lessons learned."""
        mission = self.get_mission(mission_id)
        if not mission:
            raise ValueError(f"Mission {mission_id} not found.")

        valid_verdicts = {"CORROBORATED", "FALSIFIED", "ABANDONED"}
        if verdict.upper() not in valid_verdicts:
            raise ValueError(f"Invalid verdict '{verdict}'. Must be one of {valid_verdicts}")

        now = datetime.now(timezone.utc)
        mission["verdict"] = verdict.upper()
        mission["status"] = "RESOLVED"
        mission["updated_at"] = now.isoformat()

        if "evidence" not in mission:
            mission["evidence"] = []

        evidence_entry = {
            "timestamp": now.isoformat(),
            "agent": agent,
            "citation": citation,
            "details": evidence
        }
        mission["evidence"].append(evidence_entry)

        if lessons:
            if "lessons_learned" not in mission:
                mission["lessons_learned"] = []
            mission["lessons_learned"].extend(lessons)

        self.save_mission(mission)

        # Log event into .agent/learning/events.jsonl if root has it
        if self.events_file.parent.exists():
            event = {
                "event_type": "MISSION_RESOLVED",
                "mission_id": mission_id,
                "timestamp": now.isoformat(),
                "verdict": verdict.upper(),
                "citation": citation,
                "agent": agent,
                "lessons": lessons or []
            }
            with open(self.events_file, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(event) + "\n")

        # Recompile index and markdown
        self.compile()
        return mission

    def complete_model_routing(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the mission's model_routing, filling any missing or empty keys via recommend_model_routing.

        Existing non-empty values are preserved; only gaps are populated.
        """
        required = ("cheapest_viable_tier", "recommended_model", "cost_weight", "fallback_models", "cli_target")
        existing = mission.get("model_routing") or {}
        present = {k: v for k, v in existing.items() if v not in (None, "", [])}
        if all(k in present for k in required):
            return existing
        generated = recommend_model_routing(
            mission.get("task_type") or "general",
            f"{mission.get('title', '')} {mission.get('hypothesis', '')}",
            failures=mission.get("failures", 0),
        )
        merged = {**generated, **present}
        # Keep the cost weight and fallbacks consistent with the model actually recommended.
        if "cost_weight" not in present and merged["recommended_model"] in ROUTING_WEIGHTS:
            merged["cost_weight"] = ROUTING_WEIGHTS[merged["recommended_model"]]
        if "fallback_models" not in present:
            merged["fallback_models"] = [f for f in generated["fallback_models"] if f != merged["recommended_model"]]
        return merged

    def compile(self) -> None:
        """Validates all missions, writes Tier 1 index.jsonl, and generates backlog.md."""
        missions = self.load_missions()

        # Recalculate priority scores, auto-populate model routing, and validate
        for m in missions:
            m["model_routing"] = self.complete_model_routing(m)
            scores = m.get("scores", {})
            leverage = scores.get("strategic_leverage", m.get("strategic_leverage", 1))
            depth = scores.get("epistemic_depth", m.get("epistemic_depth", 1))
            cost = scores.get("cost_weight", m.get("cost_weight", 1.0))
            friction = scores.get("operational_friction", m.get("operational_friction", 1))

            calc_score = calculate_priority_score(leverage, depth, cost, friction)
            scores["strategic_leverage"] = leverage
            scores["epistemic_depth"] = depth
            scores["cost_weight"] = cost
            scores["operational_friction"] = friction
            scores["priority_score"] = calc_score
            m["scores"] = scores
            m["priority_score"] = calc_score
            self.save_mission(m)

        # Sort by priority score descending
        missions.sort(
            key=lambda x: (
                x.get("priority_score", 0.0),
                x.get("scores", {}).get("strategic_leverage", 0),
                x.get("scores", {}).get("epistemic_depth", 0)
            ),
            reverse=True
        )

        # 1. Write Tier 1 index.jsonl
        with open(self.index_file, "w", encoding="utf-8") as fp:
            for rank, m in enumerate(missions, start=1):
                index_item = build_index_item(rank, m)
                fp.write(json.dumps(index_item) + "\n")

        # 2. Write Human-Readable backlog.md
        md_lines = [
            "<!-- THIS FILE IS AUTOMATICALLY GENERATED BY manage_backlog.py compile. DO NOT EDIT BY HAND -->",
            "# Prioritized Sovereign Research Backlog",
            "",
            "## 1. Executive Summary & Philosophy",
            "",
            "Project Sanctuary operates as a **sovereign engineering habitat, anti-Asch epistemic platform, and autonomous agent substrate**.",
            "Autonomous agents maintain their backlogged inquiries natively using **Tier 1 machine index (`index.jsonl`)** and structured JSON manifests (`missions/*.json`).",
            "This document is a generated human projection representing the current portfolio.",
            "",
            "---",
            "",
            "## 2. Priority & ROI Formula",
            "",
            "$$\\text{Priority Score} = \\frac{\\text{Strategic Leverage (1–5)} \\times \\text{Epistemic Depth (1–5)}}{\\text{Cost Weight (0.05–4.5)} \\times \\text{Operational Friction (1–3)}}$$",
            "",
            "---",
            "",
            "## 3. Prioritized Missions Portfolio",
            "",
            "| Rank | ID | Title | Leverage | Depth | Cost | Friction | Score | Status | Verdict |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for rank, m in enumerate(missions, start=1):
            s = m.get("scores", {})
            verdict_str = f"`{m['verdict']}`" if m.get("verdict") else "-"
            status_str = f"`{m.get('status', 'PROPOSED')}`"
            md_lines.append(
                f"| **#{rank}** | `{m['id']}` | **{m['title']}** | {s.get('strategic_leverage', 1)} | "
                f"{s.get('epistemic_depth', 1)} | {s.get('cost_weight', 1.0)}x | {s.get('operational_friction', 1)} | "
                f"**{m.get('priority_score', 0.0)}** | {status_str} | {verdict_str} |"
            )

        md_lines.extend(["", "---", "", "## 4. Mission Deep-Dives", ""])

        for rank, m in enumerate(missions, start=1):
            md_lines.append(f"### {m['id']}: {m['title']} (Rank #{rank})")
            md_lines.append(f"- **Status**: `{m.get('status', 'PROPOSED')}`")
            if m.get("verdict"):
                md_lines.append(f"- **Empirical Verdict**: `{m['verdict']}`")
            md_lines.append(f"- **Execution Surface**: `{m.get('execution_surface', 'local')}`")
            md_lines.append(f"- **Target Platform**: {m.get('target_platform', 'Sanctuary core')}")
            
            mr = m.get("model_routing")
            if mr:
                rec_m = mr.get("recommended_model", "N/A")
                tier = mr.get("cheapest_viable_tier", "standard")
                cli = mr.get("cli_target", "cli-agent")
                md_lines.append(f"- **Cheapest Viable Model**: `{rec_m}` (Tier: `{tier}`, CLI: `{cli}`)")
                fallbacks = mr.get("fallback_models")
                if fallbacks:
                    md_lines.append(f"- **Fallback Models**: {', '.join(f'`{f}`' for f in fallbacks)}")

            md_lines.append(f"- **Hypothesis (Popperian Prediction)**:\n  > {m.get('hypothesis', 'N/A')}")
            md_lines.append(f"- **Falsification Gate**:\n  > {m.get('falsification_criteria', 'N/A')}")

            evidence = m.get("evidence", [])
            if evidence:
                md_lines.append("- **Empirical Evidence & Trials**:")
                for ev in evidence:
                    if isinstance(ev, dict):
                        cit = f" ({ev.get('citation')})" if ev.get("citation") else ""
                        md_lines.append(f"  * `{ev.get('timestamp', '')}` [{ev.get('agent', 'agent')}]{cit}: {ev.get('details', '')}")
                    else:
                        md_lines.append(f"  * {ev}")

            lessons = m.get("lessons_learned", [])
            if lessons:
                md_lines.append("- **Lessons Learned & Epistemic Compounding**:")
                for les in lessons:
                    md_lines.append(f"  * {les}")

            md_lines.append("")

        with open(self.markdown_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(md_lines) + "\n")

    def validate(self) -> List[str]:
        """Validates all missions for schema correctness and score consistency."""
        errors = []
        missions = self.load_missions()

        for m in missions:
            mid = m.get("id", "UNKNOWN")
            if not m.get("title"):
                errors.append(f"[{mid}] Missing title")
            if not m.get("hypothesis"):
                errors.append(f"[{mid}] Missing hypothesis (Popperian prediction required)")
            if not m.get("falsification_criteria"):
                errors.append(f"[{mid}] Missing falsification criteria")

            errors.extend(f"[{mid}] {e}" for e in validate_projected_impact(m.get("projected_impact")))

            scores = m.get("scores", {})
            leverage = scores.get("strategic_leverage", m.get("strategic_leverage"))
            depth = scores.get("epistemic_depth", m.get("epistemic_depth"))
            cost = scores.get("cost_weight", m.get("cost_weight"))
            friction = scores.get("operational_friction", m.get("operational_friction"))

            if None in (leverage, depth, cost, friction):
                errors.append(f"[{mid}] Missing one or more scoring dimensions")
            else:
                expected = calculate_priority_score(leverage, depth, cost, friction)
                actual = m.get("priority_score")
                if actual is not None and abs(actual - expected) > 0.05:
                    errors.append(f"[{mid}] Stored priority score ({actual}) != calculated ({expected})")

        errors.extend(self._schema_errors(missions))
        errors.extend(self._stale_location_errors())
        return errors

    def _stale_location_errors(self) -> List[str]:
        """The manifests moved from LEARNING/research/missions to LEARNING/missions; a file left (or newly written) at the old path would be silently ignored."""
        old = self.root_dir / "LEARNING" / "research" / "missions"
        if not old.is_dir():
            return []
        stray = sorted(p.name for p in old.iterdir() if not p.name.startswith("."))
        return [f"[location] {len(stray)} file(s) at LEARNING/research/missions/ ({', '.join(stray[:3])}...): mission manifests now live in LEARNING/missions/; move them with git mv"] if stray else []

    def _schema_errors(self, missions: List[Dict[str, Any]]) -> List[str]:
        """Structural errors from LEARNING/missions/mission.schema.json (via validate_missions.py); none if the root has no schema."""
        schema_path = self.missions_dir / "mission.schema.json"
        if not schema_path.exists():
            return []
        import importlib.util
        script = Path(__file__).resolve().parent / "validate_missions.py"
        try:
            spec = importlib.util.spec_from_file_location("validate_missions", str(script))
            vm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vm)
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ImportError, AttributeError) as exc:
            return [f"[schema] mission.schema.json exists but could not be applied: {exc}"]
        out: List[str] = []
        for m in missions:
            out.extend(vm.validate_manifest(m, schema))
        return out


def main():
    parser = argparse.ArgumentParser(description="Agent-Native Cognitive Backlog Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_p = subparsers.add_parser("list", help="List missions")
    list_p.add_argument("--status", help="Filter by status")
    list_p.add_argument("--limit", type=int, default=None, help="Limit number of items displayed")
    list_p.add_argument("--format", choices=["text", "json", "jsonl"], default="text", help="Output format")

    # route
    route_p = subparsers.add_parser("route", help="Recommend cheapest viable model for a task")
    route_p.add_argument("description", help="Task description")
    route_p.add_argument("--type", default="general", help="Task type (inspection, test, exploration, formatting, ...)")
    route_p.add_argument("--failures", type=int, default=0, help="Failed attempts so far (frontier allowed at >= 2)")
    route_p.add_argument("--format", choices=["json", "text"], default="json", help="Output format")

    # get
    get_p = subparsers.add_parser("get", help="Get a single mission manifest")
    get_p.add_argument("id", help="Mission ID (e.g. MISSION-001)")

    # capture
    cap_p = subparsers.add_parser("capture", help="Capture a raw idea to inbox.jsonl")
    cap_p.add_argument("--idea", required=True, help="Idea text")
    cap_p.add_argument("--author", default="agent", help="Author/agent identifier")
    cap_p.add_argument("--tags", nargs="*", help="Tags")

    # related
    rel_p = subparsers.add_parser("related", help="Suggest related missions for a mission or a draft idea (lexical)")
    rel_p.add_argument("--id", help="Existing mission ID to compare against")
    rel_p.add_argument("--text", help="Draft title/hypothesis text for a mission not yet created")
    rel_p.add_argument("--tags", nargs="*", help="Tags of the draft (with --text)")
    rel_p.add_argument("--top", type=int, default=10, help="Number of candidates")
    rel_p.add_argument("--format", choices=["text", "json"], default="text")

    # compile
    subparsers.add_parser("compile", help="Compile manifests into index.jsonl and backlog.md")

    # validate
    subparsers.add_parser("validate", help="Validate all mission manifests")

    # record-result
    res_p = subparsers.add_parser("record-result", help="Record empirical verdict and evidence")
    res_p.add_argument("--id", required=True, help="Mission ID")
    res_p.add_argument("--verdict", required=True, choices=["CORROBORATED", "FALSIFIED", "ABANDONED"])
    res_p.add_argument("--evidence", required=True, help="Evidence details")
    res_p.add_argument("--citation", default="", help="Citation ref, commit SHA, or file link")
    res_p.add_argument("--lessons", nargs="*", help="Key lessons learned")
    res_p.add_argument("--agent", default="agent", help="Executing agent identity")

    args = parser.parse_args()
    if args.command == "route":
        run_route(args)
        return
    mgr = BacklogManager()

    if args.command == "list":
        missions = mgr.load_missions()
        if args.status:
            missions = [m for m in missions if m.get("status", "").upper() == args.status.upper()]

        # Sort by priority score descending
        missions.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)

        if args.limit and args.limit > 0:
            missions = missions[:args.limit]

        if args.format == "json":
            print(json.dumps(missions, indent=2))
        elif args.format == "jsonl":
            for rank, m in enumerate(missions, start=1):
                item = build_index_item(rank, m)
                print(json.dumps(item))
        else:
            print(f"{'Rank':<5} {'ID':<12} {'Status':<10} {'Score':<8} {'Model Rec':<20} {'Title'}")
            print("-" * 80)
            for rank, m in enumerate(missions, start=1):
                rec_m = m.get("model_routing", {}).get("recommended_model") or "-"
                print(f"#{rank:<4} {m['id']:<12} {m.get('status',''):<10} {m.get('priority_score',0.0):<8.1f} {rec_m:<20} {m.get('title','')}")

    elif args.command == "get":
        mission = mgr.get_mission(args.id)
        if not mission:
            print(f"Mission '{args.id}' not found.", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(mission, indent=2))

    elif args.command == "related":
        if bool(args.id) == bool(args.text):
            print("Provide exactly one of --id or --text.", file=sys.stderr)
            sys.exit(2)
        try:
            found = related_missions(mgr.load_missions(), target_id=args.id, text=args.text,
                                     tags=args.tags, top=args.top)
        except KeyError:
            print(f"Mission '{args.id}' not found.", file=sys.stderr)
            sys.exit(1)
        if args.format == "json":
            print(json.dumps(found, indent=2))
        else:
            print(f"{'ID':<12} {'Status':<11} {'Score':<6} {'Link':<9} Shared terms | Title")
            for r in found:
                print(f"{r['id']:<12} {r['status']:<11} {r['score']:<6.2f} {r['link']:<9} {', '.join(r['shared_terms'])} | {r['title'][:60]}")
            print("Lexical suggestions only: judge each, then add bidirectional related_missions links.")

    elif args.command == "capture":
        entry = mgr.capture_inbox(args.idea, author=args.author, tags=args.tags)
        print(f"Captured inbox item: {entry['id']}")

    elif args.command == "compile":
        mgr.compile()
        print("Compiled missions into index.jsonl and generated backlog.md successfully.")

    elif args.command == "validate":
        errors = mgr.validate()
        if errors:
            print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Validation passed. All mission manifests conform to schema and invariants.")

    elif args.command == "record-result":
        updated = mgr.record_result(
            args.id,
            args.verdict,
            args.evidence,
            citation=args.citation,
            lessons=args.lessons,
            agent=args.agent
        )
        print(f"Recorded verdict '{updated['verdict']}' for {args.id}. Backlog recompiled.")


if __name__ == "__main__":
    main()
