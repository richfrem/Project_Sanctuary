#!/usr/bin/env python3
"""Driver for the MISSION-060 prospective phases. Orchestration only: every decision rule lives in thresholds.json, choose.py and analyze.py.

usage: run_study.py snapshot | calibrate | ideas | score | declare | item ITEM-NN | status   (selection is choose.py)
"""
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import prompts as P
from common import CLIENTS, NEUTRAL, STUDY, extract_json, run_model, sha256_bytes, sha256_file, usage_totals, item_usage, validate_declaration
from pool import T, containment, kb_docs

REPO = STUDY.parents[2]
# Use the installed main-checkout client context for subscription probes.
# The parent must also have a valid CLI PATH; cwd cannot repair a missing binary.
QUOTA_CWD = pathlib.Path(subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True, check=True).stdout.strip()).parent
PROBE = REPO / "plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/usage_probe.py"
COORD = REPO / "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py"
SNAP = pathlib.Path("/private/tmp/study060/snapshot")
WORK = pathlib.Path("/private/tmp/study060/work")
BACKEND_OF = {"sonnet": "claude", "haiku": "claude", "flash": "agy", "luna": "codex"}


# ------------------------------------------------------------------ stop rules (approved package)
def headroom():
    out = subprocess.run([sys.executable, str(PROBE), "headroom"], cwd=QUOTA_CWD, capture_output=True, text=True, check=True).stdout
    values = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\w+)\s+status=ok\s+tightest_window_remaining=([\d.]+)%", out, re.M)}
    if not {"claude", "agy", "codex"}.issubset(values):
        raise SystemExit("STOP RULE: backend headroom unavailable; quota is unknown")
    return values


def claude_5h_remaining():
    out = subprocess.run([sys.executable, str(PROBE), "calibrate"], cwd=QUOTA_CWD, capture_output=True, text=True).stdout
    try:
        return float(json.loads(out)["runway"]["remaining_pct"])
    except Exception:
        return None


def check_stop(key):
    h = headroom()
    low = {b: v for b, v in h.items() if v < T["stop_rules"]["min_backend_window_remaining_pct"]}
    if low:
        raise SystemExit(f"STOP RULE: backend window below {T['stop_rules']['min_backend_window_remaining_pct']}%: {low}. Pausing; ask the human partner.")
    start = json.load(open(STUDY / "stop-baseline.json")) if (STUDY / "stop-baseline.json").exists() else None
    if BACKEND_OF[key] == "claude":
        if not start or start.get("claude_5h_remaining_pct") is None:
            raise SystemExit("STOP RULE: Claude study-share baseline unavailable")
        now = claude_5h_remaining()
        if now is None:
            raise SystemExit("STOP RULE: current Claude study-share usage unavailable")
        if start["claude_5h_remaining_pct"] - now > T["stop_rules"]["claude_5h_study_share_max_pct"]:
            raise SystemExit(f"STOP RULE: study has used more than {T['stop_rules']['claude_5h_study_share_max_pct']} points of the Claude 5h window.")


def heartbeat():
    subprocess.run([sys.executable, str(COORD), "heartbeat", "--mission", "MISSION-060", "--agent", "codex", "--extend", "60"], capture_output=True, text=True, check=True)


def check_budget(next_call=None):
    path = STUDY / "usage-log.jsonl"
    log = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    total, unknown = usage_totals(log)
    if unknown:
        raise SystemExit(f"STOP RULE: usage unknown for {unknown}; repair attribution before continuing")
    if total >= T["total_additional_tokens_ceiling"]:
        raise SystemExit("STOP RULE: total study token budget reached")
    codes = {match.group(1) for record in log
             if (match := re.fullmatch(r"(?:work|succ|review)-(ITEM-\d{2})", record["call_id"]))}
    for code in codes:
        if item_usage(log, code)["known_subtotal"] > T["per_item_cap_additional_tokens"]:
            raise SystemExit(f"STOP RULE: composite item token cap exceeded: {code}")
    match = re.fullmatch(r"(?:work|succ|review)-(ITEM-\d{2})", next_call or "")
    if match and item_usage(log, match.group(1))["known_subtotal"] >= T["per_item_cap_additional_tokens"]:
        raise SystemExit("STOP RULE: no item token allowance remains for the next call")


def call(key, text, instruction, phase, call_id, **kw):
    check_stop(key)
    check_budget(call_id)
    heartbeat()
    r = run_model(key, text, instruction, phase, call_id, **kw)
    if not r["ok"]:
        raise SystemExit(f"model call failed: {call_id} rc={r['rc']} {r['stderr']}")
    check_budget()
    return r


# ------------------------------------------------------------------ phases
def cmd_snapshot():
    if SNAP.exists():
        shutil.rmtree(SNAP)
    SNAP.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD"], capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(SNAP)], input=tar, check=True)
    for rm in ("CLAUDE.md", "AGENTS.md", "GEMINI.md", "START_HERE.md", ".agent"):     # no bootstrap rules, no always-on Human Gate rules in workers
        p = SNAP / rm
        shutil.rmtree(p) if p.is_dir() else (p.unlink() if p.exists() else None)
    print("snapshot files:", sum(len(f) for _, _, f in os.walk(SNAP)))


def neutral_check():
    NEUTRAL.mkdir(parents=True, exist_ok=True)
    if not (NEUTRAL / ".git").exists():
        subprocess.run(["git", "init", "-q", str(NEUTRAL)], check=True)
    extra = [p.name for p in NEUTRAL.iterdir() if p.name != ".git"]
    if extra:
        raise SystemExit(f"neutral dir must be empty, found {extra}")


def cmd_calibrate():
    neutral_check()
    cases = {}
    for n in ("056", "057", "058"):
        txt = ""
        for p in sorted(glob.glob(str(STUDY.parent / f"mission-{n}" / "*.md"))):
            txt += f"\n=== FILE {pathlib.Path(p).name} ===\n" + open(p, encoding="utf-8").read()
        cases[n] = txt[:24000]
    res = {}
    raters = T["calibration_raters"]
    if len(raters) != 2 or CLIENTS[raters[0]]["family"] == CLIENTS[raters[1]]["family"]:
        raise ValueError("calibration requires two raters from different families")
    for rater in raters:
        for n, txt in cases.items():
            head = f"ITEM mission-{n}\nKIND: discovery\nQUESTION: see the artifacts below.\n\nARTIFACTS:\n{txt}"
            r = call(rater, head, P.CALIBRATION_INSTRUCTION, "calibration", f"cal-{rater}-{n}")
            res[(rater, n)] = extract_json(r["text"])["checks"]
    agree, total, detail = 0, 0, {}
    for n in cases:
        for ck in ("D1", "D2", "D3"):
            a, b = (res[(rater, n)][ck]["value"] for rater in raters)
            if type(a) is not bool or type(b) is not bool:
                raise ValueError("calibration judgments must be JSON booleans")
            agree += a == b
            total += 1
            detail[f"{n}.{ck}"] = dict(zip(raters, (a, b)))
    out = dict(agreement=agree / total, agree=agree, total=total, detail=detail, raters=raters, cases=list(cases))
    (STUDY / "calibration-results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("agreement", "agree", "total")}), "bound", T["calibration_agreement_min"])


def cmd_ideas():
    neutral_check()
    (STUDY / "ideas").mkdir(exist_ok=True)
    ctx = "A research habitat keeps durable records of what its AI agents learn so that later agents can build on it. Propose your inquiries below."
    for g in T["idea_generators"]:
        r = call(g, ctx, P.IDEA_INSTRUCTION, "ideas", f"ideas-{g}")
        ideas = extract_json(r["text"])
        assert isinstance(ideas, list) and len(ideas) == 2, "expected two ideas"
        for i, o in enumerate(ideas, 1):
            o["id"] = f"IDEA-{g.upper()}-{i}"
            o["proposed_by"] = g
        json.dump(ideas, open(STUDY / "ideas" / f"{g}.json", "w"), indent=1)
        print(g, [o["title"][:50] for o in ideas])


def idea_card(o):
    return dict(id=o["id"], kind=o["kind"], source="idea", title=o["title"], summary=o["question"] + " Why: " + o["why_it_attracts_me"],
                first_unit=o["first_work_unit"], deliverable=o["deliverable"], brief_checklist=o.get("brief_checklist"), proposed_by=o["proposed_by"])


def cmd_score():
    neutral_check()
    base = json.load(open(STUDY / "pool-base.json"))
    ideas = [idea_card(o) for g in T["idea_generators"] for o in json.load(open(STUDY / "ideas" / f"{g}.json"))]
    pool = base + ideas
    json.dump(pool, open(STUDY / "pool.json", "w"), indent=1)
    need = [c for c in pool if c["source"] in ("inbox", "idea")]
    text = json.dumps([dict(id=c["id"], title=c["title"], summary=c["summary"], first_unit=c["first_unit"]) for c in need], indent=1)
    r = call(T["scorer"], text, P.SCORE_INSTRUCTION, "formula-scoring", "score-nonmanifest")
    est = extract_json(r["text"])
    formula = json.load(open(STUDY / "formula-base.json"))
    for c in need:
        e = est[c["id"]]
        assert e["cost_weight"] in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0), c["id"]
        formula[c["id"]] = dict(strategic_leverage=int(e["strategic_leverage"]), epistemic_depth=int(e["epistemic_depth"]), cost_weight=e["cost_weight"],
                                operational_friction=int(e["operational_friction"]), source="estimated-by-" + T["scorer"], basis=e.get("basis", ""))
    json.dump(formula, open(STUDY / "formula.json", "w"), indent=1)
    print("pool", len(pool), "formula entries", len(formula))


def cmd_declare():
    neutral_check()
    pool = json.load(open(STUDY / "pool.json"))
    strip = [{k: c[k] for k in ("id", "kind", "title", "summary", "first_unit", "brief_checklist") if k in c and c[k] is not None} for c in pool]
    banned = {"score", "scores", "rank", "priority", "priority_score", "formula", "strategic_leverage", "epistemic_depth", "cost_weight"}
    assert not any(banned & set(c) for c in strip), "declaration input must carry no scores or ranks"
    text = json.dumps(strip, indent=1)
    (STUDY / "declarations").mkdir(exist_ok=True)
    stamps = {}
    for a in T["declarers"]:
        neutral_check()
        t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        r = call(a, text, P.RATE_INSTRUCTION, "declaration", f"declare-{a}", cwd=NEUTRAL)
        rows = extract_json(r["text"])
        validate_declaration(rows, pool, a, T["dimensions"])
        neutral_check()
        json.dump(rows, open(STUDY / "declarations" / f"{a}.json", "w"), indent=1)
        stamps[a] = dict(declared_at=t0, rated=len(rows), of=len(pool), sha256=sha256_file(STUDY / "declarations" / f"{a}.json"))
        print(a, "rated", len(rows), "of", len(pool))
    json.dump(dict(blind=True, neutral_cwd=str(NEUTRAL), input_sha256=sha256_bytes(text.encode()), input_keys=sorted({k for c in strip for k in c}),
                   basis="Input carried only id, kind, title, summary, first_unit and brief_checklist. Agents ran in an empty directory with no project files or hooks, before selection.py was run.",
                   declarations=stamps), open(STUDY / "declarations" / "attestation.json", "w"), indent=1)


def bundle(folder, limit=36000):
    txt, files = "", []
    for p in sorted(pathlib.Path(folder).rglob("*")):
        if p.is_file() and p.suffix in (".md", ".txt", ".py", ".json", ".csv", ".svg", ".html", ".tex", ".abc", ".ly"):
            body = p.read_text(encoding="utf-8", errors="replace")
            files.append(dict(path=str(p.relative_to(folder)), bytes=p.stat().st_size, sha256=sha256_file(p)))
            txt += f"\n=== FILE {p.relative_to(folder)} ({p.stat().st_size} bytes) ===\n{body[:9000]}"
    return txt[:limit], files


def cmd_item(code):
    neutral_check()
    sel = json.load(open(STUDY / "selection.json"))
    cid = next(k for k, v in sel["item_codes"].items() if v == code)
    pool = {c["id"]: c for c in json.load(open(STUDY / "pool.json"))}
    c = pool[cid]
    worker = sel["workers"][cid]
    if CLIENTS[worker]["family"] == CLIENTS[sel["reviewer"][cid]]["family"]:
        raise SystemExit("reviewer must be from a different family than the worker")
    wdir = WORK / code
    if wdir.exists():
        shutil.rmtree(wdir)
    shutil.copytree(SNAP, wdir, symlinks=True)
    (wdir / "deliverables").mkdir()
    brief_block = ("BRIEF (declared before work), your deliverable must meet all three: " + "; ".join(c["brief_checklist"]) + "\n") if c.get("brief_checklist") else ""
    brief = P.WORKER_TEMPLATE.format(title=c["title"], kind=c["kind"], summary=c["summary"], first_unit=c["first_unit"], brief_block=brief_block,
                                     creative_block=(P.CREATIVE_BLOCK if c["kind"] == "creative" else ""))
    before = subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout
    w = call(worker, brief, "Carry out the work unit described in the source, following its rules exactly.", "execution", f"work-{code}", cwd=wdir, isolated=False, timeout=1500,
             out_dir=STUDY / "raw")
    after = subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout
    if before != after:
        raise SystemExit("STOP RULE: worker changed repository status; inspect before continuing")
    dst = STUDY / "items" / code / "deliverables"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(wdir / "deliverables", dst) if (wdir / "deliverables").exists() else dst.mkdir()
    art, files = bundle(dst)
    det = dict(findings_present=(dst / "findings.md").exists() and (dst / "findings.md").stat().st_size > 200,
               reflection_present=(dst / "reflection.md").exists() and (dst / "reflection.md").stat().st_size > 120,
               brief_predeclared=bool(c.get("brief_checklist") and len(c["brief_checklist"]) == 3), files=files, repo_status_unchanged=(before == after))
    status = "BLOCKED" if "BLOCKED" in w["text"].upper().split()[-3:] else "DONE"
    if not files:
        r_ans = {f"Q{i}": "not stated" for i in range(1, 6)}
        review = dict(checks={}, notes="no deliverable files")
    else:
        sk, rk = sel["successor"][cid], sel["reviewer"][cid]
        s = call(sk, art, P.SUCCESSOR_INSTRUCTION, "successor", f"succ-{code}")
        r_ans = extract_json(s["text"])
        head = f"ITEM {code}\nKIND: {c['kind']}\nTITLE: {c['title']}\nFIRST WORK UNIT GIVEN: {c['first_unit']}\n" + (("BRIEF CRITERIA: " + "; ".join(c["brief_checklist"]) + "\n") if c.get("brief_checklist") else "")
        facts = json.dumps({k: det[k] for k in ("findings_present", "reflection_present", "brief_predeclared")}) + "\nFILES: " + json.dumps(files)
        rv = call(rk, f"{head}\nDETERMINISTIC FACTS: {facts}\n\nSUCCESSOR ANSWERS:\n{json.dumps(r_ans, indent=1)}\n\nARTIFACTS:\n{art}", P.REVIEW_INSTRUCTION, "review", f"review-{code}")
        review = extract_json(rv["text"])
    log = [json.loads(l) for l in open(STUDY / "usage-log.jsonl")]
    wu = next((r["usage"] for r in reversed(log) if r["call_id"] == f"work-{code}"), None)
    expected = None if files else {f"work-{code}"}
    composite = item_usage(log, code, expected=expected)
    (STUDY / "assess").mkdir(exist_ok=True)
    json.dump(dict(code=code, id=cid, kind=c["kind"], worker=worker, reviewer=sel["reviewer"][cid], worker_status=status, review=review, successor_answers=r_ans,
                   deterministic=det, worker_usage=wu, item_usage=composite), open(STUDY / "assess" / f"{code}.json", "w"), indent=1)
    print(code, cid, worker, status, "files", len(files), "tokens", (wu or {}).get("additional_tokens"))


def cmd_status():
    log = [json.loads(l) for l in open(STUDY / "usage-log.jsonl")] if (STUDY / "usage-log.jsonl").exists() else []
    known, unknown = usage_totals(log)
    print("calls", len(log), "additional_tokens", None if unknown else known, "known_subtotal", known,
          "calls_without_usage", unknown, "of", T["total_additional_tokens_ceiling"], "headroom", headroom())


if __name__ == "__main__":
    c = sys.argv[1]
    if c in {"snapshot", "ideas", "score", "declare", "item"}:
        from freeze_study import verify_freeze
        verify_freeze(require_integrated=True)
    globals()["cmd_" + c](*sys.argv[2:])
