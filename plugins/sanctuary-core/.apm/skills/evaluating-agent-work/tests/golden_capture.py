"""Capture PRE-migration behavior of score_contribution / track_budget / evaluate_dispatch / usage_probe (run on the unmodified code)."""
import json, sys, tempfile, itertools, io, contextlib, os
from pathlib import Path
S = Path(sys.argv[1]); sys.path.insert(0, str(S))
import score_contribution as sc, track_budget as tb, evaluate_dispatch as ed, usage_probe as up
out = {}
models = ["standard","gpt-6-astra","astra","claude-opus-5-5","opus","claude-fable-5-1","fable","gemini-4-argon","o1","o3","gpt-6.1-sol","sol","gpt-6-luna","luna","claude-sonnet-5-5","claude-3-5-sonnet","sonnet","gemini-pro","pro","gpt-4o","claude-haiku-5-5","claude-3-5-haiku","haiku","gemini-3.8-flash","gemini-flash","flash","gpt-4o-mini","mini","local-llama","llama","qwen","Qwen2.5-Coder-7B","unknown-model-x","","  Opus  "]
out["score_weights"] = {m: sc.get_model_cost_weight(m) for m in models}
out["dispatch_tiers"] = {m: ed.get_model_tier(m) for m in models}
cases = []
for r in [(0,0,0,0,0),(3,3,3,3,3),(1,2,3,0,1),(2,2,2,1,3),(0,3,0,3,0),(1,1,1,1,1)]:
    for m in ["opus","haiku","standard","claude-fable-5-1","qwen"]:
        dims = dict(zip(["useful_outcome","learning_depth","verification_rigor","token_economy","anti_asch_integrity"], r))
        raw = sum(dims.values()); fit = round(raw/15.0,3); w = sc.get_model_cost_weight(m); roi = round(fit/w,3)
        bloat = None
        if w >= 3.0 and (r[3] <= 1 or r[0] <= 1): bloat = True
        cases.append({"ratings": r, "model": m, "raw": raw, "fitness": fit, "weight": w, "roi": roi, "bloat": bool(bloat)})
out["scores"] = cases
# projection rebuild
assess = []
for i,(agent,r) in enumerate([("a",(3,3,3,3,3)),("b",(1,2,3,0,1)),("a",(2,2,2,1,3)),("c",(0,0,0,0,0))]):
    dims = dict(zip(["useful_outcome","learning_depth","verification_rigor","token_economy","anti_asch_integrity"], r))
    assess.append({"assessment_id": f"id{i}", "timestamp": f"2026-01-0{i+1}T00:00:00+00:00", "task_id": f"t{i}", "agent_id": agent, "raw_score": sum(r), "dimensions": dims, "cognitive_fitness": round(sum(r)/15,3), "notes": f"n{i}"})
out["assessments"] = assess
out["projection"] = sc.rebuild_projection(list(reversed(assess)))
# track_budget CLI via patched root
def run_tb(argv, root):
    tb.get_project_root = lambda: root
    old = sys.argv; sys.argv = ["track_budget.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): tb.main()
    finally: sys.argv = old
    return buf.getvalue()
root = Path(tempfile.mkdtemp())
for a in (["record","--task-id","t1","--backend","codex","--model","gpt-6-luna","--tokens-est","1000","--turns","2","--summary","s","--format","json"],
          ["record","--task-id","t1","--backend","claude","--model","haiku","--tokens-est","500","--source","measured","--task-class","edit","--tokens-in","400","--tokens-out","100","--quality","2","--format","json"],
          ["record","--task-id","t2","--backend","agy","--tokens-est","50","--format","json"]):
    run_tb(a, root)
led = json.loads((root/".agent/learning/budget_ledger.json").read_text())
for rec in led["recent_records"]: rec["record_id"]="X"; rec["timestamp"]="T"
led["updated_at"]="T"
out["budget_ledger"] = led
out["budget_check"] = json.loads(run_tb(["check","--profile","claude-pro","--planned-turns","20","--format","json"], root))
out["budget_check_ok"] = json.loads(run_tb(["check","--profile","gemini-pro","--planned-turns","3","--format","json"], root))
out["budget_profiles"] = tb.BUDGET_PROFILES
# usage probe pure functions
out["cost_usd"] = {"none": up.cost_usd("no-such-model", 1000, 1000, prices={}), "priced": up.cost_usd("m", 1000000, 500000, prices={"m": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "cached_input_per_mtok": 0.3}}, cache_read=2000000)}
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
print("goldens", len(json.dumps(out)))

# real CLI path for score_contribution (root patched to a temp dir)
cli = []
for r, m in [((3,3,3,3,3),"haiku"),((1,2,3,0,1),"opus"),((0,1,0,1,0),"claude-fable-5-1"),((2,2,2,2,2),"standard")]:
    root = Path(tempfile.mkdtemp()); sc.get_project_root = lambda root=root: root
    old = sys.argv
    sys.argv = ["score_contribution.py","--task-id","T","--agent-id","ag","--model",m,"--useful-outcome",str(r[0]),"--learning-depth",str(r[1]),"--verification-rigor",str(r[2]),"--token-economy",str(r[3]),"--anti-asch-integrity",str(r[4]),"--notes","n","--format","json"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): sc.main()
    sys.argv = old
    j = json.loads(buf.getvalue()); j["assessment_id"]="X"; j["timestamp"]="T"
    led = json.loads((root/".agent/learning/cognitive_ledger.json").read_text()); led["updated_at"]="T"
    for x in led["recent_assessments"]: x["assessment_id"]="X"; x["timestamp"]="T"
    ev = json.loads((root/".agent/learning/events.jsonl").read_text().splitlines()[0]); ev["assessment_id"]="X"; ev["timestamp"]="T"
    cli.append({"in": [list(r), m], "stdout": j, "ledger": led, "event": ev})
out["score_cli"] = cli
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
print("goldens+cli", len(json.dumps(out)))
