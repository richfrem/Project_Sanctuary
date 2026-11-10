#!/usr/bin/env python3
"""run.py - MISSION-071: three consecutive live smoke runs through the installed cli-agent skill.

Writes results.json (per-run checks, tokens, cost, ledger delta) and prints the verdict
from the frozen falsification criteria in MISSION-071.json.
"""
import hashlib, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.expanduser("~/.claude/skills/cli-agent/scripts")
SMOKE = os.path.join(SKILL, "smoke_test_openrouter.py")
TASK = os.path.join(HERE, "task.json")
RUNS = 3
MAX_COST = 0.0001


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    for f in ("jev_router.py", "openrouter_client.py", "smoke_test_openrouter.py"):
        if not os.path.isfile(os.path.join(SKILL, f)):
            sys.exit(f"ABANDONED: installed cli-agent lacks {f}")
    freeze = {"task_sha256": sha(TASK), "evaluator_sha256": sha(SMOKE), "runs": RUNS, "max_cost_usd": MAX_COST,
              "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(freeze, open(os.path.join(HERE, "freeze.json"), "w"), indent=2)

    runs = []
    for i in range(1, RUNS + 1):
        p = subprocess.run([sys.executable, "-I", SMOKE, "--task", TASK], capture_output=True, text=True)
        out = p.stdout
        if out.startswith("SKIP"):
            sys.exit("ABANDONED: no OPENROUTER_API_KEY")
        m = re.search(r"tokens=(\d+)in/(\d+)out cost=\$([\d.]+) latency=([\d.]+)s fallback=(\w+)", out)
        v = re.search(r"\[verify\] route=(\w+) output=(\w+) live_openrouter=(\w+) ledger_incremented=(\w+)", out)
        cost = float(m.group(3)) if m else None
        r = {"run": i, "exit": p.returncode, "prompt_tokens": int(m.group(1)) if m else None,
             "completion_tokens": int(m.group(2)) if m else None, "cost_usd": cost,
             "latency_s": float(m.group(4)) if m else None, "fallback": m.group(5) if m else None,
             "checks": dict(zip(("route", "output", "live", "ledger"), v.groups())) if v else None,
             "cost_ok": cost is not None and cost <= MAX_COST, "stdout": out.strip()}
        r["pass"] = p.returncode == 0 and r["cost_ok"]
        runs.append(r)
        print(f"run {i}: {'PASS' if r['pass'] else 'FAIL'} cost=${cost} checks={r['checks']}")

    verdict = "CORROBORATED" if all(r["pass"] for r in runs) else "FALSIFIED"
    results = {"mission": "MISSION-071", "verdict": verdict, "freeze": freeze, "runs": runs,
               "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in runs), 8)}
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(f"VERDICT: {verdict}  total_cost=${results['total_cost_usd']}")
    sys.exit(0 if verdict == "CORROBORATED" else 1)


if __name__ == "__main__":
    main()
