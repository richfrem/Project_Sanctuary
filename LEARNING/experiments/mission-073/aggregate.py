#!/usr/bin/env python3
"""Burnside aggregation for MISSION-073. Reads burnside JSON lines, prints exact orbit counts.

Usage: aggregate.py burnside_n3.jsonl > summary.json
Orbit count for a class of sets = (1/|G|) * sum_g |Fix(g) in class|. Classes: k = number of axes
spanned (k=1..n). Integrality is checked for every (group, k) sum. For the identity element the
per-span-mask counts must depend only on the number of axes in the mask (axis-permutation symmetry).
"""
import json, sys
from math import factorial

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = rows[0]["n"]
assert len(rows) == factorial(n) * 2 ** n, len(rows)
out = {"n": n, "group_order_full": len(rows)}
ident = [r for r in rows if r["perm"] == list(range(n)) and r["flip"] == 0]
assert len(ident) == 1
ic = ident[0]["cnt"]
sym_ok = True
for k in range(1, n + 1):
    vals = {ic[a] for a in range(1 << n) if bin(a).count("1") == k}
    if len(vals) != 1:
        sym_ok = False
out["identity_axis_symmetry_ok"] = sym_ok
out["identity_count_by_k"] = {k: sum(ic[a] for a in range(1 << n) if bin(a).count("1") == k) for k in range(1, n + 1)}
for name, sel in (("rotations", [r for r in rows if r["det"] == 1]), ("all_symmetries", rows)):
    res = {"group_order": len(sel), "orbits_by_k": {}, "burnside_sum_by_k": {}, "integral": True}
    for k in range(1, n + 1):
        s = sum(r["cnt"][a] for r in sel for a in range(1 << n) if bin(a).count("1") == k)
        res["burnside_sum_by_k"][k] = s
        if s % len(sel):
            res["integral"] = False
        res["orbits_by_k"][k] = s // len(sel)
    res["orbits_k_ge_n_minus_1"] = sum(res["orbits_by_k"][k] for k in (n - 1, n))
    out[name] = res
print(json.dumps(out, indent=1))
