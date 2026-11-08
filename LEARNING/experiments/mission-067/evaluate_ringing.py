#!/usr/bin/env python3
"""MISSION-067 evaluator: exact mixing of random change ringing vs lazy adjacent transposition on S_n.

Chain C: uniform over nonempty sets of disjoint adjacent transpositions (ringing "changes").
Chain A: lazy random adjacent transposition (stay 1/2, else uniform adjacent swap).
t_X(n) = least t with TV(P^t(rounds, .), uniform) <= 0.25. Stdlib only.
Usage: evaluate_ringing.py --mode dev|heldout
"""
import argparse, itertools, json, math, time

DEV, HELDOUT, EPS, TCAP = (4, 5, 6), (7, 8), 0.25, 5000


def changes(n):
    """All nonempty sets of disjoint adjacent swaps, as lists of left positions."""
    out = []
    def rec(i, cur):
        if i >= n - 1:
            if cur:
                out.append(list(cur))
            return
        rec(i + 1, cur)
        rec(i + 2, cur + [i])
    rec(0, [])
    return out


def fib(k):
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


def build(n, moves):
    perms = list(itertools.permutations(range(n)))
    idx = {p: i for i, p in enumerate(perms)}
    maps = []
    for m in moves:
        mp = []
        for p in perms:
            q = list(p)
            for i in m:
                q[i], q[i + 1] = q[i + 1], q[i]
            mp.append(idx[tuple(q)])
        maps.append(mp)
    return perms, maps


def tv(dist):
    u = 1.0 / len(dist)
    return 0.5 * sum(abs(x - u) for x in dist)


def mix(n, chain):
    if chain == "C":
        moves = changes(n)
        lazy, w = 0.0, 1.0 / len(moves)
    else:
        moves = [[i] for i in range(n - 1)]
        lazy, w = 0.5, 0.5 / (n - 1)
    perms, maps = build(n, moves)
    M = len(perms)
    d = [0.0] * M
    d[0] = 1.0  # rounds = identity
    hist, maxdrift = [], 0.0
    for t in range(1, TCAP + 1):
        new = [lazy * x for x in d]
        for mp in maps:
            for i, x in enumerate(d):
                if x:
                    new[mp[i]] += w * x
        d = new
        maxdrift = max(maxdrift, abs(sum(d) - 1))
        v = tv(d)
        hist.append(v)
        if v <= EPS:
            break
    else:
        return None
    swaps = sum(len(m) for m in moves) / len(moves) if chain == "C" else 0.5
    return {"t": t, "mean_swaps_per_step": swaps, "n_moves": len(moves),
            "tv_hist": [round(x, 6) for x in hist], "max_mass_drift": maxdrift,
            "monotone": all(hist[i + 1] <= hist[i] + 1e-12 for i in range(len(hist) - 1))}


def run(ns):
    res, gate = {}, {"fib_ok": True, "mass_ok": True, "A_monotone": True}
    for n in ns:
        c, a = mix(n, "C"), mix(n, "A")
        if c is None or a is None:
            res[n] = None
            continue
        gate["fib_ok"] &= c["n_moves"] == fib(n + 1) - 1
        gate["mass_ok"] &= max(c["max_mass_drift"], a["max_mass_drift"]) < 1e-9
        gate["A_monotone"] &= a["monotone"]
        res[n] = {"t_C": c["t"], "t_A": a["t"], "R": a["t"] / c["t"],
                  "s_C": c["mean_swaps_per_step"], "swaps_C": c["t"] * c["mean_swaps_per_step"],
                  "swaps_A": a["t"] * 0.5, "C_monotone": c["monotone"], "tv_C": c["tv_hist"]}
    gate["pass"] = all(v for v in gate.values()) and all(r is not None for r in res.values())
    return res, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    ns = DEV if a.mode == "dev" else (6,) + HELDOUT  # R(6) needed for H1
    res, gate = run(ns)
    out = {"mode": a.mode, "gate": gate, "results": res}
    if a.mode == "heldout" and gate["pass"]:
        r6, r7, r8 = (res[n]["R"] for n in (6, 7, 8))
        out["H1"] = r7 > r6 and r8 > r6 and r8 > r7
        out["H2"] = all(res[n]["swaps_C"] < res[n]["swaps_A"] for n in HELDOUT)
        out["verdict"] = "CORROBORATED" if out["H1"] and out["H2"] else "FALSIFIED"
    elif a.mode == "heldout":
        out["verdict"] = "UNASSESSED"
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
