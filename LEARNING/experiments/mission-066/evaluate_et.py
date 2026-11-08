#!/usr/bin/env python3
"""MISSION-066 evaluator: 5-limit worst-case error records of N-tone equal temperaments.

E(N) = max over r in {3/2, 5/4, 6/5} of |1200*round(N*log2 r)/N - 1200*log2 r| cents.
N is a record if E(N) < E(M) for all M < N. Stdlib only.
Usage: evaluate_et.py --mode dev|heldout
"""
import argparse, json, math

INTERVALS = {"3/2": 3 / 2, "5/4": 5 / 4, "6/5": 6 / 5}
DEV = (1, 30)
HELDOUT = (31, 200)
H2_SET = {31, 34, 53, 65, 87, 118, 171}


def errors(n):
    out = {}
    for name, r in INTERVALS.items():
        c = 1200 * math.log2(r)
        out[name] = 1200 * round(n * math.log2(r)) / n - c
    return out


def worst(n):
    return max(abs(e) for e in errors(n).values())


def records(lo, hi):
    """Running-minimum scan over 1..hi; returns records with lo <= N <= hi."""
    best, recs = math.inf, []
    for n in range(1, hi + 1):
        e = worst(n)
        if e < best:
            best = e
            if n >= lo:
                recs.append(n)
    return recs


def records_brute(lo, hi):
    return [n for n in range(lo, hi + 1) if all(worst(n) < worst(m) for m in range(1, n))]


def gate():
    e12 = errors(12)
    expect = {"3/2": 1.955, "5/4": 13.686, "6/5": 15.641}
    ok12 = all(abs(abs(e12[k]) - v) < 1e-3 for k, v in expect.items())
    okscan = records(*DEV) == records_brute(*DEV)
    return {"N12_errors": e12, "N12_ok": ok12, "scan_matches_brute_dev": okscan, "pass": ok12 and okscan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    a = ap.parse_args()
    g = gate()
    res = {"gate": g}
    lo, hi = DEV if a.mode == "dev" else HELDOUT
    recs = records(lo, hi)
    res["records"] = [{"N": n, "worst_cents": round(worst(n), 4),
                       "errors": {k: round(v, 4) for k, v in errors(n).items()},
                       "worst_steps": round(worst(n) * n / 1200, 4)} for n in recs]
    if a.mode == "heldout":
        h1 = 53 in recs and 118 in recs
        h2 = set(recs) <= H2_SET
        res["H1"], res["H2"] = h1, h2
        res["verdict"] = "UNASSESSED" if not g["pass"] else ("CORROBORATED" if h1 and h2 else "FALSIFIED")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
