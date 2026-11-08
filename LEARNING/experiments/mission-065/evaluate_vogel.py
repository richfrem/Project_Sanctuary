#!/usr/bin/env python3
"""MISSION-065 evaluator: is the golden angle the finite-N optimum of a Vogel spiral?

Vogel spiral: point k = 1..N at polar (r = sqrt(k), theta = 2*pi*k*t), t = divergence
angle as a fraction of a full turn. Metric: d_min(t, N) = minimum pairwise Euclidean
distance (larger = more uniform packing; all t share the same radial profile).

Modes:
  --mode dev      instrument checks on N in {25, 50} (brute force vs grid hash,
                  analytic t = 1/2 case). Never touches held-out N.
  --mode heldout  frozen confirmatory run on N in {100, 200, 400, 800}; verifies
                  freeze.json hashes first.

Standard library only.
"""
import argparse, hashlib, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PHI = (1 + 5 ** 0.5) / 2
T_GOLD = 2 - PHI  # 0.381966..., 137.5078 degrees
DEV_N = [25, 50]
HELDOUT_N = [100, 200, 400, 800]
GLOBAL_N = [100, 200]          # full-range scan done only here
WINDOW_DEG = 1.0               # local window half-width around golden
CELL = 2.5                     # grid-hash cell size; valid iff d_min < CELL


def points(n, t):
    out = []
    for k in range(1, n + 1):
        r = math.sqrt(k)
        a = 2 * math.pi * k * t
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def dmin_brute(pts):
    best = float("inf")
    for i in range(len(pts)):
        xi, yi = pts[i]
        for j in range(i + 1, len(pts)):
            d = (xi - pts[j][0]) ** 2 + (yi - pts[j][1]) ** 2
            if d < best:
                best = d
    return math.sqrt(best)


def dmin_grid(pts):
    grid = {}
    for idx, (x, y) in enumerate(pts):
        grid.setdefault((math.floor(x / CELL), math.floor(y / CELL)), []).append(idx)
    best = float("inf")
    for (cx, cy), members in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                other = grid.get((cx + dx, cy + dy))
                if not other:
                    continue
                for i in members:
                    xi, yi = pts[i]
                    for j in other:
                        if j <= i:
                            continue
                        d = (xi - pts[j][0]) ** 2 + (yi - pts[j][1]) ** 2
                        if d < best:
                            best = d
    best = math.sqrt(best)
    if best >= CELL:  # grid search not guaranteed exact; fall back
        return dmin_brute(pts)
    return best


def dmin(n, t):
    return dmin_grid(points(n, t))


def step_for(n):
    # largest point moves ~2*pi*N*sqrt(N)*dt; keep that <= 0.05 length units
    return 0.05 / (2 * math.pi * n ** 1.5)


def scan(n, lo, hi, step):
    best_t, best_d, k = None, -1.0, 0
    t = lo
    while t <= hi:
        d = dmin(n, t)
        if d > best_d:
            best_t, best_d = t, d
        t = lo + (k := k + 1) * step
    return best_t, best_d, k


def refine(n, t0, step, rounds=3):
    best_t, best_d = t0, dmin(n, t0)
    for _ in range(rounds):
        step /= 10
        for i in range(-20, 21):
            t = best_t + i * step
            d = dmin(n, t)
            if d > best_d:
                best_t, best_d = t, d
    return best_t, best_d


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_dev():
    checks = []
    for n in DEV_N:
        for t in [T_GOLD, 0.2, 0.3, 0.4142, 0.45, 0.5, 0.01]:
            p = points(n, t)
            b, g = dmin_brute(p), dmin_grid(p)
            checks.append({"n": n, "t": t, "brute": b, "grid": g, "ok": abs(b - g) < 1e-12})
    # analytic t = 1/2: points on a line, alternating sides; same-side neighbours k, k+2
    # and opposite-side ones are farther; d_min = min_k sqrt(k+2) - sqrt(k) at largest k
    analytic = []
    for n in DEV_N:
        expect = math.sqrt(n) - math.sqrt(n - 2)
        got = dmin(n, 0.5)
        analytic.append({"n": n, "expect": expect, "got": got, "ok": abs(expect - got) < 1e-9})
    ok = all(c["ok"] for c in checks) and all(a["ok"] for a in analytic)
    # dev landscape glance (development only)
    dev_scan = {}
    for n in DEV_N:
        w = WINDOW_DEG / 360
        bt, bd, evals = scan(n, T_GOLD - w, T_GOLD + w, step_for(n))
        bt, bd = refine(n, bt, step_for(n))
        dev_scan[n] = {"golden_dmin": dmin(n, T_GOLD), "window_best_t": bt,
                       "window_best_deg": bt * 360, "window_best_dmin": bd, "evals": evals}
    out = {"mode": "dev", "gate_ok": ok, "checks": checks, "analytic": analytic, "dev_scan": dev_scan}
    json.dump(out, open(os.path.join(HERE, "dev_results.json"), "w"), indent=2)
    print(json.dumps({"gate_ok": ok, "dev_scan": dev_scan}, indent=2))


def run_heldout():
    freeze = json.load(open(os.path.join(HERE, "freeze.json")))
    for name, h in freeze["sha256"].items():
        if sha(os.path.join(HERE, name)) != h:
            sys.exit(f"FREEZE VIOLATION: {name} hash mismatch")
    t0 = time.time()
    res = {"mode": "heldout", "per_n": {}}
    for n in HELDOUT_N:
        # gate re-check on held-out sizes at golden and one other angle
        gate = all(abs(dmin_brute(points(n, t)) - dmin(n, t)) < 1e-12 for t in (T_GOLD, 0.3))
        g = dmin(n, T_GOLD)
        w = WINDOW_DEG / 360
        st = step_for(n)
        bt, bd, evals = scan(n, T_GOLD - w, T_GOLD + w, st)
        bt, bd = refine(n, bt, st)
        row = {"gate_ok": gate, "golden_dmin": g, "window_best_t": bt,
               "window_best_deg": bt * 360, "offset_deg": (bt - T_GOLD) * 360,
               "window_best_dmin": bd, "ratio_best_over_golden": bd / g,
               "window_evals": evals, "step": st}
        if n in GLOBAL_N:
            gt, gd, gev = scan(n, 0.02, 0.5, st)
            gt, gd = refine(n, gt, st)
            row.update({"global_best_t": gt, "global_best_deg": gt * 360,
                        "global_best_dmin": gd, "global_evals": gev,
                        "global_in_window": abs(gt - T_GOLD) * 360 <= WINDOW_DEG})
        res["per_n"][n] = row
        print(n, json.dumps(row), flush=True)
    P = res["per_n"]
    gate = all(r["gate_ok"] for r in P.values())
    h1 = all(r["ratio_best_over_golden"] >= 1.01 for r in P.values())
    h2 = all(abs(r["offset_deg"]) <= 0.1 for r in P.values())
    h3 = all(P[n]["global_in_window"] for n in GLOBAL_N)
    res.update({"gate_ok": gate, "H1": h1, "H2": h2, "H3": h3,
                "verdict": ("UNASSESSED" if not gate else
                            "CORROBORATED" if (h1 and h2 and h3) else "FALSIFIED"),
                "elapsed_seconds": round(time.time() - t0, 2)})
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("gate_ok", "H1", "H2", "H3", "verdict", "elapsed_seconds")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    a = ap.parse_args()
    run_dev() if a.mode == "dev" else run_heldout()
