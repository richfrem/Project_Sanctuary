"""MISSION-057 runner: `python3 run.py dev` (debug window, no verdict) or `python3 run.py heldout` (refuses unless frozen files match freeze.json)."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import primes_lastdigit as P

FROZEN = ["primes_lastdigit.py", "test_checks.py", "run.py", "protocol.md"]
SMALL, LARGE, DEV = (10 ** 5, 10 ** 6), (10 ** 7, 10 ** 8), (10 ** 4, 10 ** 5)
REPS = {SMALL: 20, LARGE: 5, DEV: 20}
MIN_LARGE_REPULSION, FORMULA_TOL, SEED0 = 0.005, 0.003, 101


def sha(f):
    return hashlib.sha256((HERE / f).read_bytes()).hexdigest()


def measure(primes, win):
    lo, hi = win
    real = P.same_digit_probs(P.transitions(P.window(primes, lo, hi)))
    C = P.transitions(P.window(primes, lo, hi))
    reps = []
    for r in range(REPS[win]):
        reps.append(P.same_digit_probs(P.transitions(P.cramer_set(lo, hi, np.random.default_rng(SEED0 + r)))))
    ctrl = {d: float(np.mean([x[d] for x in reps])) for d in P.DIGITS}
    ctrl_sd = {d: float(np.std([x[d] for x in reps])) for d in P.DIGITS}
    n_real = len(P.window(primes, lo, hi))
    q_eff = float(np.mean([len(P.cramer_set(lo, hi, np.random.default_rng(SEED0 + r))) for r in range(2)]) / (0.4 * (hi - lo)))
    analytic = P.cramer_same_digit_formula(q_eff)
    return {"window": [lo, hi], "n_primes": n_real, "matrix_counts": C.tolist(),
            "same_digit_prob_real": {str(d): real[d] for d in P.DIGITS},
            "same_digit_prob_control_mean": {str(d): ctrl[d] for d in P.DIGITS}, "control_replicate_sd": {str(d): ctrl_sd[d] for d in P.DIGITS},
            "repulsion_beyond_control": {str(d): ctrl[d] - real[d] for d in P.DIGITS},          # positive = real primes repel their own last digit MORE than the random model
            "control_analytic_same_digit": analytic, "control_effective_inclusion_q": q_eff,
            "control_minus_analytic": {str(d): ctrl[d] - analytic for d in P.DIGITS}, "control_replicates": REPS[win],
            "row_probabilities_real": [[float(C[i, j] / C[i].sum()) for j in range(4)] for i in range(4)]}


def main(mode):
    if mode == "heldout":
        fr = json.loads((HERE / "freeze.json").read_text())
        bad = [f for f in FROZEN if fr["sha256"].get(f) != sha(f)]
        if bad:
            raise SystemExit(f"REFUSING held-out run: frozen files changed since the freeze: {bad}")
    t0 = time.time()
    primes = P.sieve(10 ** 8)
    wins = [DEV] if mode == "dev" else [SMALL, LARGE]
    res = {("small" if w == SMALL else "large" if w == LARGE else "dev"): measure(primes, w) for w in wins}
    out = {"mode": mode, "sieve_limit": 10 ** 8, "seed0": SEED0, "results": res}
    if mode == "heldout":
        s, l = res["small"], res["large"]
        h1 = all(l["repulsion_beyond_control"][str(d)] >= MIN_LARGE_REPULSION for d in P.DIGITS)
        h2 = all(s["repulsion_beyond_control"][str(d)] > l["repulsion_beyond_control"][str(d)] for d in P.DIGITS)
        h3 = all(abs(w["control_minus_analytic"][str(d)]) < FORMULA_TOL for w in (s, l) for d in P.DIGITS)
        verdict = "UNASSESSED" if not h3 else ("NOT_FALSIFIED" if (h1 and h2) else "FALSIFIED")
        out["decision"] = {"H1_large_window_repulsion_beyond_control_ge_0.5pp_every_digit": h1, "H2_repulsion_larger_small_than_large_every_digit": h2,
                           "H3_control_matches_analytic_formula_within_0.3pp": h3, "verdict": verdict}
    out["wall_seconds"] = round(time.time() - t0, 1)
    (HERE / f"results-{mode}.json").write_text(json.dumps(out, indent=1))
    for k, v in res.items():
        print(k, v["n_primes"], "real", {d: round(x, 4) for d, x in v["same_digit_prob_real"].items()}, "ctrl", {d: round(x, 4) for d, x in v["same_digit_prob_control_mean"].items()})
    print(out.get("decision"), out["wall_seconds"], "s")


if __name__ == "__main__":
    main(sys.argv[1])
