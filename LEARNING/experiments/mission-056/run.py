"""MISSION-056 runner. `python3 run.py dev` (debug only) or `python3 run.py heldout` (refuses unless the frozen files match freeze.json).
The decision rule is fixed here and in protocol.md: primary = mean over (n,k) pairs of [even - mean irregular] success at p=0.25;
FALSIFIED if primary < 0.05, otherwise the hypothesis is supported within the tested domain."""
import hashlib, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rhythm as R

THRESHOLD, PS = 0.05, (0.10, 0.25, 0.50)
FROZEN = ["rhythm.py", "brute.py", "test_checks.py", "run.py", "protocol.md"]


def pairs(ns):
    return [(n, k) for n in ns for k in range(3, min(8, n - 1) + 1)]


def sha(p):
    return hashlib.sha256((HERE / p).read_bytes()).hexdigest()


def summarize(results):
    undefined = [(r["n"], r["k"]) for r in results if r.get("undefined")]
    results = [r for r in results if not r.get("undefined")]       # pairs with no irregular template cannot enter the average; they are reported
    d25 = [r["diff"]["0.25"] for r in results]
    primary = sum(d25) / len(d25)
    sym = [r for r in results if r["even"]["distinct_rotations"] < r["n"]]
    asym = [r for r in results if r["even"]["distinct_rotations"] >= r["n"]]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {"pairs": len(results), "undefined_pairs_excluded": undefined, "primary_mean_difference_p0.25": primary, "threshold": THRESHOLD,
            "verdict": "FALSIFIED" if primary < THRESHOLD else "NOT_FALSIFIED",
            "pairs_even_better": sum(x > 0 for x in d25), "pairs_even_worse": sum(x < 0 for x in d25),
            "per_pair_difference_min": min(d25), "per_pair_difference_max": max(d25),
            "secondary": {"mean_diff_p0.10": mean([r["diff"]["0.1"] for r in results]), "mean_diff_p0.50": mean([r["diff"]["0.5"] for r in results]),
                          "even_symmetric_pairs": len(sym), "mean_diff_p0.25_symmetric_even": mean([r["diff"]["0.25"] for r in sym]),
                          "even_asymmetric_pairs": len(asym), "mean_diff_p0.25_asymmetric_even": mean([r["diff"]["0.25"] for r in asym]),
                          "mean_clustered_minus_irregular_p0.25": mean([r["clustered"]["success"]["0.25"] - r["irregular"]["mean_success"]["0.25"] for r in results]),
                          "mean_empty_observation_prob_p0.25": mean([0.25 ** r["k"] for r in results]),
                          "pairs_with_fewer_than_40_irregular": [(r["n"], r["k"], r["irregular"]["count"]) for r in results if r["irregular"]["count"] < 40]}}


def main(mode):
    if mode == "heldout":
        fr = json.loads((HERE / "freeze.json").read_text())
        bad = [f for f in FROZEN if fr["sha256"].get(f) != sha(f)]
        if bad:
            raise SystemExit(f"REFUSING held-out run: frozen files changed since the freeze: {bad}")
    ns = range(8, 13) if mode == "dev" else range(13, 25)
    t0 = time.time()
    res = [R.pair_result(n, k, PS) for n, k in pairs(ns)]
    out = {"mode": mode, "seed": R.SEED, "wall_seconds": round(time.time() - t0, 2), "summary": summarize(res), "pairs": res}
    (HERE / f"results-{mode}.json").write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(mode, "pairs", s["pairs"], "primary", round(s["primary_mean_difference_p0.25"], 5), s["verdict"], "wall", out["wall_seconds"], "s")


if __name__ == "__main__":
    main(sys.argv[1])
