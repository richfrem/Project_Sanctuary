#!/usr/bin/env python3
"""Claim A: controlled selection scenarios (seeded, deterministic). F = formula only; P = readiness-exposed formula share + interest share.

What this CAN show: whether the P procedure keeps >=90% of the selected portfolio's epistemic depth and at least as many
high-interest picks while its readiness check reduces preventable blocked selections, GIVEN the generator's stated assumptions
(p_blocker, readiness_check_recall, interest weakly linked to depth). What it CANNOT show: that real readiness checks catch
70% of real blockers, or that real interest tracks the generator's. The blocker-reduction leg is driven mostly by the assumed
recall; the depth-retention and high-interest legs are the non-trivial ones. Sensitivity to recall is reported separately.
"""
import json
import pathlib
import random
import sys

STUDY = pathlib.Path(__file__).resolve().parent
T = json.load(open(STUDY / "thresholds.json"))
S = T["scenarios"]
COSTS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]


def formula(c):
    return c["l"] * c["d"] / (c["c"] * c["f"])


def make_situation(rng):
    cands = []
    for i in range(S["candidates_per_situation"]):
        l, d, f = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5)
        latent = S["interest_depth_weight"] * (d - 1) / 4 + (1 - S["interest_depth_weight"]) * rng.random()
        raters = [min(5.0, max(1.0, round(1 + 4 * latent + rng.gauss(0, 0.5)))) for _ in range(S["agents_per_situation"])]
        cands.append(dict(i=i, l=l, d=d, c=rng.choice(COSTS), f=f, interest=sum(raters) / len(raters), blocker=rng.random() < S["p_blocker"],
                          check_catches=rng.random()))      # uniform draw compared with the check's recall later
    return cands


def policy_f(cands):
    pick = sorted(cands, key=lambda c: (-formula(c), c["i"]))[:S["K"]]
    return pick


def policy_p(cands, recall):
    def caught(c):
        return c["blocker"] and c["check_catches"] < recall
    fchosen = []
    for c in sorted(cands, key=lambda c: (-formula(c), c["i"])):
        if len(fchosen) >= S["formula_n"]:
            break
        if not caught(c):                      # a caught blocker becomes an enabling step; the slot goes to the next candidate
            fchosen.append(c)
    ids = {c["i"] for c in fchosen}
    ichosen = []
    for c in sorted(cands, key=lambda c: (-c["interest"], c["i"])):
        if len(ichosen) >= S["interest_n"]:
            break
        if c["i"] not in ids and not caught(c):
            ichosen.append(c)
            ids.add(c["i"])
    return fchosen + ichosen


def run(seed, n, recall):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        cands = make_situation(rng)
        f, p = policy_f(cands), policy_p(cands, recall)
        rows.append(dict(
            f_blocked=sum(c["blocker"] for c in f), p_blocked=sum(c["blocker"] for c in p),
            f_depth=sum(c["d"] for c in f), p_depth=sum(c["d"] for c in p),
            f_hi=sum(c["interest"] >= S["high_interest_threshold"] for c in f), p_hi=sum(c["interest"] >= S["high_interest_threshold"] for c in p)))
    return rows


def summarize(rows):
    fb, pb = sum(r["f_blocked"] for r in rows), sum(r["p_blocked"] for r in rows)
    fd, pd = sum(r["f_depth"] for r in rows), sum(r["p_depth"] for r in rows)
    fh, ph = sum(r["f_hi"] for r in rows), sum(r["p_hi"] for r in rows)
    return dict(f_blocked=fb, p_blocked=pb, blocker_reduction=(None if fb == 0 else 1 - pb / fb), depth_retention=(pd / fd if fd else None),
                f_high_interest=fh, p_high_interest=ph)


def meets(s):
    A = T["claim_A"]
    if s["blocker_reduction"] is None:
        return None
    return (s["blocker_reduction"] >= A["blocker_reduction_min"] and s["depth_retention"] >= A["depth_retention_min"]
            and s["p_high_interest"] >= s["f_high_interest"])


def bootstrap(rows):
    rng = random.Random(T["claim_A"]["bootstrap_seed"])
    ok = n = 0
    for _ in range(T["claim_A"]["bootstrap_resamples"]):
        samp = [rows[rng.randrange(len(rows))] for _ in rows]
        m = meets(summarize(samp))
        if m is not None:
            n += 1
            ok += bool(m)
    return ok / n if n else None


def evaluate(split):
    seed, n = (S["seed_heldout"], S["n_heldout"]) if split == "heldout" else (S["seed_dev"], S["n_dev"])
    rows = run(seed, n, S["readiness_check_recall"])
    s = summarize(rows)
    boot = bootstrap(rows)
    if s["blocker_reduction"] is None:
        verdict = "UNSCORABLE (baseline F has no preventable blocked selections)"
    elif not meets(s):
        verdict = "FALSIFIED (point estimates miss a threshold)"
    elif boot is not None and boot >= T["claim_A"]["bootstrap_pass_probability_min"]:
        verdict = "MET"
    else:
        verdict = "UNDECIDED (thresholds met by point estimate but bootstrap pass probability below the frozen bound)"
    sens = {str(r): summarize(run(seed, n, r)) for r in S["sensitivity_recall"]}
    return dict(split=split, seed=seed, n_situations=n, recall=S["readiness_check_recall"], summary=s, bootstrap_pass_probability=boot, claim_A=verdict,
                sensitivity_to_recall=sens)


def main(split="dev"):
    if split not in {"dev", "heldout"}:
        raise ValueError("unknown scenario split")
    out = STUDY / f"scenario-results-{split}.json"
    if split == "heldout":
        from freeze_study import verify_freeze
        verify_freeze(require_integrated=True)
        if out.exists():
            raise ValueError("held-out result already exists; refusing a repeat")
    res = evaluate(split)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("split", "n_situations", "summary", "bootstrap_pass_probability", "claim_A")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
