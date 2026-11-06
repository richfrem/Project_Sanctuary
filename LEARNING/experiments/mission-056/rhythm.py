"""Exact reconstruction probability of cyclic rhythms under random beat deletion (MISSION-056).

Model: n grid positions on a circle, a known template with k beats, unknown cyclic phase. The true pattern is one of the template's
DISTINCT rotations, chosen uniformly. Each beat is deleted independently with probability p; no false positives. The decoder knows n
and the grid origin, keeps the distinct rotations that contain every observed beat, and guesses uniformly among them, so its success
probability for an observation is 1/candidate_count. Expectations are computed EXACTLY by enumerating every true rotation and every
one of the 2^k deletion masks (weight p^deleted * (1-p)^kept); nothing here is a Monte-Carlo estimate.
"""
import itertools
import random
from math import comb

import numpy as np

SEED = 73185


def to_mask(positions):
    m = 0
    for p in positions:
        m |= 1 << p
    return m


def rotate(mask, n, s):
    full = (1 << n) - 1
    return ((mask << s) | (mask >> (n - s))) & full if s else mask


def distinct_rotations(mask, n):
    return sorted({rotate(mask, n, s) for s in range(n)})


def even_template(n, k):
    return [(j * n) // k for j in range(k)]


def clustered_template(n, k):
    return list(range(k))


def is_rotation_of(a_mask, b_mask, n):
    return a_mask in set(distinct_rotations(b_mask, n))


def positions_of(mask, n):
    return [i for i in range(n) if mask >> i & 1]


def cand_table(mask, n):
    """For every (distinct true rotation, deletion mask): (deleted_count, candidate_count). Arrays of length R * 2^k."""
    rots = np.array(distinct_rotations(mask, n), dtype=np.int64)
    k = bin(mask).count("1")
    idx = np.arange(1 << k, dtype=np.int64)
    kept_popcount = np.zeros(1 << k, dtype=np.int64)
    for j in range(k):
        kept_popcount += (idx >> j) & 1
    deleted = k - kept_popcount
    d_all, c_all = [], []
    for t in rots:
        beats = [1 << p for p in positions_of(int(t), n)]
        obs = np.zeros(1 << k, dtype=np.int64)
        for j, b in enumerate(beats):
            obs += b * ((idx >> j) & 1)
        cand = ((rots[None, :] & obs[:, None]) == obs[:, None]).sum(axis=1)
        d_all.append(deleted)
        c_all.append(cand)
    return len(rots), k, np.concatenate(d_all), np.concatenate(c_all)


def expected_success(mask, n, ps, table=None):
    """Exact expected success for each p in ps. Also returns diagnostics. Weights over masks of one rotation sum to 1."""
    R, k, d, c = table or cand_table(mask, n)
    out = {}
    for p in ps:
        w = (p ** d) * ((1 - p) ** (k - d))
        out[p] = float(np.sum(w / c) / R)
    mass = float(np.sum((0.25 ** d) * (0.75 ** (k - d))) / R)       # == 1 (probability mass of the masks of one rotation)
    return out, {"R": int(R), "k": int(k), "mask_mass_at_p_0.25": mass, "empty_obs_prob_at_p_0.25": 0.25 ** k}


def irregular_templates(n, k, count=40, seed=SEED):
    """Up to `count` distinct uniformly sampled k-subsets excluding rotations of the even template. Per-pair stream: Random(f"{seed}:{n}:{k}").
    If fewer eligible subsets exist, all of them are returned. Returns (list of position-lists, number_eligible_or_None)."""
    rng = random.Random(f"{seed}:{n}:{k}")
    even_rots = set(distinct_rotations(to_mask(even_template(n, k)), n))
    if comb(n, k) <= 20000:
        elig = [c for c in itertools.combinations(range(n), k) if to_mask(c) not in even_rots]
        picked = elig if len(elig) <= count else rng.sample(elig, count)
        return [list(c) for c in picked], len(elig)
    seen, picked = set(), []
    while len(picked) < count:
        c = tuple(sorted(rng.sample(range(n), k)))
        if c in seen or to_mask(c) in even_rots:
            continue
        seen.add(c); picked.append(list(c))
    return picked, None


def pair_result(n, k, ps=(0.10, 0.25, 0.50)):
    ev_mask = to_mask(even_template(n, k))
    ev, evd = expected_success(ev_mask, n, ps)
    irr, n_elig = irregular_templates(n, k)
    if not irr:          # every k-subset is a rotation of the even template (k = n-1): the comparison is undefined for this pair
        return {"n": n, "k": k, "undefined": True, "reason": "no irregular template exists (all k-subsets are rotations of the even template)"}
    irr_s = [expected_success(to_mask(t), n, ps) for t in irr]
    cl, cld = expected_success(to_mask(clustered_template(n, k)), n, ps)
    out = {"n": n, "k": k, "even": {"positions": even_template(n, k), "success": {str(p): ev[p] for p in ps}, "distinct_rotations": evd["R"]},
           "irregular": {"count": len(irr), "eligible_total": n_elig,
                         "mean_success": {str(p): float(np.mean([s[0][p] for s in irr_s])) for p in ps},
                         "min_success_at_0.25": float(min(s[0][0.25] for s in irr_s)), "max_success_at_0.25": float(max(s[0][0.25] for s in irr_s)),
                         "mean_distinct_rotations": float(np.mean([s[1]["R"] for s in irr_s])),
                         "templates_with_rotational_symmetry": int(sum(s[1]["R"] < n for s in irr_s))},
           "clustered": {"positions": clustered_template(n, k), "success": {str(p): cl[p] for p in ps}, "distinct_rotations": cld["R"]},
           "diff": {str(p): ev[p] - float(np.mean([s[0][p] for s in irr_s])) for p in ps}}
    return out
