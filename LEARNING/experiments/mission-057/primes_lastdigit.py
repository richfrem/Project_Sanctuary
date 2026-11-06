"""Last-digit transitions between consecutive primes, and a Cramer random control (MISSION-057)."""
import numpy as np

DIGITS = (1, 3, 7, 9)


def sieve(limit):
    """All primes <= limit (numpy, odd-only). Returns int64 array."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    n = (limit - 1) // 2                       # index i represents the odd number 2*i+3
    flags = np.ones(n, dtype=bool)
    i = 0
    while True:
        p = 2 * i + 3
        if p * p > limit:
            break
        if flags[i]:
            flags[(p * p - 3) // 2::p] = False
        i += 1
    return np.concatenate(([2], 2 * np.nonzero(flags)[0] + 3)).astype(np.int64)


def window(primes, lo, hi):
    a, b = np.searchsorted(primes, lo, "left"), np.searchsorted(primes, hi, "left")
    return primes[a:b]


def transitions(seq):
    """4x4 count matrix C[a][b] of (last digit of p_i = DIGITS[a], last digit of p_{i+1} = DIGITS[b]) over consecutive elements of seq."""
    last = (seq % 10).astype(np.int64)
    idx = {d: j for j, d in enumerate(DIGITS)}
    lut = np.full(10, -1, dtype=np.int64)
    for d, j in idx.items():
        lut[d] = j
    a, b = lut[last[:-1]], lut[last[1:]]
    ok = (a >= 0) & (b >= 0)
    C = np.zeros((4, 4), dtype=np.int64)
    np.add.at(C, (a[ok], b[ok]), 1)
    return C


def same_digit_probs(C):
    return {d: float(C[j, j] / C[j].sum()) for j, d in enumerate(DIGITS)}


def cramer_set(lo, hi, rng):
    """Integers in [lo,hi) with last digit in {1,3,7,9}, each included independently with probability 2.5/ln n
    (the prime density among integers coprime to 10)."""
    n = np.arange(lo, hi, dtype=np.int64)
    n = n[np.isin(n % 10, DIGITS)]
    keep = rng.random(len(n)) < 2.5 / np.log(n.astype(np.float64))
    return n[keep]


def trial_division_primes(lo, hi):
    out = []
    for m in range(max(lo, 2), hi):
        if m % 2 == 0 and m != 2:
            continue
        r = int(m ** 0.5)
        if all(m % q for q in range(3, r + 1, 2)):
            out.append(m)
    return np.array(out, dtype=np.int64)


def cramer_same_digit_formula(q):
    """Analytic probability that, in the Cramer model with candidate-inclusion probability q (candidates are the integers coprime to 10,
    visited in cyclic order 1,3,7,9,1,...), the next included candidate has the SAME last digit as the current one: q(1-q)^3 / (1-(1-q)^4).
    It tends to 0.25 only as q -> 0; at finite q it is smaller, so 0.25 is NOT the right null for primes of finite size."""
    return q * (1 - q) ** 3 / (1 - (1 - q) ** 4)
