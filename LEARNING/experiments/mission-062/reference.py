#!/usr/bin/env python3
"""Independent literal-subset sumset counter for MISSION-062.

Purpose: verify the optimized normalized-shape C instrument with iterative set sums.
Usage: imported by evaluate.py; no model calls and no translation normalization.
Key Input Dependencies: Python standard library, integer h/k/q bounds.
Key Functions: sumset(), histograms(), candidates(), ok(), threshold().
"""
from collections import Counter
from itertools import combinations
from math import comb
import time


# Build hA by repeated Minkowski addition of the original actual subset.
def sumset(values: tuple[int, ...], h: int) -> set[int]:
    """Return all sums of h elements with repetition using iterative Python sets."""
    sums = {0}
    for _ in range(h):
        sums = {total + value for total in sums for value in values}
    return sums


# Enumerate each original subset once, bucket by maximum, then cumulate over q.
def histograms(h: int, k: int, qmax: int, deadline: float | None = None) -> dict[int, dict[int, int]]:
    """Return exact f_q(size) from literal subsets, independently of the C algorithm."""
    by_max = {q: Counter() for q in range(k, qmax + 1)}
    for position, values in enumerate(combinations(range(1, qmax + 1), k)):
        if position % 1024 == 0 and deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('reference computation budget exhausted')
        by_max[values[-1]][len(sumset(values, h))] += 1
    cumulative = Counter()
    result = {}
    for q in range(k, qmax + 1):
        cumulative.update(by_max[q])
        result[q] = dict(sorted(cumulative.items()))
    return result


# Candidate sizes follow the paper's formula, with comb(n,k)=0 when n<k.
def candidates(h: int, k: int) -> list[int]:
    """Return the h candidate sizes P_l without imposing an order of frequencies."""
    return [comb(h + k - 1, k - 1) - comb(ell + k - 2, k - 1) for ell in range(h)]


# Strict dominance over every unlisted size is the frozen empirical decision rule.
def ok(histogram: dict[int, int], h: int, k: int) -> bool:
    """Return false on missing candidates or a tie with any unlisted size."""
    listed = candidates(h, k)
    outside = max((count for size, count in histogram.items() if size not in listed), default=0)
    return all(histogram.get(size, 0) > outside for size in listed)


# This is a finite verified threshold through the supplied maximum, not asymptotic proof.
def threshold(tables: dict[int, dict[int, int]], h: int, k: int) -> int | None:
    """Return one past the last failed q if a nonempty passing suffix exists."""
    failures = [q for q, table in tables.items() if not ok(table, h, k)]
    first = max(failures) + 1 if failures else min(tables)
    return first if first <= max(tables) else None
