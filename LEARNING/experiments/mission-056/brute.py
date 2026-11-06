"""Independent brute-force implementation (exact Fractions, plain tuples, no shared code with rhythm.py) for cross-checking tiny cases."""
import itertools
from fractions import Fraction


def expected_success_brute(template, n, p):
    """template: iterable of occupied positions. p: Fraction. Returns exact Fraction."""
    base = tuple(1 if i in set(template) else 0 for i in range(n))
    rots = {tuple(base[(i - s) % n] for i in range(n)) for s in range(n)}           # distinct rotated grids as 0/1 tuples
    k = sum(base)
    total = Fraction(0)
    for true in rots:
        beats = [i for i in range(n) if true[i]]
        for keep in itertools.product((0, 1), repeat=k):
            w = Fraction(1)
            obs = [0] * n
            for b, kp in zip(beats, keep):
                if kp:
                    obs[b] = 1; w *= (1 - p)
                else:
                    w *= p
            cands = [r for r in rots if all(r[i] >= obs[i] for i in range(n))]
            total += w * Fraction(1, len(cands))
    return total / len(rots)
