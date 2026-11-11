#!/usr/bin/env python3
"""Exact discounted-survival solver for a reflected finite search walk.

Purpose: compute mean/second moment and mean reset time without a reset matrix.
Usage: imported by validation.py and evaluate.py; Python standard library only.
Key Input Dependencies: Fraction, L>=1, reset probability 0<=r<1, total reset cost c>=0.
Key Functions: tridiagonal(), survival_sum(), moments(), restart_mean(), local_predicts(), costs().
"""
from fractions import Fraction as F


# Solve the narrow-band system by exact forward elimination and back substitution.
def tridiagonal(lower: list[F], diagonal: list[F], upper: list[F], rhs: list[F]) -> list[F]:
    """Return the exact solution, refusing a zero pivot rather than approximating."""
    n = len(diagonal)
    d, b = diagonal[:], rhs[:]
    for i in range(1,n):
        if d[i-1] == 0:
            raise ArithmeticError('zero tridiagonal pivot')
        ratio = lower[i]/d[i-1]
        d[i] -= ratio*upper[i-1]
        b[i] -= ratio*b[i-1]
    answer = [F(0) for _ in range(n)]
    answer[-1] = b[-1]/d[-1]
    for i in range(n-2,-1,-1):
        answer[i] = (b[i]-upper[i]*answer[i+1])/d[i]
    return answer


# The reflected boundary has one deterministic left step; the target's value is zero.
def survival_sum(length: int, r: F, rhs: list[F] | None = None) -> list[F]:
    """Solve (I-(1-r)P)u=rhs for transient sites 1..L with exact rationals."""
    if length < 1 or not F(0) <= r < F(1):
        raise ValueError('invalid length or restart probability')
    z = 1-r
    lower = [F(0)] + [-z/F(2) for _ in range(length-1)]
    upper = [-z/F(2) for _ in range(length-1)] + [F(0)]
    if length > 1:
        lower[-1] = -z
    return tridiagonal(lower,[F(1)]*length,upper,rhs or [F(1)]*length)


# Baseline hitting-time recursions supply both moments without sampling paths.
def moments(length: int) -> tuple[list[F], list[F]]:
    """Return E[T] and E[T^2] for every start in a reset-free walk."""
    mean = survival_sum(length,F(0))
    second = survival_sum(length,F(0),[2*m-1 for m in mean])
    return mean,second


# Account explicitly for whether a reset consumes time instead of a walk step.
def restart_mean(u: F, r: F, cost: F) -> F:
    """Return [(1-r)+r*c]u/(1-r*u); c is total reset duration, not extra delay."""
    if cost < 0 or not F(0) <= r < F(1):
        raise ValueError('invalid reset cost or probability')
    denominator = 1-r*u
    if denominator <= 0:
        raise ArithmeticError('invalid survival resolvent denominator')
    return (1+r*(cost-1))*u/denominator


# A negative first derivative gives a local prediction, whose global grid extension is tested.
def local_predicts(mean: F, second: F, cost: F) -> bool:
    """Return whether mu^2-E[T(T-1)]/2+(c-1)mu is strictly negative."""
    return mean*mean-(second-mean)/2+(cost-1)*mean < 0


# Prespecified costs include the moment-critical boundary and offsets, without fitting reset data.
def costs(mean: F, second: F) -> list[F]:
    """Return the frozen unique nonnegative cost set for one baseline start."""
    critical = 1+(second-mean)/(2*mean)-mean
    return sorted({F(0),F(1),F(2),F(4),F(8),max(critical-F(1,2),F(0)),
                   max(critical,F(0)),max(critical+F(1,2),F(0))})
