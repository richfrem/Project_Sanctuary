#!/usr/bin/env python3
"""Independent exact dense Markov solver for MISSION-072.

Purpose: build reset transitions directly and solve hitting-time equations.
Usage: imported by validation/evaluation. Does not use discounted.py or its renewal formula.
Key Input Dependencies: Python Fraction; finite reflected nearest-neighbor state space.
Key Functions: solve(), transition_matrix(), baseline(), reset_time().
"""
from fractions import Fraction as F


# Dense exact Gauss-Jordan elimination is intentionally different from a tridiagonal recurrence.
def solve(matrix: list[list[F]], rhs: list[F]) -> list[F]:
    """Solve the full rational augmented system with exact pivoting."""
    n = len(rhs)
    rows = [row[:] + [rhs[i]] for i,row in enumerate(matrix)]
    for col in range(n):
        pivot = next((i for i in range(col,n) if rows[i][col]),None)
        if pivot is None:
            raise ArithmeticError('singular direct Markov system')
        rows[col],rows[pivot] = rows[pivot],rows[col]
        scale = rows[col][col]
        rows[col] = [v/scale for v in rows[col]]
        for i in range(n):
            if i == col or not rows[i][col]:
                continue
            factor = rows[i][col]
            rows[i] = [v-factor*w for v,w in zip(rows[i],rows[col])]
    return [row[-1] for row in rows]


# Construct the substochastic walk matrix from actual sites, excluding absorbing site zero.
def transition_matrix(length: int) -> list[list[F]]:
    """Return the transient walk matrix with a deterministic reflection at L."""
    matrix = [[F(0) for _ in range(length)] for _ in range(length)]
    for site in range(1,length+1):
        destinations = [(site-1,F(1))] if site == length else [(site-1,F(1,2)),(site+1,F(1,2))]
        for destination,probability in destinations:
            if destination != 0:
                matrix[site-1][destination-1] += probability
    return matrix


# Solve both baseline moment systems directly, without the discounted-survival recurrence.
def baseline(length: int) -> tuple[list[F], list[F]]:
    """Return exact baseline mean and second moment by full-matrix equations."""
    transition = transition_matrix(length)
    matrix = [[F(i==j)-transition[i][j] for j in range(length)] for i in range(length)]
    mean = solve(matrix,[F(1)]*length)
    rhs = [1+2*sum(transition[i][j]*mean[j] for j in range(length)) for i in range(length)]
    return mean,solve(matrix,rhs)


# Put the reset transition into every row and its time charge directly into the RHS.
def reset_time(length: int, start: int, r: F, cost: F) -> F:
    """Solve m_i=(1-r)+r*c+(1-r)sum_j Pij*m_j+r*m_start and return m_start."""
    transition = transition_matrix(length)
    matrix = [[F(i==j)-(1-r)*transition[i][j]-(r if j==start-1 else 0)
               for j in range(length)] for i in range(length)]
    return solve(matrix,[1-r+r*cost]*length)[start-1]
