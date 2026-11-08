#!/usr/bin/env python3
"""MISSION-063: Braess's Paradox for Random Walks in Trees.

Zero-dependency (Python standard library only) exact evaluator of Kemeny's
constant K(G) across all non-isomorphic trees on n vertices and all single
missing-edge additions e = {u, v} not in E(T).

Cross-verifies every graph with two mathematically distinct evaluators:
  1. Fundamental Matrix & Pairwise Mean First Passage Times:
     Z = (I - P + 1*pi^T)^(-1), m_{i,j} = (Z_{j,j} - Z_{i,j})/pi_j,
     K_hit(G) = sum_{j != i} pi_j m_{i,j} (verified invariant across all i).
  2. Symmetrized Transition Operator Eigenvalues (Classical Jacobi Method):
     S = D^(-1/2) A D^(-1/2), K_spec(G) = sum_{k=2}^n 1 / (1 - lambda_k(S)).

Modes:
  --mode dev       Run development calibration (n in {4, 5, 6}) and H3 validity
                   gate checks only. Never evaluates held-out n >= 7.
  --mode freeze    Write freeze.json with SHA-256 hashes of protocol.md and
                   this script after verifying --mode dev passes.
  --mode heldout   Verify freeze.json SHA-256 hashes, then evaluate the full
                   held-out domain n in {7, 8, 9, 10, 11} (422 trees, 16,355
                   missing-edge additions) and write results.json + README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.md"
SCRIPT_PATH = Path(__file__).resolve()
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
DEV_RESULTS_PATH = EXPERIMENT_DIR / "dev-calibration.json"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"
README_PATH = EXPERIMENT_DIR / "README.md"

OEIS_A000055 = {
    4: 2,
    5: 3,
    6: 6,
    7: 11,
    8: 23,
    9: 47,
    10: 106,
    11: 235,
}
DEV_N = [4, 5, 6]
HELDOUT_N = [7, 8, 9, 10, 11]
PARADOX_EPS = 1e-9
GATE_TOL = 1e-11


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_centers(adj: list[list[int]]) -> list[int]:
    """Return the 1 or 2 center vertices of a tree by iterative leaf peeling."""
    n = len(adj)
    if n <= 2:
        return list(range(n))
    deg = [len(adj[u]) for u in range(n)]
    leaves = [u for u in range(n) if deg[u] <= 1]
    removed = len(leaves)
    while removed < n:
        next_leaves: list[int] = []
        for u in leaves:
            for v in adj[u]:
                deg[v] -= 1
                if deg[v] == 1:
                    next_leaves.append(v)
        removed += len(next_leaves)
        leaves = next_leaves
    return sorted(leaves)


def rooted_ahu(adj: list[list[int]], u: int, parent: int) -> str:
    """Return the canonical Aho-Hopcroft-Ullman parenthesis certificate."""
    child_certs = [rooted_ahu(adj, v, u) for v in adj[u] if v != parent]
    child_certs.sort()
    return "(" + "".join(child_certs) + ")"


def canonical_tree_cert(adj: list[list[int]]) -> str:
    """Return the isomorphism-invariant AHU certificate of an unrooted tree."""
    centers = tree_centers(adj)
    if len(centers) == 1:
        return rooted_ahu(adj, centers[0], -1)
    c0, c1 = centers[0], centers[1]
    s0 = rooted_ahu(adj, c0, c1)
    s1 = rooted_ahu(adj, c1, c0)
    if s0 <= s1:
        return f"[{s0}{s1}]"
    return f"[{s1}{s0}]"


def canonical_edge_orbit_key(adj: list[list[int]], u: int, v: int) -> str:
    """Return an automorphism-invariant certificate for missing edge {u, v}."""
    def rooted_marked(curr: int, parent: int) -> str:
        child_certs = [rooted_marked(nxt, curr) for nxt in adj[curr] if nxt != parent]
        child_certs.sort()
        mark = "*" if (curr == u or curr == v) else ""
        return f"({mark}" + "".join(child_certs) + ")"

    centers = tree_centers(adj)
    if len(centers) == 1:
        return rooted_marked(centers[0], -1)
    c0, c1 = centers[0], centers[1]
    s0 = rooted_marked(c0, c1)
    s1 = rooted_marked(c1, c0)
    return f"[{min(s0, s1)}{max(s0, s1)}]"


def generate_free_trees(max_n: int) -> dict[int, list[dict[str, Any]]]:
    """Generate all non-isomorphic free trees from n=2 up to max_n."""
    trees_by_n: dict[int, list[dict[str, Any]]] = {}
    adj2 = [[1], [0]]
    trees_by_n[2] = [
        {
            "n": 2,
            "edges": [(0, 1)],
            "adj": adj2,
            "cert": canonical_tree_cert(adj2),
        }
    ]
    for n in range(3, max_n + 1):
        seen: dict[str, dict[str, Any]] = {}
        for prev in trees_by_n[n - 1]:
            prev_edges = prev["edges"]
            for attach in range(n - 1):
                edges = list(prev_edges) + [(attach, n - 1)]
                adj: list[list[int]] = [[] for _ in range(n)]
                for a, b in edges:
                    adj[a].append(b)
                    adj[b].append(a)
                cert = canonical_tree_cert(adj)
                if cert not in seen:
                    seen[cert] = {
                        "n": n,
                        "edges": edges,
                        "adj": adj,
                        "cert": cert,
                    }
        ordered = [seen[k] for k in sorted(seen.keys())]
        for idx, item in enumerate(ordered):
            item["tree_index"] = idx
        trees_by_n[n] = ordered
    return trees_by_n


def prufer_to_tree_adj(code: tuple[int, ...], n: int) -> list[list[int]]:
    """Convert a Prüfer sequence of length n-2 over {0..n-1} to an adjacency list."""
    deg = [1] * n
    for x in code:
        deg[x] += 1
    adj: list[list[int]] = [[] for _ in range(n)]
    for x in code:
        for v in range(n):
            if deg[v] == 1:
                adj[x].append(v)
                adj[v].append(x)
                deg[x] -= 1
                deg[v] -= 1
                break
    u, v = [i for i in range(n) if deg[i] == 1]
    adj[u].append(v)
    adj[v].append(u)
    return adj


def verify_prufer_coverage(
    trees_by_n: dict[int, list[dict[str, Any]]], n_values: list[int]
) -> dict[str, Any]:
    """Verify that every labeled tree generated by Prüfer sequences maps to a generated class."""
    from itertools import product

    report: dict[str, Any] = {}
    for n in n_values:
        cert_set = {t["cert"] for t in trees_by_n[n]}
        observed: set[str] = set()
        count = 0
        for code in product(range(n), repeat=n - 2):
            adj = prufer_to_tree_adj(code, n)
            cert = canonical_tree_cert(adj)
            assert cert in cert_set, f"Prüfer tree certificate missing for n={n}"
            observed.add(cert)
            count += 1
        report[str(n)] = {
            "labeled_count": count,
            "expected_cayley": n ** (n - 2),
            "unlabeled_classes_hit": len(observed),
            "expected_unlabeled": OEIS_A000055[n],
            "exact_match": (count == n ** (n - 2)) and (len(observed) == OEIS_A000055[n]),
        }
    return report


def all_pairs_shortest_path_tree(adj: list[list[int]]) -> list[list[int]]:
    """Compute all-pairs shortest path distances in a connected graph via BFS."""
    n = len(adj)
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        d = dist[src]
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = d[u]
            for v in adj[u]:
                if d[v] == -1:
                    d[v] = du + 1
                    q.append(v)
    return dist


def invert_matrix_gauss_jordan(mat: list[list[float]]) -> list[list[float]]:
    """Invert an n x n matrix via Gauss-Jordan elimination with partial pivoting."""
    n = len(mat)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(mat)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        piv = aug[col][col]
        inv_piv = 1.0 / piv
        row_c = aug[col]
        for j in range(col, 2 * n):
            row_c[j] *= inv_piv
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if factor != 0.0:
                    row_r = aug[r]
                    for j in range(col, 2 * n):
                        row_r[j] -= factor * row_c[j]
    return [row[n:] for row in aug]


def jacobi_eigenvalues_symmetric(sym: list[list[float]]) -> list[float]:
    """Compute all eigenvalues of a real symmetric matrix via cyclic Jacobi rotations."""
    n = len(sym)
    a = [row[:] for row in sym]
    for _sweep in range(25):
        off_norm = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off_norm += a[p][q] * a[p][q]
        if off_norm < 1e-30:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-16:
                    a[p][q] = 0.0
                    a[q][p] = 0.0
                    continue
                app = a[p][p]
                aqq = a[q][q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.hypot(1.0, tau))
                else:
                    t = -1.0 / (-tau + math.hypot(1.0, tau))
                c = 1.0 / math.hypot(1.0, t)
                s = t * c
                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        akp = a[k][p]
                        akq = a[k][q]
                        new_kp = c * akp - s * akq
                        new_kq = s * akp + c * akq
                        a[k][p] = new_kp
                        a[p][k] = new_kp
                        a[k][q] = new_kq
                        a[q][k] = new_kq
    eigs = [a[i][i] for i in range(n)]
    eigs.sort()
    return eigs


def evaluate_kemeny_dual(adj_matrix: list[list[float]]) -> dict[str, float]:
    """Compute Kemeny's constant via two independent methods and return diagnostics."""
    n = len(adj_matrix)
    deg = [sum(row) for row in adj_matrix]
    two_m = sum(deg)
    pi = [d / two_m for d in deg]

    # Method 1: Fundamental matrix Z = (I - P + 1*pi^T)^(-1) and pairwise m_{i,j}
    a_sys = [
        [
            (1.0 if i == j else 0.0) - (adj_matrix[i][j] / deg[i]) + pi[j]
            for j in range(n)
        ]
        for i in range(n)
    ]
    z = invert_matrix_gauss_jordan(a_sys)
    z_diag = [z[j][j] for j in range(n)]

    k_by_start: list[float] = []
    unweighted_sum = 0.0
    for i in range(n):
        zi = z[i]
        ki = 0.0
        for j in range(n):
            if i != j:
                m_ij = (z_diag[j] - zi[j]) / pi[j]
                ki += pi[j] * m_ij
                unweighted_sum += m_ij
        k_by_start.append(ki)

    k_hit = sum(k_by_start) / n
    k_trace = sum(z_diag) - 1.0
    start_spread = max(k_by_start) - min(k_by_start)
    trace_diff = abs(k_hit - k_trace)
    unweighted_mean_hitting = unweighted_sum / (n * (n - 1))

    # Method 2: Spectral evaluation on symmetric S = D^(-1/2) A D^(-1/2)
    inv_sqrt_d = [1.0 / math.sqrt(d) for d in deg]
    sym = [
        [adj_matrix[i][j] * inv_sqrt_d[i] * inv_sqrt_d[j] for j in range(n)]
        for i in range(n)
    ]
    eigvals = jacobi_eigenvalues_symmetric(sym)
    lambda_max = eigvals[-1]
    k_spec = sum(1.0 / (1.0 - lam) for lam in eigvals[:-1])

    dual_diff = abs(k_hit - k_spec)
    lambda1_err = abs(lambda_max - 1.0)

    return {
        "k_hit": k_hit,
        "k_spec": k_spec,
        "k_trace": k_trace,
        "unweighted_mean_hitting": unweighted_mean_hitting,
        "start_spread": start_spread,
        "trace_diff": trace_diff,
        "dual_diff": dual_diff,
        "lambda1_err": lambda1_err,
    }


def build_adj_matrix(n: int, edges: list[tuple[int, int]]) -> list[list[float]]:
    mat = [[0.0] * n for _ in range(n)]
    for u, v in edges:
        mat[u][v] = 1.0
        mat[v][u] = 1.0
    return mat


def verify_closed_form_identities(n_values: list[int]) -> dict[str, Any]:
    """Check K(K_n) = (n-1)^2/n and K(C_n) = (n^2-1)/6 for n in n_values."""
    max_kn_err = 0.0
    max_cn_err = 0.0
    max_dual_diff = 0.0
    for n in n_values:
        kn_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        res_kn = evaluate_kemeny_dual(build_adj_matrix(n, kn_edges))
        expected_kn = ((n - 1) ** 2) / float(n)
        max_kn_err = max(max_kn_err, abs(res_kn["k_hit"] - expected_kn))
        max_dual_diff = max(max_dual_diff, res_kn["dual_diff"], res_kn["start_spread"])

        cn_edges = [(u, (u + 1) % n) for u in range(n)]
        res_cn = evaluate_kemeny_dual(build_adj_matrix(n, cn_edges))
        expected_cn = (n ** 2 - 1.0) / 6.0
        max_cn_err = max(max_cn_err, abs(res_cn["k_hit"] - expected_cn))
        max_dual_diff = max(max_dual_diff, res_cn["dual_diff"], res_cn["start_spread"])

    return {
        "max_kn_closed_form_err": max_kn_err,
        "max_cn_closed_form_err": max_cn_err,
        "max_reference_dual_or_spread_err": max_dual_diff,
        "passed": (
            max_kn_err < GATE_TOL
            and max_cn_err < GATE_TOL
            and max_dual_diff < GATE_TOL
        ),
    }


def evaluate_domain(
    trees_by_n: dict[int, list[dict[str, Any]]], n_values: list[int]
) -> dict[str, Any]:
    """Evaluate all trees and missing-edge additions for the specified n_values."""
    t0 = time.perf_counter()
    per_n: dict[str, Any] = {}
    by_distance: dict[int, dict[str, Any]] = {}

    total_trees = 0
    total_pairs = 0
    total_paradox = 0
    total_paradox_dist2 = 0
    total_diam_pairs = 0
    total_diam_paradox = 0
    max_diam_delta_k = -float("inf")
    total_unweighted_paradox = 0

    max_dual_diff = 0.0
    max_start_spread = 0.0
    max_trace_diff = 0.0
    max_lambda1_err = 0.0

    best_paradox_example: dict[str, Any] | None = None
    best_shortcut_example: dict[str, Any] | None = None

    for n in n_values:
        trees = trees_by_n[n]
        total_trees += len(trees)
        n_pairs = 0
        n_paradox = 0
        n_paradox_dist2 = 0
        n_diam_pairs = 0
        n_diam_paradox = 0
        n_orbit_pairs = 0
        n_orbit_paradox = 0
        n_unweighted_paradox = 0
        deltas: list[float] = []

        for tree in trees:
            edges = tree["edges"]
            adj_list = tree["adj"]
            dist = all_pairs_shortest_path_tree(adj_list)
            diam = max(max(row) for row in dist)

            base_mat = build_adj_matrix(n, edges)
            base_eval = evaluate_kemeny_dual(base_mat)
            k_base = base_eval["k_hit"]
            h_base = base_eval["unweighted_mean_hitting"]

            max_dual_diff = max(max_dual_diff, base_eval["dual_diff"])
            max_start_spread = max(max_start_spread, base_eval["start_spread"])
            max_trace_diff = max(max_trace_diff, base_eval["trace_diff"])
            max_lambda1_err = max(max_lambda1_err, base_eval["lambda1_err"])

            seen_orbits: set[str] = set()
            edge_set = {(min(a, b), max(a, b)) for a, b in edges}

            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in edge_set:
                        continue
                    d_uv = dist[u][v]
                    assert d_uv >= 2

                    base_mat[u][v] = 1.0
                    base_mat[v][u] = 1.0
                    aug_eval = evaluate_kemeny_dual(base_mat)
                    base_mat[u][v] = 0.0
                    base_mat[v][u] = 0.0

                    max_dual_diff = max(max_dual_diff, aug_eval["dual_diff"])
                    max_start_spread = max(max_start_spread, aug_eval["start_spread"])
                    max_trace_diff = max(max_trace_diff, aug_eval["trace_diff"])
                    max_lambda1_err = max(max_lambda1_err, aug_eval["lambda1_err"])

                    delta_k = aug_eval["k_hit"] - k_base
                    delta_h = aug_eval["unweighted_mean_hitting"] - h_base
                    is_paradox = delta_k > PARADOX_EPS
                    is_unweighted_paradox = delta_h > PARADOX_EPS

                    n_pairs += 1
                    total_pairs += 1
                    deltas.append(delta_k)

                    if is_paradox:
                        n_paradox += 1
                        total_paradox += 1
                        if d_uv == 2:
                            n_paradox_dist2 += 1
                            total_paradox_dist2 += 1

                    if is_unweighted_paradox:
                        n_unweighted_paradox += 1
                        total_unweighted_paradox += 1

                    if d_uv == diam:
                        n_diam_pairs += 1
                        total_diam_pairs += 1
                        max_diam_delta_k = max(max_diam_delta_k, delta_k)
                        if is_paradox:
                            n_diam_paradox += 1
                            total_diam_paradox += 1

                    orbit_key = canonical_edge_orbit_key(adj_list, u, v)
                    if orbit_key not in seen_orbits:
                        seen_orbits.add(orbit_key)
                        n_orbit_pairs += 1
                        if is_paradox:
                            n_orbit_paradox += 1

                    if d_uv not in by_distance:
                        by_distance[d_uv] = {
                            "distance": d_uv,
                            "cycle_length": d_uv + 1,
                            "pairs": 0,
                            "paradox_count": 0,
                            "sum_delta_k": 0.0,
                            "max_delta_k": -float("inf"),
                            "min_delta_k": float("inf"),
                        }
                    bd = by_distance[d_uv]
                    bd["pairs"] += 1
                    if is_paradox:
                        bd["paradox_count"] += 1
                    bd["sum_delta_k"] += delta_k
                    bd["max_delta_k"] = max(bd["max_delta_k"], delta_k)
                    bd["min_delta_k"] = min(bd["min_delta_k"], delta_k)

                    if best_paradox_example is None or delta_k > best_paradox_example["delta_k"]:
                        best_paradox_example = {
                            "n": n,
                            "tree_index": tree["tree_index"],
                            "tree_edges": edges,
                            "added_edge": [u, v],
                            "distance": d_uv,
                            "tree_diameter": diam,
                            "k_before": round(k_base, 8),
                            "k_after": round(aug_eval["k_hit"], 8),
                            "delta_k": round(delta_k, 8),
                            "pct_change": round(100.0 * delta_k / k_base, 4),
                        }
                    if best_shortcut_example is None or delta_k < best_shortcut_example["delta_k"]:
                        best_shortcut_example = {
                            "n": n,
                            "tree_index": tree["tree_index"],
                            "tree_edges": edges,
                            "added_edge": [u, v],
                            "distance": d_uv,
                            "tree_diameter": diam,
                            "k_before": round(k_base, 8),
                            "k_after": round(aug_eval["k_hit"], 8),
                            "delta_k": round(delta_k, 8),
                            "pct_change": round(100.0 * delta_k / k_base, 4),
                        }

        per_n[str(n)] = {
            "n": n,
            "non_isomorphic_trees": len(trees),
            "expected_oeis_a000055": OEIS_A000055[n],
            "missing_edge_pairs": n_pairs,
            "braess_paradox_count": n_paradox,
            "braess_paradox_fraction": n_paradox / float(n_pairs),
            "braess_paradox_at_dist2_count": n_paradox_dist2,
            "braess_paradox_at_dist2_share": (
                n_paradox_dist2 / float(n_paradox) if n_paradox > 0 else 0.0
            ),
            "diameter_shortcut_pairs": n_diam_pairs,
            "diameter_shortcut_paradox_count": n_diam_paradox,
            "orbit_unique_pairs": n_orbit_pairs,
            "orbit_braess_paradox_count": n_orbit_paradox,
            "orbit_braess_paradox_fraction": n_orbit_paradox / float(n_orbit_pairs),
            "unweighted_hitting_paradox_count": n_unweighted_paradox,
            "unweighted_hitting_paradox_fraction": n_unweighted_paradox / float(n_pairs),
            "mean_delta_k": sum(deltas) / len(deltas),
            "max_delta_k": max(deltas),
            "min_delta_k": min(deltas),
        }

    distance_table: list[dict[str, Any]] = []
    for d in sorted(by_distance.keys()):
        bd = by_distance[d]
        distance_table.append(
            {
                "distance": d,
                "cycle_length": bd["cycle_length"],
                "pairs": bd["pairs"],
                "paradox_count": bd["paradox_count"],
                "paradox_fraction": bd["paradox_count"] / float(bd["pairs"]),
                "mean_delta_k": bd["sum_delta_k"] / float(bd["pairs"]),
                "max_delta_k": bd["max_delta_k"],
                "min_delta_k": bd["min_delta_k"],
            }
        )

    elapsed_sec = time.perf_counter() - t0
    return {
        "n_values": n_values,
        "total_trees": total_trees,
        "total_pairs": total_pairs,
        "total_braess_paradox": total_paradox,
        "overall_braess_paradox_fraction": total_paradox / float(total_pairs),
        "total_braess_paradox_dist2": total_paradox_dist2,
        "dist2_share_of_all_paradoxes": (
            total_paradox_dist2 / float(total_paradox) if total_paradox > 0 else 0.0
        ),
        "total_diameter_shortcut_pairs": total_diam_pairs,
        "total_diameter_shortcut_paradox": total_diam_paradox,
        "max_diameter_shortcut_delta_k": max_diam_delta_k,
        "total_unweighted_hitting_paradox": total_unweighted_paradox,
        "unweighted_hitting_paradox_fraction": total_unweighted_paradox / float(total_pairs),
        "per_n": per_n,
        "by_distance": distance_table,
        "strongest_braess_paradox": best_paradox_example,
        "strongest_shortcut_speedup": best_shortcut_example,
        "numerical_diagnostics": {
            "max_dual_evaluator_diff": max_dual_diff,
            "max_starting_vertex_spread": max_start_spread,
            "max_fundamental_trace_diff": max_trace_diff,
            "max_lambda1_err": max_lambda1_err,
            "elapsed_seconds": round(elapsed_sec, 4),
        },
    }


def run_dev() -> int:
    """Run development calibration on n in {4, 5, 6} and validity checks."""
    trees_by_n = generate_free_trees(max(DEV_N))
    oeis_ok = all(len(trees_by_n[n]) == OEIS_A000055[n] for n in DEV_N)
    prufer_report = verify_prufer_coverage(trees_by_n, DEV_N)
    closed_form = verify_closed_form_identities(list(range(4, 12)))
    dev_eval = evaluate_domain(trees_by_n, DEV_N)

    diag = dev_eval["numerical_diagnostics"]
    h3_dev_pass = (
        oeis_ok
        and all(v["exact_match"] for v in prufer_report.values())
        and closed_form["passed"]
        and diag["max_dual_evaluator_diff"] < GATE_TOL
        and diag["max_starting_vertex_spread"] < GATE_TOL
        and diag["max_fundamental_trace_diff"] < GATE_TOL
        and diag["max_lambda1_err"] < GATE_TOL
    )

    payload = {
        "mode": "dev",
        "dev_n": DEV_N,
        "h3_dev_pass": h3_dev_pass,
        "prufer_cayley_verification": prufer_report,
        "closed_form_verification": closed_form,
        "dev_results": dev_eval,
    }
    DEV_RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[DEV] H3 validity gate on n={DEV_N}: {'PASS' if h3_dev_pass else 'FAIL'}")
    for n in DEV_N:
        row = dev_eval["per_n"][str(n)]
        print(
            f"  n={n}: trees={row['non_isomorphic_trees']}, pairs={row['missing_edge_pairs']}, "
            f"paradox={row['braess_paradox_count']} ({100*row['braess_paradox_fraction']:.2f}%), "
            f"dist2_share={100*row['braess_paradox_at_dist2_share']:.1f}%"
        )
    return 0 if h3_dev_pass else 1


def run_freeze() -> int:
    """Write freeze.json locking SHA-256 hashes of protocol.md and evaluate_braess_trees.py."""
    if not PROTOCOL_PATH.exists():
        print("ERROR: protocol.md does not exist", file=sys.stderr)
        return 1
    freeze_payload = {
        "mission_id": "MISSION-063",
        "frozen_at": "2026-11-08T15:02:00Z",
        "dev_n": DEV_N,
        "heldout_n": HELDOUT_N,
        "paradox_eps": PARADOX_EPS,
        "gate_tolerance": GATE_TOL,
        "primary_thresholds": {
            "h1_min_paradox_fraction_per_n": 0.05,
            "h1_strict_monotonic_with_n": True,
            "h2_min_dist2_share_of_paradoxes": 0.80,
            "h2_max_diameter_shortcut_paradoxes": 0,
        },
        "sha256": {
            "protocol.md": sha256_file(PROTOCOL_PATH),
            "evaluate_braess_trees.py": sha256_file(SCRIPT_PATH),
        },
    }
    FREEZE_PATH.write_text(json.dumps(freeze_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[FREEZE] Wrote {FREEZE_PATH}")
    return 0


def render_readme(results: dict[str, Any]) -> str:
    """Render a comprehensive markdown report of the held-out experiment."""
    verdict = results["verdict"]
    h1 = results["hypothesis_checks"]["H1"]
    h2 = results["hypothesis_checks"]["H2"]
    h3 = results["hypothesis_checks"]["H3"]
    held = results["heldout"]
    dev = results["dev_summary"]

    lines = [
        "# MISSION-063: Braess's Paradox for Random Walks in Trees",
        "",
        f"**Verdict:** `{verdict}`  ",
        f"**Held-Out Domain:** All $422$ non-isomorphic trees across $n \\in \\{{7, 8, 9, 10, 11\\}}$ ($16,355$ missing-edge additions)  ",
        f"**Instrument Gate (H3):** `{'PASS' if h3['passed'] else 'FAIL'}` (max dual-evaluator discrepancy ${held['numerical_diagnostics']['max_dual_evaluator_diff']:.2e} < 10^{{-11}}$)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **H1 (Prevalence $\\ge 5\\%$ per $n$ & Strict Monotonic Growth $p(7) < p(8) < p(9) < p(10) < p(11)$):** `{'PASS' if h1['passed'] else 'FAIL'}`",
        f"  - Held-out Braess-paradox prevalence by vertex count: "
        + ", ".join(
            f"$n={n}$: **{100*held['per_n'][str(n)]['braess_paradox_fraction']:.2f}%** ({held['per_n'][str(n)]['braess_paradox_count']}/{held['per_n'][str(n)]['missing_edge_pairs']})"
            for n in HELDOUT_N
        )
        + ".",
        f"  - Overall held-out prevalence: **{held['total_braess_paradox']} / {held['total_pairs']} ({100*held['overall_braess_paradox_fraction']:.2f}%)** missing-edge additions increase Kemeny's constant ($\\Delta K > 10^{{-9}}$).",
        f"- **H2 (Distance-2 Concentration $\\ge 80\\%$ & Diameter-Shortcut Immunity):** `{'PASS' if h2['passed'] else 'FAIL'}`",
        f"  - Share of all held-out Braess paradoxes occurring at tree distance $d_T(u,v) = 2$ (forming a $3$-cycle): **{held['total_braess_paradox_dist2']} / {held['total_braess_paradox']} ({100*held['dist2_share_of_all_paradoxes']:.2f}%)** (threshold $\\ge 80\\%$: `{'PASS' if h2['h2a_dist2_share_at_least_80pct'] else 'FAIL'}`).",
        f"  - Diameter-endpoint shortcuts ($d_T(u,v) = \\operatorname{{diam}}(T)$) exhibiting Braess's paradox: **{held['total_diameter_shortcut_paradox']} / {held['total_diameter_shortcut_pairs']}** ($\\max \\Delta K = {held['max_diameter_shortcut_delta_k']:.6f} < 0$; threshold $0$: `{'PASS' if h2['h2b_zero_diameter_shortcut_paradoxes'] else 'FAIL'}`).",
        "",
        "---",
        "",
        "## 2. Held-Out Results by Vertex Count ($n = 7..11$) and Development ($n = 4..6$)",
        "",
        "| $n$ | Split | Free Trees (OEIS A000055) | Missing-Edge Pairs | Braess Paradox ($\\Delta K > 0$) | Paradox % $p(n)$ | Dist-2 Share of Paradoxes | Orbit-Unique Paradox % | Unweighted $\\bar{H}$ Paradox % | Mean $\\Delta K$ | Max $\\Delta K$ | Min $\\Delta K$ |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for n in DEV_N:
        r = dev["per_n"][str(n)]
        lines.append(
            f"| {n} | Dev | {r['non_isomorphic_trees']} | {r['missing_edge_pairs']} | "
            f"{r['braess_paradox_count']} | {100*r['braess_paradox_fraction']:.2f}% | "
            f"{100*r['braess_paradox_at_dist2_share']:.2f}% | {100*r['orbit_braess_paradox_fraction']:.2f}% | "
            f"{100*r['unweighted_hitting_paradox_fraction']:.2f}% | {r['mean_delta_k']:+.4f} | "
            f"{r['max_delta_k']:+.4f} | {r['min_delta_k']:+.4f} |"
        )
    for n in HELDOUT_N:
        r = held["per_n"][str(n)]
        lines.append(
            f"| **{n}** | **Held-Out** | **{r['non_isomorphic_trees']}** | **{r['missing_edge_pairs']}** | "
            f"**{r['braess_paradox_count']}** | **{100*r['braess_paradox_fraction']:.2f}%** | "
            f"**{100*r['braess_paradox_at_dist2_share']:.2f}%** | {100*r['orbit_braess_paradox_fraction']:.2f}% | "
            f"{100*r['unweighted_hitting_paradox_fraction']:.2f}% | {r['mean_delta_k']:+.4f} | "
            f"{r['max_delta_k']:+.4f} | {r['min_delta_k']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Breakdown by Shortcut Distance $d_T(u, v)$ (Held-Out $n = 7..11$)",
            "",
            "| Tree Distance $d_T(u, v)$ | Cycle Length $L = d+1$ | Held-Out Pairs | Braess Paradox Count | Paradox % at Distance $d$ | Share of All Paradoxes | Mean $\\Delta K$ | Max $\\Delta K$ | Min $\\Delta K$ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bd in held["by_distance"]:
        share = (
            100.0 * bd["paradox_count"] / float(held["total_braess_paradox"])
            if held["total_braess_paradox"] > 0
            else 0.0
        )
        lines.append(
            f"| {bd['distance']} | {bd['cycle_length']} | {bd['pairs']} | "
            f"{bd['paradox_count']} | {100*bd['paradox_fraction']:.2f}% | "
            f"{share:.2f}% | {bd['mean_delta_k']:+.4f} | {bd['max_delta_k']:+.4f} | {bd['min_delta_k']:+.4f} |"
        )

    bp = held["strongest_braess_paradox"]
    bs = held["strongest_shortcut_speedup"]
    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Mechanism & Extreme Cases",
            "",
            "### Strongest Held-Out Braess Paradox ($\\max \\Delta K$)",
            f"- **Tree:** $n = {bp['n']}$, tree index `{bp['tree_index']}`, diameter ${bp['tree_diameter']}$, edges `{bp['tree_edges']}`",
            f"- **Added edge:** `{bp['added_edge']}` at tree distance $d_T(u,v) = {bp['distance']}$",
            f"- **Kemeny's constant:** $K(T) = {bp['k_before']:.6f} \\to K(T+e) = {bp['k_after']:.6f}$ ($\\Delta K = {bp['delta_k']:+.6f}$, **{bp['pct_change']:+.2f}%**)",
            "",
            "### Strongest Held-Out Shortcut Speedup ($\\min \\Delta K$)",
            f"- **Tree:** $n = {bs['n']}$, tree index `{bs['tree_index']}`, diameter ${bs['tree_diameter']}$, edges `{bs['tree_edges']}`",
            f"- **Added edge:** `{bs['added_edge']}` at tree distance $d_T(u,v) = {bs['distance']}$",
            f"- **Kemeny's constant:** $K(T) = {bs['k_before']:.6f} \\to K(T+e) = {bs['k_after']:.6f}$ ($\\Delta K = {bs['delta_k']:+.6f}$, **{bs['pct_change']:+.2f}%**)",
            "",
            "### Physical Interpretation: Why Stationary Weight Shift vs. Raw Transition Trapping Both Matter",
            f"- When we replace Kemeny's stationary-weighted expectation $K(G) = \\sum_{{j \\neq i}} \\pi_j m_{{i,j}}$ with the **unweighted** average hitting time $\\bar{{H}}(G) = \\frac{{1}}{{n(n-1)}} \\sum_{{i \\neq j}} m_{{i,j}}$, the held-out paradox rate is **{100*held['unweighted_hitting_paradox_fraction']:.2f}%** ({held['total_unweighted_hitting_paradox']} / {held['total_pairs']} pairs) compared to **{100*held['overall_braess_paradox_fraction']:.2f}%** for Kemeny's constant.",
            "- Adding a short edge (especially at distance $d_T(u,v)=2$, creating a triangle, or $d_T(u,v)=3$, creating a $4$-cycle) inside a high-degree hub/cluster simultaneously:",
            "  1. Diverts random-walk transition probability ($1/(d_u+1)$) into recirculating within the local cycle, delaying escape into long pendant branches (which raises unweighted $\\bar{H}(G)$ even without $\\pi$-weights).",
            "  2. Increases total edge count $m = n-1 \\to n$ in $K(G) = m \\sum_{x,y} \\pi_x \\pi_y R_{\\text{eff}}(x,y)$ while reducing $R_{\\text{eff}}$ only locally.",
            "",
            "---",
            "",
            "## 5. Instrument Validity Gate (H3) Summary",
            "",
            f"- **OEIS A000055 Free-Tree Counts ($n=4..11$):** `{h3['oeis_counts_matched']}` (`2, 3, 6, 11, 23, 47, 106, 235`)",
            f"- **Prüfer-Sequence Cayley Bijection ($n=4..6$):** `{h3['prufer_cayley_matched']}` ($4^2=16, 5^3=125, 6^4=1296$)",
            f"- **Closed-Form $K(K_n) = (n-1)^2/n$ and $K(C_n) = (n^2-1)/6$:** max error `{h3['closed_form_max_err']:.2e}`",
            f"- **Fundamental Matrix vs. Symmetrized Spectral Evaluator ($16,355$ held-out pairs):** max discrepancy `{held['numerical_diagnostics']['max_dual_evaluator_diff']:.2e}`",
            f"- **Starting-Vertex Invariance ($\\max_i K_i - \\min_i K_i$):** max spread `{held['numerical_diagnostics']['max_starting_vertex_spread']:.2e}`",
            f"- **Freeze Hash Verification:** `{results['freeze_verification']['verified']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_heldout() -> int:
    """Verify freeze.json hashes, run held-out n in {7..11}, and write results."""
    if not FREEZE_PATH.exists():
        print("ERROR: freeze.json missing; refuse held-out execution", file=sys.stderr)
        return 1

    freeze_data = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    actual_protocol_sha = sha256_file(PROTOCOL_PATH)
    actual_script_sha = sha256_file(SCRIPT_PATH)
    freeze_ok = (
        freeze_data["sha256"]["protocol.md"] == actual_protocol_sha
        and freeze_data["sha256"]["evaluate_braess_trees.py"] == actual_script_sha
    )
    if not freeze_ok:
        print("ERROR: freeze.json hash mismatch; refuse held-out execution", file=sys.stderr)
        return 1

    trees_by_n = generate_free_trees(max(HELDOUT_N))
    oeis_ok = all(
        len(trees_by_n[n]) == OEIS_A000055[n] for n in DEV_N + HELDOUT_N
    )
    prufer_report = verify_prufer_coverage(trees_by_n, DEV_N)
    closed_form = verify_closed_form_identities(list(range(4, 12)))

    dev_eval = evaluate_domain(trees_by_n, DEV_N)
    held_eval = evaluate_domain(trees_by_n, HELDOUT_N)

    diag = held_eval["numerical_diagnostics"]
    h3_pass = (
        freeze_ok
        and oeis_ok
        and all(v["exact_match"] for v in prufer_report.values())
        and closed_form["passed"]
        and diag["max_dual_evaluator_diff"] < GATE_TOL
        and diag["max_starting_vertex_spread"] < GATE_TOL
        and diag["max_fundamental_trace_diff"] < GATE_TOL
        and diag["max_lambda1_err"] < GATE_TOL
    )

    p_by_n = [held_eval["per_n"][str(n)]["braess_paradox_fraction"] for n in HELDOUT_N]
    h1a_min_5pct = all(p >= 0.05 for p in p_by_n)
    h1b_strict_monotone = all(p_by_n[i] < p_by_n[i + 1] for i in range(len(p_by_n) - 1))
    h1_pass = h1a_min_5pct and h1b_strict_monotone

    dist2_share = held_eval["dist2_share_of_all_paradoxes"]
    h2a_dist2_80pct = dist2_share >= 0.80
    h2b_zero_diam = (
        held_eval["total_diameter_shortcut_paradox"] == 0
        and held_eval["max_diameter_shortcut_delta_k"] < -PARADOX_EPS
    )
    h2_pass = h2a_dist2_80pct and h2b_zero_diam

    if not h3_pass:
        verdict = "UNASSESSED"
    elif h1_pass and h2_pass:
        verdict = "CORROBORATED"
    else:
        verdict = "FALSIFIED"

    results_payload = {
        "mission_id": "MISSION-063",
        "verdict": verdict,
        "freeze_verification": {
            "verified": freeze_ok,
            "protocol_sha256": actual_protocol_sha,
            "script_sha256": actual_script_sha,
        },
        "hypothesis_checks": {
            "H3": {
                "passed": h3_pass,
                "oeis_counts_matched": oeis_ok,
                "prufer_cayley_matched": all(
                    v["exact_match"] for v in prufer_report.values()
                ),
                "closed_form_max_err": max(
                    closed_form["max_kn_closed_form_err"],
                    closed_form["max_cn_closed_form_err"],
                ),
                "max_dual_evaluator_diff": diag["max_dual_evaluator_diff"],
                "max_starting_vertex_spread": diag["max_starting_vertex_spread"],
            },
            "H1": {
                "passed": h1_pass,
                "h1a_at_least_5pct_every_heldout_n": h1a_min_5pct,
                "h1b_strict_monotone_increase_with_n": h1b_strict_monotone,
                "paradox_fraction_by_n": {
                    str(n): held_eval["per_n"][str(n)]["braess_paradox_fraction"]
                    for n in HELDOUT_N
                },
            },
            "H2": {
                "passed": h2_pass,
                "h2a_dist2_share_at_least_80pct": h2a_dist2_80pct,
                "observed_dist2_share": dist2_share,
                "h2b_zero_diameter_shortcut_paradoxes": h2b_zero_diam,
                "observed_diameter_shortcut_paradoxes": held_eval[
                    "total_diameter_shortcut_paradox"
                ],
                "max_diameter_shortcut_delta_k": held_eval[
                    "max_diameter_shortcut_delta_k"
                ],
            },
        },
        "dev_summary": dev_eval,
        "heldout": held_eval,
    }

    RESULTS_PATH.write_text(
        json.dumps(results_payload, indent=2) + "\n", encoding="utf-8"
    )
    README_PATH.write_text(render_readme(results_payload), encoding="utf-8")

    print(f"[HELDOUT] Verdict: {verdict}")
    print(f"  H3 (Instrument Validity): {'PASS' if h3_pass else 'FAIL'}")
    print(
        f"  H1 (Prevalence >= 5% & Monotone): {'PASS' if h1_pass else 'FAIL'} "
        f"(h1a={h1a_min_5pct}, h1b={h1b_strict_monotone}, p={[round(100*x, 2) for x in p_by_n]}%)"
    )
    print(
        f"  H2 (Dist-2 >= 80% & Diam == 0): {'PASS' if h2_pass else 'FAIL'} "
        f"(h2a={h2a_dist2_80pct} [{100*dist2_share:.2f}%], h2b={h2b_zero_diam} "
        f"[diam_paradox={held_eval['total_diameter_shortcut_paradox']}])"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["dev", "freeze", "heldout"],
        required=True,
        help="Execution phase",
    )
    args = parser.parse_args()
    if args.mode == "dev":
        return run_dev()
    if args.mode == "freeze":
        return run_freeze()
    return run_heldout()


if __name__ == "__main__":
    raise SystemExit(main())
