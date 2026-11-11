#!/usr/bin/env python3
"""Exploratory (unfrozen) side artifact for MISSION-073: ring the cube.

Checks that (1) the 24 cube rotations act faithfully on the 4 body diagonals as all of S4,
(2) the 3 adjacent-place swaps of 4-bell ringing correspond to rotations that are half-turns
about edge-midpoint axes, and (3) a 4-bell plain-changes extent (24 rows, each row one adjacent
swap from the previous, ending adjacent to rounds) walks every cube rotation exactly once.
Also records that |rotations of the 4-cube| = 192 != 120 = |S5|.
"""
import itertools, json
from independent import group, setup

NV, edges, index = setup(3)
rots = [(p, s, emap, perm_v) for (p, s, det, emap) in group(3) if det == 1
        for perm_v in [None]]
diag = [(v, v ^ 7) for v in range(4)]          # the 4 body diagonals {v, v^7}
def vmap(p, s, v):
    w = 0
    for d in range(3):
        if (v >> d) & 1: w |= 1 << p[d]
    return w ^ s
def diag_perm(p, s):
    out = []
    for a, b in diag:
        img = {vmap(p, s, a), vmap(p, s, b)}
        out.append(next(i for i, (x, y) in enumerate(diag) if {x, y} == img))
    return tuple(out)
table = {}
for (p, s, emap, _) in rots:
    table[diag_perm(p, s)] = (p, s, emap)
res = {"rotations": len(rots), "distinct_diagonal_perms": len(table), "is_all_S4": len(table) == 24}
def fixed_edges(emap): return sum(1 for i, j in enumerate(emap) if i == j)
swaps = {}
for i in range(3):
    t = list(range(4)); t[i], t[i + 1] = t[i + 1], t[i]
    p, s, emap = table[tuple(t)]
    order2 = all(emap[emap[e]] == e for e in range(12))
    swaps[f"({i} {i+1})"] = {"order2": order2, "edges_fixed_as_sets": fixed_edges(emap), "vertices_fixed": sum(1 for v in range(8) if vmap(p, s, v) == v)}
res["adjacent_place_swaps_as_rotations"] = swaps
# plain changes on 4 bells via Steinhaus-Johnson-Trotter-style recursion (ringing 'plain changes')
def plain_changes(n):
    if n == 1: return [(0,)]
    prev = plain_changes(n - 1); rows = []
    for k, r in enumerate(prev):
        pos = range(n - 1, -1, -1) if k % 2 == 0 else range(n)
        for q in pos:
            rows.append(r[:q] + (n - 1,) + r[q:])
    return rows
rows = plain_changes(4)
def adjacent(a, b):
    d = [i for i in range(4) if a[i] != b[i]]
    return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]
res["extent_rows"] = len(rows); res["extent_distinct"] = len(set(rows)) == 24
res["every_step_adjacent_swap"] = all(adjacent(rows[i], rows[i + 1]) for i in range(23))
res["last_row_to_first_adjacent"] = adjacent(rows[-1], rows[0])
res["first_row"] = rows[0]
# walk: orientation_t = orientation_{t-1} composed with the swap's rotation; check all 24 distinct
def compose_diag(a, b): return tuple(a[b[i]] for i in range(4))   # a after b
cur = tuple(range(4)); seen = {cur}; tour = [cur]
for i in range(23):
    d = [j for j in range(4) if rows[i][j] != rows[i + 1][j]][0]
    t = list(range(4)); t[d], t[d + 1] = t[d + 1], t[d]
    cur = tuple(cur[t[j]] for j in range(4))
    seen.add(cur); tour.append(cur)
res["tour_visits_distinct_orientations"] = len(seen)
res["tour_covers_all_24"] = len(seen) == 24 and seen == set(table)
# 4-cube rotation group order
res["rotations_of_4cube"] = sum(1 for g in group(4) if g[2] == 1)
res["order_S5"] = 120
print(json.dumps(res, indent=1))
