#!/usr/bin/env python3
"""Structurally independent checks for MISSION-073 (pure Python; no code shared with burnside.c).

Modes:
  orbits N                 brute-force orbit counting by canonical form (feasible for N<=3)
  props N COUNT SEED       random masks -> writes hexfile for burnside --props and expected results
  fix N MAXCYC             Python brute Fix-counts for group elements with <= MAXCYC edge cycles
Edge indexing is re-derived here from the written spec: for d in 0..n-1, for v ascending with bit d
clear, edge = (v, v|1<<d).
"""
import itertools, json, random, sys

def setup(n):
    NV = 1 << n
    edges = [(v, v | (1 << d), d) for d in range(n) for v in range(NV) if not (v >> d) & 1]
    index = {}
    for i, (a, b, d) in enumerate(edges):
        index[(a, b)] = i; index[(b, a)] = i
    return NV, edges, index

def prop(mask, edges):
    """connected? span mask. Union-find, structurally different from the C flood fill."""
    es = [edges[i] for i in range(len(edges)) if (mask >> i) & 1]
    if not es:
        return False, 0
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    span = 0
    for a, b, d in es:
        span |= 1 << d
        parent[find(a)] = find(b)
    return len({find(a) for a, b, d in es}) == 1, span

def group(n):
    NV, edges, index = setup(n)
    for perm in itertools.permutations(range(n)):
        inv = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
        for s in range(1 << n):
            det = (-1) ** inv * (-1) ** bin(s).count("1")
            def vm(v):
                w = 0
                for d in range(n):
                    if (v >> d) & 1:
                        w |= 1 << perm[d]
                return w ^ s
            emap = [index[(vm(a), vm(b))] for a, b, d in edges]
            yield perm, s, det, emap

def apply(emap, mask):
    out = 0
    for i, j in enumerate(emap):
        if (mask >> i) & 1:
            out |= 1 << j
    return out

def cycles(emap):
    seen, cyc = set(), []
    for e in range(len(emap)):
        if e not in seen:
            m, x = 0, e
            while x not in seen:
                seen.add(x); m |= 1 << x; x = emap[x]
            cyc.append(m)
    return cyc

def orbits(n):
    NV, edges, index = setup(n)
    full = (1 << len(edges)) - 1
    G = list(group(n))
    res = {}
    for name, sel in (("rotations", [g for g in G if g[2] == 1]), ("all_symmetries", G)):
        seen, by_k = set(), {}
        for m in range(1, full):
            c, sp = prop(m, edges)
            if not c or m in seen:
                continue
            orb = {apply(g[3], m) for g in sel}
            seen |= orb
            k = bin(sp).count("1")
            by_k[k] = by_k.get(k, 0) + 1
        res[name] = dict(sorted(by_k.items()))
    return res

if __name__ == "__main__":
    mode, n = sys.argv[1], int(sys.argv[2])
    if mode == "orbits":
        print(json.dumps(orbits(n)))
    elif mode == "props":
        NV, edges, index = setup(n)
        cnt, seed = int(sys.argv[3]), int(sys.argv[4])
        rnd = random.Random(seed)
        full = (1 << len(edges)) - 1
        masks = [rnd.randrange(0, full + 1) for _ in range(cnt)]
        masks += [0, full] + [1 << i for i in range(len(edges))]
        with open("props_in.txt", "w") as f:
            for m in masks:
                f.write("%08x\n" % m)
        with open("props_expected.txt", "w") as f:
            for m in masks:
                c, sp = prop(m, edges)
                f.write("%08x %d %d\n" % (m, int(c), sp if c else 0))
    elif mode == "fix":
        maxcyc = int(sys.argv[3])
        NV, edges, index = setup(n)
        full = (1 << len(edges)) - 1
        for perm, s, det, emap in group(n):
            cyc = cycles(emap)
            if len(cyc) > maxcyc:
                continue
            cnt = [0] * 16
            for j in range(1 << len(cyc)):
                m = 0
                for i, c in enumerate(cyc):
                    if (j >> i) & 1:
                        m |= c
                if m == full:
                    continue
                c_, sp = prop(m, edges)
                if c_:
                    cnt[sp] += 1
            print(json.dumps({"perm": list(perm), "flip": s, "det": det, "cycles": len(cyc), "cnt": cnt}))
