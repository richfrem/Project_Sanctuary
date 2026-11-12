CRITICAL: In `tilings.c`, the `build()` function incorrectly maps tile orientations to connections. For orientation A (`t[i*L+j]==0`), it unions `N-W` and `S-E`, but on a torus with periodic boundary conditions, the S and E neighbors must wrap correctly. However, the indexing of S and E is correct (`S=((i+1)%L)*L+j`, `E=n+i*L+(j+1)%L`), and the union logic matches the quarter-arc definition. **No bug here.**

CRITICAL: In `gate.py`, the `trace()` function’s loop-walking logic contains a **critical bug** in the transition rule. Specifically:

```python
nx = a if (a!=prev or a==b) else b
if prev is not None and a==prev and b==prev: nx=a
```

The second condition is redundant and incorrect: when `a == b`, the node has only one neighbor (a self-loop in adjacency list), but the assertion `assert all(len(v)==2 for v in adj.values())` guarantees every node has exactly two neighbors — so `a == b` should never occur. More critically, the first line fails to handle the case where `prev == a` but `a != b`; in that case, `nx` should be `b`, but the condition `(a!=prev or a==b)` evaluates to `(False or False) = False`, so it picks `b` — *correctly*. Wait — actually, let’s re-evaluate:

Standard traversal: at node `cur`, with previous node `prev`, the two neighbors are `a` and `b`. The next node is the one ≠ `prev`. So:

```python
nx = b if a == prev else a
```

The current code:
```python
nx = a if (a!=prev or a==b) else b
```

- If `a != prev`: pick `a` — correct.
- If `a == prev`:
  - If `a == b`: pick `a` — but `a == b` implies both neighbors equal `prev`, violating degree-2 (impossible per assertion).
  - Else (`a == prev`, `b != a`): pick `b` — correct.

So the logic is *mathematically correct* under the degree-2 guarantee. However, the second `if` is dead code (since `a==b` is impossible), but harmless.

**No bug in `trace()`**.

CRITICAL: In `analyze.py`, the `alpha_tail()` function computes tail exponent from `ge[b]`, where `ge[b]` counts loops with size ≥ `128 * 2^b`. But `ge[0]` corresponds to size ≥ 128, `ge[1]` ≥ 256, ..., `ge[5]` ≥ 4096. For `L=1024`, `L^2 = 1048576`, so `2*L^2 = 2097152` nodes. Loop sizes range up to ~2M. However, `ge[b]` is normalized by `1024*1024 = L^2`, but should be normalized by `L^2` (number of tiles), not `L^2` in area units — but loop *count* density should be per tile, so `N(≥s)/L^2` is correct. However, the binning uses `s = 128 * 2^b`, and `log(s)` is used as `x`. That’s fine.

But the **critical error** is in the normalization factor: `m = sum(x['ge'][b] for x in r) / len(r) / (1024*1024)`. This computes average count per sample divided by `L^2`, i.e., `⟨N(≥s)⟩ / L^2`. Correct.

However, the tail exponent α is defined by `N(≥s) ~ L^2 * s^{-α}`. So `log(N(≥s)/L^2) = -α log s + const`, hence `α = -