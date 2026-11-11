- **Gray-code enumeration defect (HIGH)**: Line `m ^= p->cyc[__builtin_ctzll(g)];` assumes `g` starts at 1 for first toggle, but loop starts with `g=0`, so first subset is `base` (correct), but second subset toggles cycle `__builtin_ctzll(1)=0`, skipping the subset where only cycle 0 is added *after* base—this is correct. However, `g` increments *before* toggle, so subset for `g=0` is `base`, `g=1` toggles cycle 0, `g=2` toggles cycle 1, `g=3` toggles cycle 0&1—this is standard Gray code *only if* cycles are ordered by bit position in `g`, but `__builtin_ctzll(g)` gives least significant set bit, which yields *reflected Gray code* (correct). **No defect here**.

- **Gray-code prefix split defect (HIGH)**: Line `uint32_t base = 0; for (int i = 0; i < j->tbits; i++) if ((j->pref >> i) & 1) base |= p->cyc[free_c + i];` assumes `p->cyc[free_c .. free_c+tbits-1]` are the *fixed* cycles, but `free_c = p->ncyc - tbits`, so indices `free_c .. p->ncyc-1` are correct. However, `tbits` is `log2(threads)`, and `tbits` bits of `pref` select `tbits` cycles—correct. **No defect**.

- **Edge-permutation construction (MEDIUM)**: `vmap` flips bits via `w ^ s`, but `s` is a bitmask of *sign flips*; for signed permutations, vertex mapping should be `w` with bits flipped where `s` has 1s—correct. `edge_index` searches linearly over `NE` edges; for Q4, `NE=32`, acceptable. **No defect**.

- **Connectivity test (LOW)**: BFS uses `R` initialized to lowest set bit of `touched`; if `touched==0`, `prop` returns 0 early—correct. Loop condition `((R & a) != 0) != ((R & b) != 0)` correctly expands frontier. **No defect**.

- **Determinant/parity (LOW)**: `det = ((inv & 1) ? -1 : 1) * ((__builtin_popcount(s) & 1) ? -1 : 1);` computes sign of permutation × sign of sign-flip vector; for rotations, `det==1` required. **No defect**.

- **uint32 overflow (LOW)**: `FULL` set correctly for NE≤32; `cnt` array uses `uint64_t` in `Job`; final `printf` casts to `unsigned long long`. **No defect**.

- **Counter races (HIGH)**: `jobs[q].cnt` is local to each job; `pthread_join` ensures completion before aggregation—**no race**.

- **Missing rotation filter (CRITICAL)**: `burnside.c` iterates *all* symmetries (`s` from 0 to 2^n−1, all `perm`), but mission requires *rotations only* (det=+1). The code does not filter by `det==1`; it outputs all 48 (Q3) or 384 (Q4) elements. `aggregate.py` must filter, but protocol says “rotations = signed permutation matrices with determinant +1”, and Q3 prediction is 122 for rotations. If `aggregate.py` does not filter, counts will be double (or ~2×) for Q3. **This is the critical defect**: the engine must output only det=+1 elements for the rotation run, or `aggregate.py` must filter—but protocol freeze says “Build: `burnside 4 8 > burnside_n4.jsonl`”, implying engine outputs all symmetries, and `aggregate.py` filters. However, the Q3 check expects 122 for rotations; if `aggregate.py` fails to filter, Q3 count will be 244 (or wrong). **Severity: CRITICAL**.

- **Q3 validation gap (MEDIUM)**: Protocol says “Q3 engine vs independent brute (`dev_results.json`)” but no evidence this validation passed *with rotation-only filtering*. If `dev_results.json` used all symmetries, Q3=122 won’t match. **Severity: MEDIUM**.

**Summary**: Critical defect—engine outputs all symmetries; rotation-only filtering must happen in `aggregate.py`, but protocol doesn’t confirm it does. All other checks pass.