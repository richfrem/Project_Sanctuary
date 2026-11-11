# MISSION-073 confirmatory protocol (to be frozen before any Q4 enumeration)

## Definitions (frozen)
- Q_n: n-cube graph. Edge index: for d in 0..n-1, for vertex v ascending with bit d clear, edge (v, v|1<<d). Q3 has 12 edges, Q4 has 32.
- S_n: edge subsets that are non-empty, not the complete edge set, and connected (the subgraph formed by the edges, on the vertices they touch). Isolated unused vertices are ignored.
- k(S): number of axis directions with at least one edge in S. LeWitt's "non-planar" at n=3 equals k=3. At n=4 three readings are possible (k=4, k>=3, any k>=2); the mission reports all k and designates **k=4 (full-span)** as the primary class.
- Groups: rotations = signed permutation matrices with determinant +1 (24 at n=3, 192 at n=4); all symmetries = full hyperoctahedral group (48, 384). Mirror images are distinct under rotations only (LeWitt's convention).
- Orbit count of a class = (1/|G|) sum over g of the number of class members fixed by g (Burnside). Fixed sets are enumerated exactly as unions of g's edge cycles.

## Instruments
`burnside.c` (engine), `aggregate.py` (Burnside sums, integrality and symmetry checks), `independent.py` (brute canonical-form orbit count; Python property function; Python fixed-set brute force). Development (not held-out) validation, done before this freeze: Q3 engine vs independent brute (`dev_results.json`); Q4 C-vs-Python property agreement on 300,034 masks; Python fixed-set counts for the 308 Q4 group elements with at most 12 cycles (`fix_py.jsonl`).

## Predictions and expectations (recorded honestly)
- **Initial expectation, formed before any Q4 sampling and recorded here after the fact in this file, not before in a committed file**: the k=4 rotation orbit count lies in [1,000,000, 10,000,000] (about 60% confidence), because I assumed roughly 15% of random edge subsets are connected. Dev sampling showed 65.2% are connected, so this expectation is expected to MISS. It is retained as a recorded miss, not as a hypothesis.
- **Frozen forecast (H2)**: from 300,000 uniform random Q4 masks, the full-span connected fraction is 0.65232 +- 0.00087 (identity count about 2.8017e9; /192 = 1.4592e7). The non-identity terms are small. Frozen interval for the k=4 rotation orbit count: [14,400,000, 14,800,000].

## Hypotheses and verdict rules
- **G (gate, n=3)**: rotations k=3 count is 122 and all-symmetries k=3 is 77, equal to independent brute force; Burnside sums divisible by |G|.
- **H1 (instrument consistency, n=4)**: (a) every Burnside sum by (group, k) divisible by |G|; (b) identity counts constant across axis masks of equal size; (c) the engine's per-element counts equal `fix_py.jsonl` for all 308 elements it covers; (d) the 192 + 192 element list has the expected cycle structure (identity has 32 edge cycles).
- **H2**: rotations k=4 orbit count in [14,400,000, 14,800,000].
- **H3**: all-symmetries k=4 count N_all satisfies N_rot/2 < N_all < N_rot.
- **Verdict**: CORROBORATED iff G, H1, H2, H3 all hold. FALSIFIED iff G and H1 hold and H2 or H3 fails (the measured counts are the result). UNASSESSED (verdict null) if G or H1 fails, the run exceeds 3600 s or 1 GiB, or output is incomplete.
- Claim limits: these definitions, Q3 and Q4 only; exhaustive finite computation, not a theorem for all n; no claim about LeWitt's intent in 4D.

## Run plan
`cc -O2 -pthread -o burnside burnside.c`; `burnside 4 8 > burnside_n4.jsonl`; `aggregate.py burnside_n4.jsonl > summary_n4.json`; compare with `fix_py.jsonl`. Any change to code or forecast after this freeze is an amendment recorded separately; the frozen files stay intact.
