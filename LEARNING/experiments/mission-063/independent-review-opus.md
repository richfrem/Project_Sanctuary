# MISSION-063 Independent Adversarial Review (`claude-opus-5-5`, `--effort low`)

**Reviewer Backend:** `claude` (`/Users/richardfremmerlid/.local/bin/claude` v2.1.319, model `claude-opus-5-5`, `--effort low`, `--isolated`, `--clean-context`)  
**Dispatch Unit:** `mission-063-review-opus` (`duration_s: 97.02s`, `exit_code: 0`)  
**Prior Attempt Note:** Initial attempt `mission-063-review` (`claude-haiku-5-5` without `--clean-context`, running inside the worktree cwd) timed out at `180.0s` (`exit 143`, process group `86179` cleanly reaped via `SIGTERM->SIGKILL`). Re-running with `--clean-context` (neutral temp cwd, `--setting-sources project --disable-slash-commands --tools ""`) and `--model claude-opus-5-5 --effort low` completed cleanly in `97.02s`.

---

**Bottom line:** The headline verdict, **FALSIFIED on H1b and H2b with H3 PASS**, is consistent with the pre-registered rules. The core numerics appear correct. However, there are three groups of problems:

- **Templating bugs:** the README contains self-contradictory text.
- **Weak validity checks:** several H3 sub-checks are tautological, so they verify less than they claim.
- **Unsupported claims:** several causal statements and generalizations in README, MISSION-063.json and lessons_learned go beyond the data.

No line numbers are given in the bundle, so I cite functions, sections and fields.

---

## 1. Mathematical correctness

### 1.1 Fundamental-matrix evaluator (`evaluate_kemeny_dual`, Method 1)

**Correct:**
- `a_sys[i][j] = δ_ij − A_ij/d_i + π_j` is correctly `I − P + 1πᵀ`, since row i of `1πᵀ` is πᵀ.
- `m_ij = (Z_jj − Z_ij)/π_j` is the standard Kemeny–Snell identity.
- `K = Tr(Z) − 1` matches the protocol definition `K = Σ_{j≠i} π_j m_ij` (no "+1" convention).

**Defect M1: the starting-vertex spread and trace checks are near-tautological.**
- K_i = Σ_{j≠i}(Z_jj − Z_ij) = Tr Z − Σ_j Z_ij.
- Since Z1 = 1 holds algebraically, the spread check only tests the row sums of Z, i.e. Gauss-Jordan round-off.
- `k_hit = mean(k_by_start)` is algebraically equal to `k_trace`, so `trace_diff` is also only a round-off check.

H3.2 presents these as three independent validations. In reality only `dual_diff` (Z versus spectrum) is substantive. The protocol and README should say so.

**Defect M2: the evaluators share all their inputs.**
- Both methods consume the same `adj_matrix` and `deg`.
- A wrong edge set (for example, a bad `build_adj_matrix` call or an orbit mix-up) would pass the dual check unnoticed.
- Only the closed-form K_n and C_n tests guard graph construction, and those never exercise the tree or unicyclic code path.

**Fix:** add a third check on trees, such as the resistance identity K(T) = m Σ_{x,y} π_x π_y d_T(x,y). This is exact for trees, because R_eff equals path distance.

### 1.2 Spectral evaluator (Method 2)

**Correct:**
- S = D^{-1/2} A D^{-1/2} is similar to P.
- Excluding the largest eigenvalue (λ=1, simple because the graph is connected) is right.
- For bipartite trees the eigenvalue −1 contributes 1/2, which is correct.

**Defect M3: no convergence check.** `jacobi_eigenvalues_symmetric` runs at most 25 sweeps and never reports whether `off_norm < 1e-30` was reached. Non-convergence would currently surface only indirectly, through `dual_diff`. It should be recorded explicitly, for example as `max_sweeps_used` and `final_off_norm`.

**Defect M4: `lambda1_err` checks only the top eigenvalue.** It should also confirm the second-largest eigenvalue is below 1 − δ, i.e. no near-degenerate unit eigenvalue.

### 1.3 Closed forms

- K(K_n) = (n−1)²/n and K(C_n) = (n²−1)/6 are correct under this convention. As a sanity check, both formulas give 4/3 at n = 3.
- **Minor:** `verify_closed_form_identities` merges `dual_diff` and `start_spread` into a single max. It also omits `trace_diff` and `lambda1_err` for the reference graphs, which is inconsistent with H3.2's "every evaluated graph".

### 1.4 Tolerance semantics

`GATE_TOL = 1e-11` is absolute while K reaches about 35, which is fine. However:
- **Defect M5:** the number of pairs with |ΔK| < 1e-9 is never reported. Ties near zero would be classified as "no paradox" silently, yet they would also make H2b fail through the strict `< -1e-9` clause. This count should be reported.

---

## 2. Enumeration coverage

### 2.1 Free-tree generation (`generate_free_trees`)

- **Completeness holds.** Every tree on n ≥ 3 vertices has a leaf, and deleting it gives a tree isomorphic to some stored representative on n−1 vertices. Attaching a leaf at every vertex of every representative therefore reaches every isomorphism class.
- **Dedup is sound.** The AHU certificate rooted at the center (or at the bicenter, ordered as a min/max pair) is a complete invariant for free trees. The `[…]` and `(…)` prefixes prevent unicentral/bicentral collisions, and the balanced-paren encoding is uniquely parseable.
- The OEIS match for n = 4..11 is therefore credible.

**Defect E1: the Prüfer "Cayley bijection" check is tautological.**
- `labeled_count == n^(n−2)` holds by construction of `product(range(n), repeat=n−2)`.
- It only tests that the decoder terminates, and that the map is surjective onto the classes.
- **Fix:** test the orbit–stabilizer identity Σ_classes n!/|Aut(T)| = n^{n−2}, or check that per-class labeled counts match n!/|Aut|.
- The README label "Cayley Bijection" overstates what the check does.

**Prüfer decoder correctness:** decoding is correct. The current x always has deg ≥ 2 when its leaf is chosen, and used leaves drop to deg 0.

### 2.2 `canonical_edge_orbit_key`

- The key is correct as an orbit invariant for unordered pairs {u,v}.
- Marking both endpoints with the same `*` and rooting at the center (which every automorphism preserves) yields a canonical form of the 2-vertex-colored tree.
- Equal keys hold exactly when some automorphism maps {u,v} to {u′,v′}. The bicentral swap is covered by `min`/`max`.
- **Minor:** the key is recomputed for each pair at O(n log n) string cost. That is fine at this scale.

**Defect E2: pre-registered secondary outputs are missing.**
- MISSION-063.json `design_constraints` promise "Prüfer-weighted labeled-tree averages" and "degree-sum correlations".
- `experiment` and `falsification_criteria` mention breakdowns "by tree diameter, maximum degree".
- None of these is implemented. They are descriptive only, so they do not change the verdict, but the omission must be stated.

### 2.3 Count inconsistency

**Defect E3:** MISSION-063.json `experiment` says the dev set has "11 trees, 95 missing edges". The protocol and the computed results both give 6 + 18 + 60 = **84**. Correct the JSON.

The held-out total of 16,355 checks out: 11·15 + 23·21 + 47·28 + 106·36 + 235·45.

---

## 3. Verdict consistency

| Check | Rule | Observed | Status |
|---|---|---|---|
| H3 | all gates | PASS (see M1/E1 caveats) | ✔ |
| H1a | p(n) ≥ 5% for all n | 20.00 / 18.22 / 15.50 / 14.49 / 13.83% | holds |
| H1b | strictly increasing | strictly *decreasing* | **fails** |
| H2a | dist-2 share ≥ 80% | 87.65% | holds |
| H2b | 0 diameter paradoxes and max ΔK < −1e-9 | 145/1746, max +0.439 | **fails** |

`FALSIFIED` follows from the protocol §4 rules. The code logic in `run_heldout` matches the protocol.

**Process caveats:**
- **V1: freeze ordering is not shown.** Runtime hash verification proves the files did not change *after* freeze.json was written. It does not prove that freeze.json was committed before the held-out run. `frozen_at` is a hard-coded string in `run_freeze`, not a real timestamp. The bundle needs git commit SHAs and timestamps for the freeze commit versus the results commit; without them the "frozen before held-out" claim is unverified.
- **V2: exit code ignores the verdict.** `run_heldout` returns 0 even when the verdict is UNASSESSED, so CI cannot detect an instrument failure. It should return non-zero when `h3_pass` is false.

---

## 4. Overclaims and unsupported causal statements

### README.md

**R1: template bug causes a self-contradiction.**
- The H2b bullet renders as `$\max \Delta K = 0.439394 < 0$`.
- `render_readme` hard-codes "< 0" regardless of the value, so the README asserts something false.
- **Fix:** render the comparison conditionally.

**R2: "Physical Interpretation" section is untested causal narrative.**
- The "recirculating within the local cycle, delaying escape into long pendant branches" mechanism is never measured. No escape-time, per-vertex m_ij, or branch analysis was done.
- The heading "Why … Both Matter" is not supported by the data shown:
  - The aggregate unweighted rate (14.12%) is nearly identical to the Kemeny rate (14.31%). That suggests π-reweighting contributes little in aggregate at n = 7..11.
  - However, at n = 4 the rates are 0% versus 50%.
  - The pair-level overlap (paradox under both, under K only, under H̄ only) is never computed, so no attribution claim can be made.
- **Fix:** compute the 2×2 overlap table, or relabel the section as conjecture.

**R3: "reducing R_eff only locally" is incorrect.**
- Adding {u,v} to a tree lowers R_eff(x,y) for *every* pair whose tree path uses at least one edge of the new cycle. That can be a large, non-local set.
- Pairs whose path avoids the cycle are unchanged.

**R4: mislabeled column.** The column "Braess Paradox (ΔK > 0)" should read ΔK > 10⁻⁹.

### MISSION-063.json

**J1: overstated Rayleigh claims.**
- `interest_statement` says adding an edge "decreases every pairwise effective resistance".
- `strategic_intent` says it "strictly lowers effective resistance everywhere".
- Both are false, for the reason in R3. Rayleigh's law gives non-increase only.

**J2: dependency contradiction.**
- `target_platform` and the script docstring say "zero-dependency Python standard library".
- `score_basis` and `routing_guidance` say "NumPy".
- Pick one; the code uses only the standard library.

**J3: unmeasured runtime.** The "~9 seconds" figure in `readiness` is not reported in README. `elapsed_seconds` exists in results.json and should be cited instead.

**J4: unaudited literature citations.** Kirkland & Neumann 2013 and "Breen, Kirkland & Ciardo" should be verified or marked as unverified.

**J5: evidence characterization not computed.**
- The evidence entry calls the 145 diameter paradoxes "low-diameter star/spider trees".
- No per-diameter or per-shape breakdown was computed.
- "diam ≤ 4" and "0 for diam ≥ 5" *are* derivable from the distance table, since max ΔK < 0 for every d ≥ 5.
- The "star/spider" description is not derivable and should be removed or backed by a table.

### lessons_learned

**L1: "strictly local … (100% at d ≤ 4)" is presented as structural.**
- It is an empirical fact for n ≤ 11 only.
- Max ΔK at d = 3 (+1.71) and d = 4 (+0.48) may grow with n, so d = 5 paradoxes at larger n are not ruled out.
- The per-n dist-2 share is itself falling: 100 → 95.5 → 93.1 → 89.3 → 85.5%. That trend suggests H2a could fail at larger n, which contradicts the "concentrated" framing. Add this caveat.

**L2: "share … falls with n even as maximum slowdown grows linearly with n".**
- "Linearly" is unsupported. There are five points, with increments of .52, .55, .57 and .59, which are not constant, and no fit was done.
- "So" asserts a causal link (locality → falling fraction) that was not tested. A plausible test: compare the count of d = 2 pairs, Σ_v C(d_v, 2), against C(n−1, 2).
- Representation sensitivity is ignored. The orbit-deduplicated fraction is roughly flat and non-monotone (11.39, 13.30, 12.37, 12.03, 11.98%). The "decline with n" therefore partly reflects counting each labeled missing edge once per isomorphism class. The verdict stands, since the raw fraction was pre-registered, but the lesson should note this.

**L3: "immunity holds *only when* d_T(u,v) ≥ 5".**
- Logically wrong: most d ≤ 4 diameter shortcuts are *not* paradoxical, so d ≥ 5 is not necessary for immunity.
- It also extrapolates beyond n ≤ 11.
- Rephrase as: "for n ≤ 11, no shortcut with d ≥ 5 increased K".
