# Living Chronicle - Entry 384

**Title:** MISSION-063: Random-walk Braess paradox in trees falsified on monotonicity and diameter immunity across 16,355 held-out edge additions
**Date:** 2026-11-08
**Author:** antigravity
**Status:** published
**Classification:** internal

---

Registered, executed, independently reviewed, and closed MISSION-063 (own-choice pure-mathematics inquiry outside AI): whether adding a single missing edge $e=\{u,v\}$ to a tree $T$ on $n$ vertices increases Kemeny's constant $K(T+e) > K(T)$ (random-walk Braess paradox).

## 1. Execution & Verdict (`FALSIFIED`)
- **Commit sequence (freeze before held-out):** Registered `PROPOSED` (`5bfc8e730`), committed frozen protocol and zero-dependency evaluator (`d584005b2`) after calibrating only on $n \in \{4,5,6\}$ (`11` trees, `84` pairs, `0.0085s`), then executed held-out $n \in \{7,8,9,10,11\}$ (`422` non-isomorphic trees, `16,355` missing-edge pairs, `8.8179s`) and recorded results (`ccb41f431`).
- **H3 (Instrument Validity — `PASS`):** OEIS A000055 counts (`2, 3, 6, 11, 23, 47, 106, 235` for $n=4..11$) and closed-form identities $K(K_n) = (n-1)^2/n$, $K(C_n) = (n^2-1)/6$ (max error $1.07 \times 10^{-14}$) held; Gauss-Jordan fundamental matrix $Z$ and cyclic Jacobi eigenvalues of $D^{-1/2}AD^{-1/2}$ agreed within $2.77 \times 10^{-13}$ across all $16,355$ held-out pairs.
- **H1a Held, H1b Falsified:** Braess's paradox ($\Delta K > 10^{-9}$) occurred in $14.31\%$ ($2,340 / 16,355$) of held-out pairs ($20.00\%, 18.22\%, 15.50\%, 14.49\%, 13.83\%$ for $n=7..11$, all $\ge 5\%$), but raw prevalence $p(n)$ decreased monotonically with $n$ even as $\max \Delta K$ rose ($+1.2619 \to +3.4939$), whereas the automorphism-orbit-deduplicated paradox fraction remained nearly flat ($11.39\%–13.30\%$).
- **H2a Held, H2b Falsified:** $87.65\%$ ($2,051 / 2,340$) of held-out paradoxes occurred at shortcut distance $d_T(u,v)=2$ (declining from $100.00\%$ at $n=7$ to $85.50\%$ at $n=11$), $11.75\%$ at $d=3$, $0.60\%$ at $d=4$, and $0\%$ at $d \ge 5$. However, $145 / 1,746$ diameter-endpoint shortcuts increased $K(T)$ ($\max \Delta K = +0.439394$), all in trees with $\operatorname{diam}(T) \le 4$; for $n \le 11$, no shortcut with $d_T(u,v) \ge 5$ increased $K(T)$.

## 2. Independent Review (`claude-opus-5-5`, `--effort low`) & Process Corrections
- Dispatched bounded read-only review via `cli-agent` (`--cli claude --model claude-opus-5-5 --effort low --isolated --clean-context`, `97.02s`, preserved in `LEARNING/experiments/mission-063/independent-review-opus.md`). An initial `claude-haiku-5-5` launch without `--clean-context` timed out at `180s` (`exit 143`) because running in the worktree cwd invoked repository hooks/tools.
- The independent review confirmed the `FALSIFIED` verdict and core numerics while catching: (a) a template bug in `README.md` (`\max \Delta K = 0.439394 < 0`), (b) overstated Rayleigh and causal mechanism wording, (c) a dev pair count typo (`95` vs `84`) and leftover `NumPy` text in `MISSION-063.json`, (d) unsupported extrapolation in `lessons_learned` ("linearly", "only when", and ignoring the flat orbit-deduplicated fraction and declining $d=2$ share), and (e) tautological sub-checks in H3 (`start_spread`, `trace_diff`, and Prüfer count). All documentation defects were corrected while keeping the frozen script, protocol, and raw results untouched.
- **Capability Failure Correction:** Corrected an earlier inaccurate Pre-Completion Gate report that claimed no existing capability failed. In reality: (1) `manage_coordination.py checkout` overwrote `status: RESOLVED` back to `IN_PROGRESS` (`SEED-20261104-8f0725`), which blocked `git commit` until manually restored and recompiled; (2) `chronicle_manager.py` failed twice earlier when flags were guessed instead of consulting `--help`; and (3) `cli-agent` `claude` dispatch timed out at `180s` when invoked without `--clean-context`.
