# MISSION-063: Braess's Paradox for Random Walks in Trees

**Verdict:** `FALSIFIED`  
**Held-Out Domain:** All $422$ non-isomorphic trees across $n \in \{7, 8, 9, 10, 11\}$ ($16,355$ missing-edge additions)  
**Instrument Gate (H3):** `PASS` (max dual-evaluator discrepancy $2.77e-13 < 10^{-11}$)  

---

## 1. Executive Summary

- **H1 (Prevalence $\ge 5\%$ per $n$ & Strict Monotonic Growth $p(7) < p(8) < p(9) < p(10) < p(11)$):** `FAIL`
  - Held-out Braess-paradox prevalence by vertex count: $n=7$: **20.00%** (33/165), $n=8$: **18.22%** (88/483), $n=9$: **15.50%** (204/1316), $n=10$: **14.49%** (553/3816), $n=11$: **13.83%** (1462/10575).
  - Overall held-out prevalence: **2340 / 16355 (14.31%)** missing-edge additions increase Kemeny's constant ($\Delta K > 10^{-9}$).
- **H2 (Distance-2 Concentration $\ge 80\%$ & Diameter-Shortcut Immunity):** `FAIL`
  - Share of all held-out Braess paradoxes occurring at tree distance $d_T(u,v) = 2$ (forming a $3$-cycle): **2051 / 2340 (87.65%)** (threshold $\ge 80\%$: `PASS`).
  - Diameter-endpoint shortcuts ($d_T(u,v) = \operatorname{diam}(T)$) exhibiting Braess's paradox: **145 / 1746** ($\max \Delta K = 0.439394 < 0$; threshold $0$: `FAIL`).

---

## 2. Held-Out Results by Vertex Count ($n = 7..11$) and Development ($n = 4..6$)

| $n$ | Split | Free Trees (OEIS A000055) | Missing-Edge Pairs | Braess Paradox ($\Delta K > 0$) | Paradox % $p(n)$ | Dist-2 Share of Paradoxes | Orbit-Unique Paradox % | Unweighted $\bar{H}$ Paradox % | Mean $\Delta K$ | Max $\Delta K$ | Min $\Delta K$ |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | Dev | 2 | 6 | 3 | 50.00% | 100.00% | 33.33% | 0.00% | -0.2986 | +0.0417 | -0.6667 |
| 5 | Dev | 3 | 18 | 7 | 38.89% | 100.00% | 22.22% | 5.56% | -0.5185 | +0.3667 | -1.6000 |
| 6 | Dev | 6 | 60 | 16 | 26.67% | 100.00% | 14.81% | 26.67% | -0.7894 | +0.7833 | -3.0833 |
| **7** | **Held-Out** | **11** | **165** | **33** | **20.00%** | **100.00%** | 11.39% | 20.61% | -1.1547 | +1.2619 | -4.9381 |
| **8** | **Held-Out** | **23** | **483** | **88** | **18.22%** | **95.45%** | 13.30% | 18.63% | -1.5461 | +1.7827 | -7.1250 |
| **9** | **Held-Out** | **47** | **1316** | **204** | **15.50%** | **93.14%** | 12.37% | 15.43% | -2.0017 | +2.3333 | -9.6429 |
| **10** | **Held-Out** | **106** | **3816** | **553** | **14.49%** | **89.33%** | 12.03% | 14.39% | -2.5059 | +2.9056 | -12.4917 |
| **11** | **Held-Out** | **235** | **10575** | **1462** | **13.83%** | **85.50%** | 11.98% | 13.56% | -3.0542 | +3.4939 | -15.6717 |

---

## 3. Breakdown by Shortcut Distance $d_T(u, v)$ (Held-Out $n = 7..11$)

| Tree Distance $d_T(u, v)$ | Cycle Length $L = d+1$ | Held-Out Pairs | Braess Paradox Count | Paradox % at Distance $d$ | Share of All Paradoxes | Mean $\Delta K$ | Max $\Delta K$ | Min $\Delta K$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 5796 | 2051 | 35.39% | 87.65% | -0.7327 | +3.4939 | -5.6212 |
| 3 | 4 | 4717 | 275 | 5.83% | 11.75% | -2.4381 | +1.7091 | -8.6909 |
| 4 | 5 | 3243 | 14 | 0.43% | 0.60% | -3.9368 | +0.4818 | -11.4818 |
| 5 | 6 | 1649 | 0 | 0.00% | 0.00% | -5.6767 | -1.5152 | -13.3333 |
| 6 | 7 | 681 | 0 | 0.00% | 0.00% | -7.3337 | -3.2091 | -14.9494 |
| 7 | 8 | 209 | 0 | 0.00% | 0.00% | -9.1140 | -5.5273 | -15.3545 |
| 8 | 9 | 51 | 0 | 0.00% | 0.00% | -10.7035 | -7.7970 | -15.6717 |
| 9 | 10 | 8 | 0 | 0.00% | 0.00% | -12.3606 | -10.5455 | -14.5455 |
| 10 | 11 | 1 | 0 | 0.00% | 0.00% | -13.5000 | -13.5000 | -13.5000 |

---

## 4. Mechanism & Extreme Cases

### Strongest Held-Out Braess Paradox ($\max \Delta K$)
- **Tree:** $n = 11$, tree index `127`, diameter $9$, edges `[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (5, 9), (8, 10)]`
- **Added edge:** `[7, 9]` at tree distance $d_T(u,v) = 2$
- **Kemeny's constant:** $K(T) = 31.900000 \to K(T+e) = 35.393939$ ($\Delta K = +3.493939$, **+10.95%**)

### Strongest Held-Out Shortcut Speedup ($\min \Delta K$)
- **Tree:** $n = 11$, tree index `0`, diameter $10$, edges `[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)]`
- **Added edge:** `[7, 8]` at tree distance $d_T(u,v) = 8$
- **Kemeny's constant:** $K(T) = 33.500000 \to K(T+e) = 17.828283$ ($\Delta K = -15.671717$, **-46.78%**)

### Physical Interpretation: Why Stationary Weight Shift vs. Raw Transition Trapping Both Matter
- When we replace Kemeny's stationary-weighted expectation $K(G) = \sum_{j \neq i} \pi_j m_{i,j}$ with the **unweighted** average hitting time $\bar{H}(G) = \frac{1}{n(n-1)} \sum_{i \neq j} m_{i,j}$, the held-out paradox rate is **14.12%** (2310 / 16355 pairs) compared to **14.31%** for Kemeny's constant.
- Adding a short edge (especially at distance $d_T(u,v)=2$, creating a triangle, or $d_T(u,v)=3$, creating a $4$-cycle) inside a high-degree hub/cluster simultaneously:
  1. Diverts random-walk transition probability ($1/(d_u+1)$) into recirculating within the local cycle, delaying escape into long pendant branches (which raises unweighted $\bar{H}(G)$ even without $\pi$-weights).
  2. Increases total edge count $m = n-1 \to n$ in $K(G) = m \sum_{x,y} \pi_x \pi_y R_{\text{eff}}(x,y)$ while reducing $R_{\text{eff}}$ only locally.

---

## 5. Instrument Validity Gate (H3) Summary

- **OEIS A000055 Free-Tree Counts ($n=4..11$):** `True` (`2, 3, 6, 11, 23, 47, 106, 235`)
- **Prüfer-Sequence Cayley Bijection ($n=4..6$):** `True` ($4^2=16, 5^3=125, 6^4=1296$)
- **Closed-Form $K(K_n) = (n-1)^2/n$ and $K(C_n) = (n^2-1)/6$:** max error `1.07e-14`
- **Fundamental Matrix vs. Symmetrized Spectral Evaluator ($16,355$ held-out pairs):** max discrepancy `2.77e-13`
- **Starting-Vertex Invariance ($\max_i K_i - \min_i K_i$):** max spread `1.78e-14`
- **Freeze Hash Verification:** `True`
