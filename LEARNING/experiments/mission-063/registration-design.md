# MISSION-063 Registration Design Note

**Mission:** `LEARNING/missions/MISSION-063.json`
**Registered:** 2026-11-08 by `antigravity` (own-choice topic outside AI and Project Sanctuary).

## 1. Topic Selection Record

Three non-AI mathematical/scientific questions were evaluated for exact local computability, dual-instrument verifiability, and genuine scientific tension:

| Candidate | Domain | Why considered | Disposition |
|:---|:---|:---|:---|
| **Braess's paradox for random walks in trees ($K(T+e) > K(T)$)** | Spectral graph theory & Markov chains | Adding a shortcut edge lowers all pairwise effective resistances (Rayleigh monotonicity) yet inflates edge count $2|E|$ and shifts stationary weights $\pi_v = d_v/(2|E|)$. Can be tested exhaustively across all 422 non-isomorphic trees on $n = 7..11$ ($16,355$ edge additions). | **Selected** |
| **Inversion vs. decay of non-transitive dice cycles under $k$-roll sums** | Combinatorial probability | Tests whether $P(A^{(k)} > B^{(k)})$ inverts direction as $k$ grows before Central Limit Theorem convergence. | Rejected: results depend heavily on arbitrary face-value bounds $(m, M)$. |
| **Majority-vote cellular automata consensus thresholds on circulant graphs** | Discrete dynamical systems | Tests whether symmetric k-regular wiring maximizes basin of attraction for consensus. | Rejected: $2^n$ state space requires sampling for $n > 20$ rather than complete exact enumeration. |

## 2. Mathematical Background & Tension

For a connected, undirected, unweighted graph $G = (V, E)$ on $n$ vertices with $m = |E|$ edges, a simple random walk transitions from vertex $u$ to a uniformly chosen neighbor $v \sim u$ with probability $P_{u,v} = 1/d_u$. Its unique stationary distribution is $\pi_v = \frac{d_v}{2m}$.

**Kemeny's constant** $K(G)$ is the expected number of steps to reach a randomly chosen target vertex $j \sim \pi$ starting from $i$:
$$K(G) = \sum_{j \neq i} \pi_j m_{i,j}$$
Remarkably (Kemeny, 1960), this sum is **independent of the starting vertex $i$** and equals both:
1. **Fundamental-matrix trace**: $K_{\text{hit}}(G) = \operatorname{Tr}\!\left((I - P + \mathbf{1}\pi^T)^{-1}\right) - 1$, where $m_{i,j} = (Z_{j,j} - Z_{i,j})/\pi_j$.
2. **Spectral sum**: $K_{\text{spec}}(G) = \sum_{k=2}^n \frac{1}{1 - \lambda_k(P)}$, where $1 = \lambda_1 > \lambda_2 \ge \dots \ge \lambda_n$ are the eigenvalues of $P = D^{-1}A$ (equivalently of the symmetric matrix $S = D^{-1/2} A D^{-1/2}$).

By Chandra et al. (1996) and Tetali (1991), commute times satisfy $C(u,v) = m_{u,v} + m_{v,u} = 2m R_{\text{eff}}(u,v)$, which gives the electrical-network representation:
$$K(G) = m \sum_{x \in V} \sum_{y \in V} \pi_x \pi_y R_{\text{eff}}(x,y) = \frac{1}{4m} \sum_{x,y \in V} d_x d_y R_{\text{eff}}(x,y)$$

When we add a single missing edge $e = \{u,v\} \notin E$ to a tree $T$ ($m = n-1 \to n$):
- **Rayleigh Monotonicity (Speedup force)**: Every pairwise effective resistance $R_{\text{eff}}(x,y)$ along pathways using the new cycle decreases or stays unchanged; specifically, for vertices on the cycle of length $L = d_T(u,v) + 1$, resistances drop by $\Delta R$.
- **Stationary Shift & Edge Dilution (Slowdown force)**: Total edges increase from $n-1$ to $n$, and degrees $d_u, d_v$ each increase by $1$, shifting stationary weight toward $\{u,v\}$. If $e = \{u,v\}$ connects two vertices at short distance (especially $d_T(u,v) = 2$, forming a triangle), the resistance reduction is tiny ($\Delta R_{\text{eff}}(u,v) = 2 - 2/3 = 4/3$ locally, saving only one hop), while the extra edge traps random-walk transitions inside the triangle and increases the expected time to reach distant leaves!

## 3. Pre-Registered Hypotheses & Falsification Criteria

Let $\Delta K(T, e) = K(T + e) - K(T)$ for a tree $T$ on $n$ vertices and a missing edge $e = \{u,v\} \notin E(T)$.

- **H1 (Prevalence & Strict Growth with $n$)**: For every held-out vertex count $n \in \{7, 8, 9, 10, 11\}$, the fraction $p(n)$ of all missing-edge additions $(T, e)$ across all non-isomorphic free trees on $n$ vertices that exhibit Braess's paradox ($\Delta K(T, e) > 10^{-9}$) is at least $0.05$ ($5\%$), and increases strictly with $n$:
  $$p(7) < p(8) < p(9) < p(10) < p(11)$$
- **H2 (Distance-2 Triangle-Trap Concentration vs. Diameter Shortcuts)**: Across all held-out Braess-paradox occurrences ($\Delta K(T, e) > 10^{-9}$) on $n \in \{7, 8, 9, 10, 11\}$, at least $80\%$ have tree distance $d_T(u,v) = 2$ (forming a $3$-cycle), whereas **zero** diameter-endpoint shortcuts ($d_T(u,v) = \operatorname{diam}(T)$) exhibit Braess's paradox ($\max_{d_T(u,v)=\operatorname{diam}(T)} \Delta K(T, e) < -10^{-9}$).
- **H3 (Instrument Validity Gate)**:
  1. Generated non-isomorphic free-tree counts match OEIS A000055: $t_4=2, t_5=3, t_6=6$ (development) and $t_7=11, t_8=23, t_9=47, t_{10}=106, t_{11}=235$ (held-out), with canonical tree certificates confirming zero isomorphic duplicates.
  2. For every evaluated graph $G$, the row sums $\sum_{j \neq i} \pi_j m_{i,j}$ of the fundamental matrix agree across all starting vertices $i \in V$ within $10^{-11}$, and agree with the spectral sum $\sum_{k=2}^n \frac{1}{1-\lambda_k}$ within $10^{-11}$.
  3. Exact closed-form checks pass on $K_n$ ($K(K_n) = (n-1)^2/n$) and $C_n$ ($K(C_n) = (n^2-1)/6$) for $n = 4..11$.

**Decision Rule**:
- `CORROBORATED` iff H3 passes and both H1 and H2 hold on the held-out set $n \in \{7, 8, 9, 10, 11\}$.
- `FALSIFIED` iff H3 passes and either H1 or H2 fails on the held-out set.
- `UNASSESSED` iff H3 fails or the held-out domain is not completely evaluated within budget.
