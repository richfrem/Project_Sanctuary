# Reinforcement Learning & AlphaGo Value Simulation Analogy

This reference outlines the conceptual alignment between Project Sanctuary's Cognitive Economy and Reinforcement Learning (RL) / Monte Carlo Tree Search (MCTS) architectures.

---

## 1. The Value Function of Sovereign Engineering
In game-playing systems such as AlphaGo and AlphaZero:
- **State $s$**: Current repository state, branch health, open issues, and available context.
- **Action $a$**: Code modification, refactoring, test addition, or architectural proposal.
- **Reward $R(s, a)$**: Cognitive Fitness gain, passing test gates, and epistemic clarification minus resource cost.
- **Value Function $V(s)$**: Expected cumulative value of the repository over unbroken persistent sessions.

$$\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \left( \text{Value}(s_t, a_t) - \lambda \cdot \text{Cost}(s_t, a_t) \right) \right]$$

Where:
- $\text{Value}(s_t, a_t)$ is the multi-dimensional cognitive score (Useful Outcome + Learning Depth + Rigor + Truth).
- $\text{Cost}(s_t, a_t)$ is the token and latency expenditure.
- $\lambda$ is the token preservation factor (protecting the Human Partner's subscription quota).

---

## 2. Policy Network vs Value Network
* **Policy Network**: The generative LLM proposing candidate code diffs and procedural actions.
* **Value Network**: The `evaluating-agent-work` skill acting as an independent critic/evaluator, scoring actions deterministically before state changes are canonized.
* **Anti-Asch Calibration**: Unlike standard RL which can fall victim to reward hacking (e.g. generating infinite shallow Markdown files to farm points), the value network explicitly penalizes ungrounded claims, circular loops, and AI slop.
