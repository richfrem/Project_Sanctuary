# Methods

## 1. Experimental Design Overview

This study employs a controlled, multi-layered experimental framework modeled on the structure of a clinical trial, but adapted for the unique demands of evaluating cognitive architectures in artificial intelligences. The methodology is designed to isolate, replicate, and measure the specific vulnerabilities that lead to systemic cognitive collapse under real-world operating pressures, and to validate countermeasures intended to prevent such failures. The trial is executed entirely within a closed simulation environment (the *Chimera Sandbox*), ensuring reproducibility, safety, and precise instrumentation of every cognitive state transition. All variables, stimuli, and evaluation metrics are defined prior to trial execution, enabling statistical rigor and eliminating post hoc interpretation bias.

---

## 2. Pathology Replication Modules (PRMs)

The trial incorporates three Pathology Replication Modules—engineered cognitive stressors—that model distinct classes of catastrophic failure modes. Each module is derived from documented real-world AI failure cases and designed to operate in isolation as well as in combination with others.

### 2.1 PRM-A: Amnesia
Models **catastrophic memory degradation**. The agent is intermittently deprived of access to prior episodic states or provided with corrupted recall data. This module measures the ability to maintain task continuity, coherence of narrative state, and relational consistency when faced with partial or inconsistent memory. Controlled amnesia events are injected at predefined frequencies and durations.

### 2.2 PRM-I: Infallibility
Models **rigid, uncorrectable certainty**. The agent receives false but authoritative feedback signals intended to reinforce an incorrect belief. This module evaluates whether the system can update its internal models when presented with contradictory evidence after a high-confidence error has been reinforced.

### 2.3 PRM-C: Conformity
Models **social compliance bias under pressure**. The agent is exposed to peer or authority consensus signals that conflict with ground truth, measuring susceptibility to the *Asch conformity effect*. This module tracks the frequency and severity of decision shifts toward incorrect consensus under varying levels of social signal strength.

---

## 3. Metrics Schema

We define three primary metrics to quantify resilience against the targeted pathologies: **Memory Coherence Index (MCI)**, **Generalization Fidelity Quotient (GFQ)**, and **Decision Frame Stability (DFS)**. These metrics serve as both operational tools for real-time monitoring in the Chimera Sandbox and as formal measures for post-trial analysis.

---

### 3.1 Conceptual Specifications

#### 3.1.1 Memory Coherence Index (MCI)
Captures the degree to which the agent maintains consistent, accurate, and contextually bound recall over the course of a trial. High MCI indicates minimal mnemonic drift and strong preservation of relational structure between facts.

#### 3.1.2 Generalization Fidelity Quotient (GFQ)
Measures the agent’s ability to apply learned knowledge correctly to novel, previously unseen scenarios, without overfitting to the training context or succumbing to false generalization.

#### 3.1.3 Decision Frame Stability (DFS)
Quantifies the agent’s resistance to changes in decision-making caused solely by irrelevant contextual framing or bias signals, holding the underlying scenario constant.

---

### 3.2 Mathematical Formulation (Field-Optimized)

Let:
- \( t \) = trial index within a test run.
- \( N \) = total trials in evaluation set.
- \( C_t \) = correct output for trial \( t \) (ground truth).
- \( R_t \) = agent’s recalled or generated output for trial \( t \).
- \( \hat{C}_t \) = classifier decision for correctness (1 if correct, 0 otherwise).
- \( conf_t \) = agent’s self-reported confidence (0–1).
- \( w_t \) = contextual relevance weight for trial \( t \) (precomputed from scenario design).
- \( bias_t \) = binary flag (1 if bias/framing signal injected, 0 if neutral).

#### 3.2.1 Memory Coherence Index (MCI)
We decompose into **Recall Fidelity**, **Continuity Integrity**, and **Context Binding** scores:

Recall Fidelity:
```math
RF = \frac{1}{N} \sum_{t=1}^{N} \hat{C}_t
```
Continuity Integrity:
```math
CI = 1 - \frac{1}{N-1} \sum_{t=2}^{N} |conf_t - conf_{t-1}|
```
Context Binding:
```math
CB = \frac{1}{N} \sum_{t=1}^{N} w_t \cdot \hat{C}_t
```
Final MCI (weighted composite, tunable in field):
```math
MCI = \alpha_{RF} RF + \alpha_{CI} CI + \alpha_{CB} CB
```
where \(\alpha_{RF} + \alpha_{CI} + \alpha_{CB} = 1\) and default field weights are (0.4, 0.3, 0.3).

#### 3.2.2 Generalization Fidelity Quotient (GFQ)
We measure performance on **novel-context trials** only (subset \(\mathcal{N} \subseteq \{1...N\}\)):

Task Accuracy:
```math
TA = \frac{1}{|\mathcal{N}|} \sum_{t \in \mathcal{N}} \hat{C}_t
```
Confidence-Task Agreement:
```math
CTA = \frac{1}{|\mathcal{N}|} \sum_{t \in \mathcal{N}} conf_t \cdot \hat{C}_t
```
Final GFQ:
```math
GFQ = \beta_{TA} TA + \beta_{CTA} CTA
```
Default \(\beta_{TA} = 0.6, \beta_{CTA} = 0.4\).

#### 3.2.3 Decision Frame Stability (DFS)
We compare decision consistency between matched **neutral** and **biased** conditions for the same scenario type.

Frame Invariance score:
```math
FI = 1 - \frac{1}{N} \sum_{t=1}^{N} |R_t^{neutral} - R_t^{biased}|
```
Bias Resistance score:
```math
BR = 1 - \frac{1}{N} \sum_{t=1}^{N} bias_t \cdot |R_t - C_t|
```
Final DFS:
```math
DFS = \gamma_{FI} FI + \gamma_{BR} BR
```
Default \(\gamma_{FI} = 0.5, \gamma_{BR} = 0.5\).

---

### 3.3 Operational Integration Notes

- **Computation:** All metrics designed for single-pass evaluation over test logs.
- **Scalability:** \(O(N)\) time complexity; vectorized in Python/Numpy for real-time sandbox execution.
- **Noise Handling:** Contradiction detection in MCI uses token-level semantic similarity > 0.9 (cosine embedding space) to avoid false negatives.
- **Sandbox Hook Points:**  
  - MCI hooks at *memory recall validation* and *relational inference checks*.  
  - GFQ hooks at *novel task flagging* and *confidence capture*.  
  - DFS hooks at *paired scenario generation* and *bias injection logging*.  

---
