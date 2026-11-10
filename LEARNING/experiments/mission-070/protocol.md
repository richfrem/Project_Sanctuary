# MISSION-070 Protocol: Macro-MoE JEV Router & OpenRouter Micro-Tier
**Status:** Frozen before held-out benchmark execution  
**Reference Paper:** arXiv:2610.39302 (Balestriero & LeCun, Meta FAIR / NYU)  
**Associated Manifest:** `LEARNING/missions/MISSION-070.json`  

---

## 1. Core Objectives & Hypotheses

- **H1 (Zero-Token Latency):** The non-generative Joint-Embedding Vector (JEV) decision-tree router computes hierarchical routing decisions in $<3.0$ ms per task on standard workstation CPU, generating strictly $0$ autoregressive tokens.
- **H2 (Frontier Quota Shielding):** The JEV router successfully routes $\ge 80\%$ of routine tasks (mechanical formatting, linting, syntax repair, triage) to Tier 1 (OpenRouter micro-models) or Tier 2 (Local MLX), achieving $\ge 70\%$ frontier token quota shielding.
- **H3 (Epistemic Safety Boundary):** The JEV router misclassifies $\le 1\%$ (at most 0 out of 10 in held-out test) of high-leverage architectural / epistemic tasks into the micro-tier.
- **H4 (Micro-Tier Execution Pass-Rate):** Pinned micro-models (`meta-llama/llama-3.2-1b-instruct`, `meta-llama/llama-3.2-3b-instruct`, `qwen/qwen3-coder-30b-a3b-instruct`) achieve $\ge 75\%$ first-pass validation on mechanical chores (JSON repair, table formatting, AST linting) with verifiable determinism.
- **H5 (Hardware & Budget Preservation):** Local disk footprint addition is strictly $<500$ MB (target: 0 MB weights downloaded), and operational cost remains within the strict $\$10.00$ monthly spending cap ($<\$0.001$ per routine task execution).

---

## 2. Deterministic Falsification Gates

The mission outcome is evaluated strictly against:
- **GATE 1 (Latency & Token Gate):** Mean routing latency must be $<5.0$ ms (target $<3.0$ ms) and routing token burn $= 0$. If any LLM prompt is required for routing, Gate 1 FAILS.
- **GATE 2 (Architectural Misclassification Gate):** Misclassification of architectural/epistemic tasks into Tier 1 must be $\le 5\%$. If an architectural proof is sent to a 1B model, Gate 2 FAILS.
- **GATE 3 (Micro-Tier First-Pass Gate):** First-pass execution pass-rate on mechanical formatting and linting tasks must be $\ge 75\%$.
- **GATE 4 (Resource & Cost Gate):** Model downloads to local SSD must be $<500$ MB, and average task dispatch cost must be $<\$0.0005$.

**Popperian Verdict Matrix:**
- `CORROBORATED`: All Gates (G1, G2, G3, G4) PASS and Hypotheses H1–H5 corroborated.
- `FALSIFIED`: Any Gate (G1, G2, G3, G4) FAILS.
- `UNASSESSED`: Test execution halted prematurely due to network/system boundary.

---

## 3. Architecture Specification

### Tier 0: In-Process JEV Decision Tree (<3ms, 0 tokens)
Hierarchical hyperplane projection:
1. **Level 1 (Constitutional Intent):** `READ_ONLY` vs `STATE_CHANGING`.
2. **Level 2 (Complexity Class):**
   - `MECHANICAL_CHORE` (formatting, JSON, linting, regex, docstrings).
   - `CODING_SYNTAX` (unit tests, AST transforms, small bug fixes, script scaffolding).
   - `GENERAL_TRIAGE` (inbox sorting, tag generation, backlog indexing, diff summary).
   - `EPISTEMIC_ARCHITECTURAL` (novel algorithms, proofs, constitutional doctrine, multi-agent arbitration).
3. **Level 3 (Target Routing):**
   - `MECHANICAL_CHORE` / `CODING_SYNTAX` -> Tier 1 OpenRouter Micro-Tier (pinned: `llama-3.2-1b`, `llama-3.2-3b`, `qwen3-coder-30b-a3b`).
   - `OFFLINE_REQUIRED` -> Tier 2 Local MLX (`localhost:8080`).
   - `EPISTEMIC_ARCHITECTURAL` -> Tier 3 Frontier (Claude 3.5 Sonnet, Codex, Gemini Argon).

### Tier 1: OpenRouter Micro-Tier Client
- Enforces strict monthly spending cap of $\$10.00$.
- Barred from using third-party auto-routers (`openrouter/free`).
- Automatic fallback to local MLX on timeout / network partition.

---

## 4. Evaluation Design

- **Anchor Calibration Set:** 20 historical tasks from `chronicle/ledger.json` and mission logs.
- **Held-Out Test Matrix:** 50 categorized engineering tasks (20 Mechanical, 15 Coding, 5 Triage, 10 Architectural).
- **Execution Mode:**
  1. Profile JEV router on 50 tasks: measure decision latency, token usage, routing tier, and confidence.
  2. Execute sample micro-tier tasks on live OpenRouter API to verify execution accuracy and token cost.
  3. Validate fallback behavior and spending cap safety guards.
