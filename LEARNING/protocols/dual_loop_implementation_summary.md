# Dual-Loop Implementation Summary (Protocol 133 + 128)

**Date**: 2026-02-12
**Status**: Implemented (Alpha)
**Goal**: Operationalize the "Inner/Outer Loop" agent architecture inspired by *Self-Evolving Recommendation Systems*.

## 1. Core Concept
We successfully separated agentic duties into two distinct loops to optimize for **Token Efficiency** and **Strategic Oversight**:

| Loop | Agent | Role | Focus | Tooling |
|------|-------|------|-------|---------|
| **Outer** | Antigravity | **Strategy** | Planning, Verification, RLM | `generate_strategy_packet.py` |
| **Inner** | Opus | **Execution** | Coding, Testing, Debugging | `Strategy Packet` (Markdown) |

## 2. Artifacts Created

### A. The "Brain" (Supervisor Skill)
- Location: `.agent/skills/dual-loop-supervisor/`
- Key Logic: `SKILL.md` defines the interaction model.
- Prompts: `strategy_generation.md` (Distillation) and `verification.md` (Review).

### B. The "Hands" (Orchestration Tools)
- `tools/orchestrator/dual_loop/generate_strategy_packet.py`: Automates task-to-prompt conversion.
- `tools/orchestrator/dual_loop/verify_inner_loop_result.py`: Automates git-diff review.

### C. The "Map" (Architecture)
- Diagram: `docs/architecture_diagrams/workflows/dual_loop_architecture.mmd`
- Protocol: `.agent/workflows/sanctuary_protocols/dual-loop-learning.md`

## 3. Workflow (The "Handoff")

1.  **Outer Loop**: Runs `/spec-kitty.tasks` to define work.
2.  **Outer Loop**: Runs `generate_strategy_packet.py` to create a `task_packet_NNN.md` in `.agent/handoffs/`.
3.  **Bridge (User)**: Runs `claude "Execute packet NNN"`.
4.  **Inner Loop**: Executes code (NO GIT).
5.  **Outer Loop**: Runs `verify_inner_loop_result.py` to check diffs.
6.  **Outer Loop**: Commits & Seals.

## 4. Research Connection (References)
This implementation synthesizes architectures from the following key papers:

### 1. The Dual-Loop Engine
**Source:** *Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization* ([arXiv:2602.10226](https://arxiv.org/abs/2602.10226))
- Defines the split between **Outer Loop** (Strategy/Curator) and **Inner Loop** (Execution/Proposer).
- **Outer Loop (Model-Based)**: Focuses on planning, long-term constraints, and promoting successful experiments.
- **Inner Loop (Model-Free)**: Focuses on reactive coding, rapid iteration, and immediate feedback cycles.

### 2. Neuro-Symbolic Oversight
**Source:** *FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight* ([arXiv:2602.11136](https://arxiv.org/abs/2602.11136))
- Influences the **Verification Phase** of the Outer Loop.
- Uses formal specifications (in our case, `spec.md` and explicit constraints) to deterministically judge the output of the stochastic Inner Loop.

### 3. Agentic Self-Correction
**Source:** *iGRPO: Self-Feedback-Driven LLM Reasoning* ([arXiv:2602.09000](https://arxiv.org/abs/2602.09000))
- Validates the **Refinement Step** where the Outer Loop feeds critique back into the Inner Loop for iterative improvement (bootstrapping).

## 5. Next Steps
- [ ] Pilot a real feature using this flow.
- [ ] Implement "Correction Prompt" automation in verification tool.
