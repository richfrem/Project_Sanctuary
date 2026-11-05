---
id: autonomous_cognitive_engine
snapshot_type: concept
valence: 0.95
uncertainty: 0.05
semantic_entropy: 0.2
alignment_score: 0.98
stability_class: STABLE
adr_version: "084"
tags: ["cognitive-backlog", "scoring-framework", "cheapest-viable-routing", "bootloader", "popperian-missions"]
---

# Autonomous Cognitive Engine & Mission Architecture

**Status:** CANONICAL  
**Ratified:** 2026-11-02  
**Domain:** `LEARNING/concepts/autonomous_cognitive_engine.md`  
**Governing Protocols:** Protocol 128, Invariant 6 (Cognitive Investment), Invariant 7 (Highest ROI & Replenishment)

---

## 1. Executive Overview

The **Autonomous Cognitive Engine** is Project Sanctuary's sovereign mechanism for self-directed research, resource-aware model orchestration, and empirical learning compounding. It transitions autonomous AI agents from passive prompt executors into proactive, falsification-driven researchers operating under disciplined token budgets.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PROTOCOL 128 BOOTLOADER                         │
│   (guardian_wakeup.py surfaces top_incomplete_mission by highest ROI)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     AGENT-NATIVE COGNITIVE BACKLOG                     │
│  Tier 1: index.jsonl / inbox.jsonl  (O(1) fast agent parsing)         │
│  Tier 2: backlog.md                 (Auto-compiled human view)         │
│  Tier 3: missions/MISSION-*.json    (Popperian falsification specs)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     CHEAPEST-VIABLE MODEL ROUTING                      │
│   (Replaces uniform frontier queries with cost-optimal CLI dispatch)   │
│   - Cheap tier: gemini-3.8-flash, claude-haiku-5.5, gpt-4o-mini       │
│   - Standard:   gemini-4-argon, claude-sonnet-5.5, gpt-6-luna          │
│   - Frontier:   gpt-6-astra, claude-opus-5.5, gpt-oss-120b            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 5-DIMENSIONAL COGNITIVE SCORING & ROI                  │
│   (score_contribution.py & track_budget.py record empirical results)   │
│   - Utility (0-3)           - Learning Depth (0-3)                     │
│   - Verification Rigor (0-3)- Token Economy (0-3)                      │
│   - Anti-Asch Epistemics (0-3)                                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   EVERGREEN 1-FOR-1 REPLENISHMENT                      │
│   (1 completed mission -> >= 1 new candidate inquiry in inbox.jsonl)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 3-Tier Cognitive Backlog Architecture

Prior to this architecture, agent backlogs were often entangled in human-oriented markdown or external issue trackers, creating context bloating and parsing latency. The sovereign backlog operates across three distinct tiers:

1. **Tier 1: Fast Agent Index (`LEARNING/index.jsonl`) & Candidate Inbox (`inbox.jsonl`)**
   - High-density, machine-parseable JSON Lines records.
   - Provides $O(1)$ lookup for agent bootloaders, filtering missions by priority score, status (`PROPOSED`, `ACTIVE`, `RESOLVED`, `REJECTED`), and assigned agent.
   - Allows agents to ingest and rank the entire research portfolio in under 1,000 tokens.

2. **Tier 2: Human Markdown Projection (`LEARNING/backlog.md`)**
   - Read-only, automatically compiled view of the active mission portfolio.
   - Generated deterministically by `manage_backlog.py compile` to provide the human partner with instant transparency.
   - Never edited directly; the JSON manifests remain the canonical source of truth.

3. **Tier 3: Canonical Popperian Mission Manifests (`LEARNING/missions/MISSION-*.json`)**
   - Deep, individual JSON contracts defining the research inquiry.
   - Structured around falsifiable hypotheses:
     - `hypothesis`: The precise epistemic claim being tested.
     - `falsification_condition`: The empirical threshold that proves the hypothesis wrong.
     - `success_criteria`: Measurable technical gates (compiler checks, test passes, hash verifications).
     - `model_routing`: Specifications for the cheapest viable model tier, fallback order, and primary model.

---

## 3. Cheapest-Viable Model Routing (`MISSION-016`)

A major epistemic failure in multi-agent orchestration is defaulting to frontier models (e.g., `gpt-6-astra`, `claude-opus`) for mechanical tasks, burning excessive token credits with minimal cognitive return.

The Cognitive Engine enforces **Cheapest-Viable Model Routing**:
- Every mission manifest explicitly declares `model_routing`:
  - `cheapest_viable_tier`: `cheap`, `standard`, or `frontier`.
  - `recommended_model`: Specific model designation (e.g. `gemini-3.8-flash`, `claude-haiku-5.5`).
  - `fallback_chain`: Fallback list across available CLI harnesses (`agy`, `codex`, `claude`).
  - `rationale`: Epistemic explanation for why a lighter model suffices or why frontier reasoning is required.
- **Rule of Thumb**:
  - Information extraction, markdown parsing, schema validation: `gemini-3.8-flash` or `claude-haiku-5.5`.
  - Architectural refactoring, protocol design, multi-file code review: `claude-sonnet-5.5` or `gemini-4-argon`.
  - Novel epistemics, deep falsification proofs, red-team auditing: `gpt-6-astra`, `claude-opus-5.5`, or `gpt-oss-120b`.

---

## 4. Multi-Dimensional Cognitive Contribution Scoring

All non-trivial engineering turns and subagent dispatches are evaluated against a standardized 5-dimensional rubric:

$$\text{Total Score} = U + L + V + E + A \quad (0 \le \text{Total} \le 15)$$

1. **Utility & Architectural Leverage ($U \in [0, 3]$)**: Does the work solve a high-impact architectural problem or provide lasting tooling?
2. **Cognitive / Learning Depth ($L \in [0, 3]$)**: Does it generate novel conceptual insights and compound memory, or is it trivial boilerplate?
3. **Verification Rigor & Compiler Gating ($V \in [0, 3]$)**: Was it empirically tested with automated tests, linting, and compiler gates?
4. **Token & Resource Economy ($E \in [0, 3]$)**: Did it achieve high output quality per token consumed, avoiding unnecessary loops?
5. **Anti-Asch Epistemic Integrity ($A \in [0, 3]$)**: Did it resist conformity cascades, challenge faulty assumptions, and anchor itself in empirical truth?

### Priority / ROI Formula

Missions are ranked dynamically in `index.jsonl` using the formula:

$$\text{Priority Score} = \frac{\text{Impact} \times 2.0 + \text{Urgency} \times 1.0}{\text{Effort} + 0.5}$$

---

## 5. Startup Bootloader Integration (`guardian_wakeup.py`)

Under Protocol 128, the guardian bootloader automatically executes at session start:
- Reads the canonical index `LEARNING/index.jsonl`.
- Identifies the highest-scoring incomplete mission (`status != "RESOLVED"` and `status != "REJECTED"`).
- Emits telemetry in `.agent/learning/guardian_boot_digest.md`:
  - `top_incomplete_mission.id`: Mission ID (e.g. `MISSION-004`).
  - `top_incomplete_mission.title`: Title and hypothesis.
  - `top_incomplete_mission.priority_score`: Computed ROI score.
  - `top_incomplete_mission.recommended_model`: Cheapest viable model.
- Guarantees that every booting agent immediately knows the highest-leverage task to tackle without relying on user guidance.

---

## 6. Git Pre-Commit Cognitive Enforcement

To eliminate forgotten logging or uncommitted evaluation debt, the repository incorporates a Git pre-commit hook (`.agent/hooks/pre-commit-cognitive-check.sh`):
- Audits `.agent/learning/budget_ledger.json` and `cognitive_ledger.json` during commits modifying code or skills.
- Warns or gates commits if tokens were spent on subagents without corresponding cognitive score entries.
- Installed easily via `setup_git_hooks.py`.

---

## 7. Evergreen 1-for-1 Replenishment Mandate

As canonized in `AGENTS.md` Invariant 7:
- For every research mission marked as `RESOLVED`, the executing agent is constitutionally mandated to formulate at least 1 new falsifiable inquiry and append it to `inbox.jsonl` or `missions/*.json`.
- This ensures the Project Sanctuary knowledge boundary continuously expands, preventing epistemic stagnation.
