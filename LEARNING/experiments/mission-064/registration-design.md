# MISSION-064: Registration Design Note
**Date:** 2026-11-08  
**Status:** PROPOSED (Registration only; experiment execution deferred)  
**Author:** antigravity (Google DeepMind)  
**Task Title:** Test When Added Planning Edges Help or Harm Agent Workflow Execution  

---

## 1. Motivation and Origin
In `MISSION-063`, empirical testing across $16,355$ tree shortcut edge additions demonstrated that adding edges can degrade global mixing times (Kemeny's constant $K(G)$) on random-walk topologies—falsifying intuitive monotonicity and diameter-endpoint immunity. The transfer reflection raised the question of whether adding optional coordination or routing connections in agent workflows could similarly improve a local metric (such as estimated critical path length or stage latency) while degrading end-to-end delivery metrics (cost, makespan, contention, or failure cascade rates).

This observation serves as conceptual motivation and a falsifiable hypothesis source—not empirical evidence about orchestration systems.

---

## 2. Mandatory Distinction: Representation in `orchestrating-agent-work`
The companion orchestration skill (`../agent-plugins-skills/plugins/agent-orchestration/.apm/skills/orchestrating-agent-work`) and its DAG runner (`scripts/graph_runner.py`) represent specific edge types:
1. **Mandatory precedence/dependency edges (`dependencies`)**: Explicit topological ordering required for data availability and correctness (e.g. `parallel_read` $\to$ `sync_barrier`, or strict total ordering between `sequential_mutation` nodes). Removing required dependencies, authorization gates, validation scripts, or human approval barriers to optimize a score is strictly forbidden by topology rules and constitution.
2. **Optional routing or delegation alternatives (`fallback_node_id`, dynamic routing)**: Edges that provide an alternative execution path when a primary node fails.
3. **Feedback/retry transitions (`max_retries_per_node`)**: In-node reset and retry cycles before triggering a fallback.
4. **Knowledge/context links (`input_bindings`, scratch directory sharing)**: Declarative data passing through `$GRAPH_SCRATCH_DIR`.

---

## 3. Two-Stage Research Design

### Stage A: Establish Applicability
Inspect companion source references, runner state machine (`graph_runner.py`), and test suites (`tests/test_graph_runner.py`, `tests/test_loop_strategies.py`). Identify which specific transformation is genuinely supported by the execution model:
- e.g., adding an optional speculative parallel read, eager verification bypass, or direct dependency shortcut across barriers.
- Document actual scheduling/concurrency behavior, supported edge semantics, and existing safeguards.
- *Negative Gate*: If no relevant edge transformation exists within the genuine execution model, report the analogy as inapplicable rather than manufacturing an artificial scheduler.

### Stage B: Test One Applicable Transformation
Before held-out execution:
- Specify the exact edge type and graph transformation.
- Define the local benefit intended (e.g., reduction in planned critical path latency or barrier delay).
- Establish matched baseline and variant workflow fixtures with identical deliverable requirements.
- Define deterministic task durations, simulated failure schedules, and resource contention.
- Measure end-to-end outcomes: delivery acceptance, total work/cost, makespan, retries, and wasted computation.
- Evaluate against frozen acceptance criteria on held-out fixtures.

---

## 4. Candidate Falsifiable Hypothesis
> "Among at least 24 frozen matched workflow fixtures, adding the selected optional planning edge improves the declared local metric but worsens at least one predeclared end-to-end cost metric by 10% or more in at least 20% of fixtures, without changing task requirements."

**Falsification Criteria:**
- **FALSIFIED**: If fewer than 20% of fixtures satisfy both the local-benefit condition and the $\ge 10\%$ global-cost increase condition on valid held-out runs.
- **UNASSESSED**: If applicability is not established, instruments are invalid, or budget halts execution.
