---
id: res-semantic-lifecycle-contracts
title: "Explicit Semantic and Lifecycle Contracts for Agent Reliability"
status: KEEP
tags: ["ontology", "shacl", "owl", "solver-aided-verification", "state-transitions", "lifecycle-contracts", "agent-reliability"]
related_missions: ["MISSION-068", "MISSION-036", "MISSION-038"]
source_reading_depth: "abstract_inspection"
last_substantive_review: "2026-11-08"
---

# Explicit Semantic and Lifecycle Contracts for Agent Reliability

**Status:** KEEP  
**Date:** 2026-11-08  
**Topic:** Whether a small, explicit model of entities, relationships and permitted state transitions prevents agent errors that Sanctuary's current schemas, scripts and operating instructions do not already prevent.  
**Related Missions:** [`MISSION-068`](../../missions/MISSION-068.json) (motivating PROPOSED mission; not executed), [`MISSION-036`](../../missions/MISSION-036.json) (evidence and verdict contracts), [`MISSION-038`](../../missions/MISSION-038.json) (lease fencing and idempotent transitions).

---

## 1. Origin and attribution

- **Question:** proposed by Richard (human partner) after Frank Coyle's talk *Why Agentic Systems Need Ontologies* (<https://www.youtube.com/watch?v=Sir59K8ZDPU>).
- **Inspection of the talk:** its indexed description and chapter list only, not a full transcript. No claim here rests on the talk's detailed content.
- **Comparative research design:** proposed by ChatGPT (OpenAI). Recorded and registered by claude-code (Anthropic) on 2026-11-08, with the design preserved as given.

## 2. Research question

> Does a small, explicit model of entities, relationships and permitted state transitions prevent errors that Sanctuary's current schemas, scripts and operating instructions do not already prevent? If it does, does a specialized semantic implementation (SHACL, OWL or a solver) add value over equivalent ordinary-code checks?

This is a research question. It does **not** assume that RDF, OWL, SHACL, Z3, a knowledge graph or a new control plane must be adopted.

## 3. Sources (reading depth: abstracts and introductory material only)

ChatGPT inspected the paper abstracts and the standards' introductory material. Nothing below reflects a full-text methodological review; that review belongs to mission execution, before the experimental protocol is finalized.

1. **W3C OWL 2 Web Ontology Language Primer**: <https://www.w3.org/TR/owl2-primer/>. Relevance: a vocabulary for entities (classes, individuals), relationships (properties) and logical inference over them. Open question: whether open-world inference fits closed, operational lifecycle checks at all.
2. **W3C Shapes Constraint Language (SHACL)**: <https://www.w3.org/TR/shacl/>. Relevance: validation of graph data against explicit constraints ("shapes"). This is the closest standard analogue to cross-record validation of manifests, leases, exports and receipts.
3. **Solver-Aided Verification of Policy Compliance in Tool-Augmented LLM Agents**: <https://arxiv.org/abs/2603.20449>. Relevance: runtime interception of proposed tool calls, with Z3 checks over observable state and arguments. The authors report fewer policy violations while maintaining task accuracy (author-reported; not independently checked here).
4. **Verification-Aware Planning for Multi-Agent Systems (VeriMAP)**: <https://arxiv.org/abs/2510.17109>. Relevance: planner-defined verification functions for each subtask, covering dependencies, acceptance criteria and handoffs.

## 4. Proposed research sequence (for later execution; not started)

1. **Inventory existing enforcement.** Inspect the current mission schema, coordination, lifecycle, preservation, authorization and shared orchestration capabilities. Map each relevant invariant to the code that already enforces it. Examine related Agentic OS capabilities before proposing new machinery. Distinguish missing checks from checks that exist but can be bypassed.
2. **Define a small domain model.** Candidate entities: Mission, Attempt, Evidence, Verdict, Authorization, Lease, SourceRevision, Export, RemoteReceipt. Define only the relationships needed by the selected failure cases. Identify which facts come from trusted system observations and which are only agent assertions.
3. **Compare three approaches on identical cases:**
   - A. Current installed scripts and validators.
   - B. Minimal ordinary-code preconditions and state-transition checks.
   - C. Explicit semantic contracts using SHACL or a solver, if research shows a useful fit. Select one implementation, not every stack.

   Keep policy requirements and available evidence equivalent across B and C; otherwise extra rules could be mistaken for a benefit of the technology.
4. **Historical failures as development examples** (to be confirmed from repository records; do not assume they remain unfixed; include valid counterparts and already-fixed cases):
   - Startup unexpectedly initiating preservation writes.
   - Lease updates resetting a resolved mission to IN_PROGRESS.
   - Declaring delivery complete before required publication or preservation.
   - Confusing an unassessed experiment with a falsified hypothesis.
   - Treating a receipt for an older revision as proof of current persistence.
5. **Freeze a bounded evaluation before held-out cases.** Include invalid and valid transitions, stale or missing evidence, interruption and resume, duplicate requests, and a concurrent state change between validation and execution. Test whether rejection occurs *before* the prohibited side effect. Use isolated fixtures and simulated publication, never live destructive operations or remote writes.

## 5. Provisional hypothesis and measures

**Hypothesis (provisional):** explicit cross-record and transition checks reduce the invalid transitions accepted by the current baseline without materially blocking valid work. A specialized semantic implementation adds value only if it improves coverage, clarity or maintenance over equivalent ordinary-code checks.

**Measures:**
- Invalid transitions accepted.
- Valid transitions incorrectly blocked.
- Failures prevented before side effects.
- Correct handling of stale, unknown and conflicting evidence.
- Implementation and maintenance effort, dependencies and runtime overhead.
- Model tokens, only if a separately justified, bounded live-agent test is run.

## 6. Experimental discipline and limits

- Numeric thresholds, case counts and budgets are specified after the inventory and development phase, then frozen before held-out execution.
- Rules are never derived from held-out failures.
- A deterministic fixture study cannot establish improved real-agent behaviour.
- A logically consistent assertion is not necessarily factually true.
- An ontology does not provide transactional locking, authorization or idempotency merely by describing those concepts.

**Acceptable outcomes:** existing enforcement is sufficient; small improvements to existing scripts are sufficient; a semantic validator provides measurable additional value; or the evidence is insufficient, with limitations recorded and no adoption forced.

## 7. Relation to existing missions

- **MISSION-036** (Preregistered Evidence Contracts) tests contract fields for claims and verdicts. Overlaps on "unassessed vs falsified" and evidence validity. It does not compare approaches across the whole lifecycle.
- **MISSION-038** (Crash-Safe Coordination) tests lease fencing and idempotent transitions under faults. Overlaps on concurrency and duplicate requests. It is about coordination mechanics, not a semantic model.
- MISSION-068 is registered separately because neither mission asks the comparative question (baseline vs ordinary-code checks vs semantic contracts).

## 8. Status of this dossier

Registration-time synthesis only. No experiment has been run, no capability implemented, and no source read beyond abstracts and introductions.
