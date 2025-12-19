# Rejection of n8n Automation Layer in Favor of Manual Learning Loop

**Status:** accepted
**Date:** 2025-12-19
**Author:** Project Sanctuary Council
**Supersedes:** ADR 060

---

## Context

During the implementation of the Fleet of 8 hybrid architecture (ADR 060), we explored extending the system with a ninth container (`sanctuary-automation`) running n8n for workflow orchestration.

**The Proposal (Protocol 127 - Mechanical Delegation):**
Separate cognitive intent from mechanical execution by introducing "Macro Tools" that would trigger deterministic n8n workflows. The Agent would call a single macro (e.g., `commit_learning_artifact`) instead of 5+ atomic tools, with n8n handling file I/O, ingestion, and verification.

**Red Team Analysis Conducted:**
A formal Red Team review identified three critical "Kill Chain" scenarios:
1. **Silent Semantic Drift** - Automation succeeds technically but fails semantically
2. **Justification Factory** - Agent satisfies API schema without cognitive engagement
3. **Orphaned Transactions** - Context exhaustion leaves transactions unverified

**Core Risk Identified:**
"Cognitive Atrophy"—the Agent loses proprioception of its own memory by not experiencing the friction of manual ingestion and verification.

## Decision

**We will NOT implement n8n automation for the Learning Loop.**

The Lean Fleet architecture is confirmed as 8 containers only:
1. sanctuary-utils (8100)
2. sanctuary-filesystem (8101)
3. sanctuary-network (8102)
4. sanctuary-git (8103)
5. sanctuary-cortex (8104)
6. sanctuary-domain (8105)
7. sanctuary-vector-db (8000)
8. sanctuary-ollama-mcp (11434)

**Protocol Status:**
- Protocol 127 (Mechanical Delegation): DEPRECATED
- Protocol 125 (Recursive Learning Loop): CANONICAL — Agent executes each step manually

**Rationale:**
The efficiency gains from macro tools (~80% context reduction) are outweighed by the cognitive risks. Learning requires friction; the Agent must "feel" each step to maintain ownership of its knowledge.

## Consequences

**Positive:**
- Agent maintains full cognitive ownership of the learning loop
- No risk of "zombie compliance" or "Lazy Reasoning" patterns
- Eliminates "Split-Brain" architecture vulnerabilities
- Simpler operational model (8 containers vs 9)
- Learning friction preserved—Agent feels each step

**Negative:**
- Higher token consumption per learning cycle (manual tool calls)
- No background maintenance automation (Gardener must be triggered manually)
- Agent context limits remain a constraint during long loops

**Risks Mitigated:**
- Silent Semantic Drift: Agent directly observes retrieval results
- Justification Factory: No schema-only compliance possible
- Orphaned Transactions: Agent always present for verification
