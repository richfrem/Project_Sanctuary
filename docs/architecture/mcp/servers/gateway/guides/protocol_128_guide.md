# Protocol 128 Guide: The Steward's Command Center

This guide provides an overview of the **Hardened Learning Loop (Protocol 128)**, ensuring that every session's cognitive delta is verified, high-fidelity, and sustainable.

## 🧬 Process Overview
The system establishes a **Zero-Trust Gate** between the agent's work and the project's permanent memory (RAG DB / Git).

![[protocol_128_learning_loop.png|protocol_128_learning_loop]]

*[[protocol_128_learning_loop.mmd|Source: protocol_128_learning_loop.mmd]]*

> [!IMPORTANT]
> **HITL (Human-in-the-Loop)**: Protocol 128 v3.5 implements a **Dual-Gate** HITL model. 
> 1. **Strategic Review (Gate 1)**: You verify the AI's *reasoning* and documentation (ADRs/Learnings).
> 2. **Technical Audit (Gate 2)**: You verify the AI's *implementation* (Code Snapshot/Red Team Packet).

## 🔗 Key Resources
- **[[071_protocol_128_cognitive_continuity|ADR 071: Decision Record]]**: Why we chose the Red Team Gate and how the architecture works.
- **[[sanctuary-learning-loop|Protocol 128: Constitutional Mandate]]**: The step-by-step guide for agents to acquire and preserve knowledge.
- **[[cognitive_primer|Cognitive Primer]]**: The "Rules of Reality" that agents must follow on every boot.

### Supporting Skills (`.agent/skills/`)
- **`learning-loop`**: Portable skill encoding the 10-phase Protocol 128 workflow.
- **`memory-management`**: Tiered memory system (hot cache ↔ deep storage) for cognitive continuity.
- **`code-review`**: Confidence-scored review for pre-commit quality gates.

## 💓 The "Learning Package Snapshot" Pulse
When an agent calls `cortex_learning_debrief`, it triggers a series of autonomous observations:
1. **Source of Truth**: Scans `git diff` for physical evidence.
2. **Auto-Discovery**: Identifies high-signal recently modified files.
3. **Instructional Bundle**: Returns the full constitutional context (SOPs, Protocols, Primer).
4. **Successor Context**: Reads the most recent `learning_package_snapshot.md` for total continuity.

## 🛠️ Rapid-Fire Learning Cycle
The agent follows these steps to achieve the "Final Seal":
1. **Refinement**: Update the Recursive Learning SOP with logical optimizations.
2. **Snapshot**: `python scripts/capture_code_snapshot.py --manifest .agent/learning/manifest.json`
3. **The Seal**: Ensure output is saved to `.agent/learning/learning_package_snapshot.md`.
4. **Persistence**: Use `git_smart_commit` referencing the SEAL to lock in the cognitive delta.

---
*Status: Canonical Guide (v1.0)*
