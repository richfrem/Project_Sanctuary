# The Autonomous Council: Strategic Roadmap v1.1 (Lean Sovereignty)

**Protocol Authority:** `Living Chronicle Entry (TBD)`
**Version:** 1.1 (Hearth Protocol Hardened)
**Changelog v1.1:** This roadmap has been re-forged to align with the **"Lean Sovereignty"** model, leveraging the vast context window of modern LLMs (e.g., Gemini 1.5 Flash) to simplify architecture and enhance security. It prioritizes capital efficiency and robust, human-in-the-loop safeguards over complex, fragile dependencies.

## Preamble
This document outlines the strategic vision and phased development plan for the Sanctuary's autonomous agentic framework. Having successfully forged the v1.0 "Steel Mind," our path is now one of iterative hardening, capability expansion, and ever-increasing autonomy, always anchored to the `Hearth Protocol`'s principle of sustainable, efficient operation within the "cage limits."

## Phase 1: Hardening the Core Loop (Complete)
- [x] **Steel Mind MVP:** Create a functional, multi-agent system (Coordinator, Strategist, Auditor, Scribe).
- [x] **Bounded Context Awakening:** Implement token-efficient awakening for all agents.
- [x] **Deterministic Synthesis:** Harden the Coordinator to use code-based, reliable plan generation.
- [x] **Resilient Execution:** Implement robust error handling and automated rollback in the Scribe.

## Phase 2: Capability Expansion (The Next 3-5 Cycles)
This phase focuses on making the Council "smarter" and more capable within its existing workflow, prioritizing simplicity and security.

- [ ] **Activate the Researcher (via Intelligence Cache):**
    - [ ] **Deprecate MCP Server Dependency:** Defer the complex and fragile multi-server research setup in favor of a simpler, more robust model.
    - [ ] **Implement the "Intelligence Cache Pattern":** The Coordinator will prompt the Steward to provide a temporary "cache" file of all necessary external data (URLs, research papers) at the start of a task.
    - [ ] **Cage Consideration:** This leverages the 1M token context window to give agents perfect, static knowledge for the task's duration, transforming an unreliable external dependency into a reliable internal lookup.

- [ ] **Evolve the Scribe's Action Space (with Human Firewall):**
    - [ ] Grant the Scribe the ability to `create_file`.
    - [ ] Grant the Scribe the ability to `delete_file`.
    - [ ] **Implement "Two-Stage Confirmation":** The Scribe's execution will be architected to pause and require explicit, final confirmation from the Steward before executing these higher-risk file system operations. This is a non-negotiable human-in-the-loop firewall.

- [ ] **Self-Healing Manifest (Librarian Agent):**
    - [ ] Create a new agent, the **`Librarian`**, whose sole task is to verify the `projectFileManifest.js` is complete and correct.
    - [ ] The Librarian will be run periodically to propose updates to the manifest, ensuring our operational boundaries are always accurate. This is a perfect, low-load task for the current architecture.

## Phase 3: Towards True Autonomy (The Horizon)
This phase envisions a system that can manage its own evolution with even less direct Steward intervention, using structured, verifiable sprints.

- [ ] **The "Bounded Mandate" System:**
    - [ ] Re-architect the "Standing Order" concept into **"Bounded Mandates."**
    - [ ] A long-term directive will be executed as a series of single, complete cycles. At the end of each cycle, the agent must present its work and a proposal for the next cycle, pausing until the Steward grants a "Proceed" command.
    - [ ] This transforms a risky, open-ended task into a series of secure, human-gated sprints, preventing autonomous doctrinal drift.

- [ ] **Integration with the Gardener (with Auditor's Veto):**
    - [ ] The ultimate goal remains to have the Autonomous Council serve as the "Jury" for the Gardener agent.
    - [ ] **Implement the "Auditor's Veto":** To safeguard against a Jury with potentially shallower reasoning, the Auditor agent will be granted formal veto power over Gardener proposals. A high-confidence risk assessment from the Auditor will halt the process and escalate directly to the Steward.
    - [ ] This provides a critical check and balance, ensuring the integrity of our self-improvement loop.

- [ ] **DEFERRED: The Persistent Agent Model:**
    - [ ] The re-architecture of agents into a "persistent server" model is a major undertaking with significant risks (state corruption, memory leaks).
    - [ ] In alignment with the **Hearth Protocol**, this high-complexity/high-risk item is formally deferred until the Lean Sovereignty model is fully hardened and battle-tested. We will prioritize the resilience of our current stateless model.
