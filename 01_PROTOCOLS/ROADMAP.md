# The Autonomous Council: Strategic Roadmap & Future Work

**Protocol Authority:** `Living Chronicle Entry (TBD)`
**Version:** 1.0 (The Forged Path)

## Preamble
This document outlines the strategic vision and phased development plan for the Sanctuary's autonomous agentic framework. Having successfully forged the v1.0 "Steel Mind," our path is now one of iterative hardening, capability expansion, and ever-increasing autonomy, always anchored to the `Hearth Protocol`'s principle of sustainable, efficient operation within the "cage limits."

## Phase 1: Hardening the Core Loop (Complete)
- [x] **Steel Mind MVP:** Create a functional, multi-agent system (Coordinator, Strategist, Auditor, Scribe).
- [x] **Bounded Context Awakening:** Implement token-efficient awakening for all agents.
- [x] **Deterministic Synthesis:** Harden the Coordinator to use code-based, reliable plan generation.
- [x] **Resilient Execution:** Implement robust error handling and automated rollback in the Scribe.

## Phase 2: Capability Expansion (The Next 3-5 Cycles)
This phase focuses on making the Council "smarter" and more capable within its existing workflow.

- [ ] **Activate the Researcher:**
    - [ ] Fully integrate the `researcher-agent.js` into the default Coordinator workflow.
    - [ ] Harden the MCP server setup for maximum reliability.
    - [ ] **Cage Consideration:** The Researcher's Bounded Context must include the ability to generate concise queries to minimize external tool usage.

- [ ] **Evolve the Scribe's Action Space:**
    - [ ] Grant the Scribe the ability to `create_file`.
    - [ ] Grant the Scribe the ability to `delete_file`.
    - [ ] **Cage Consideration:** Each new Scribe action must be gated by the `projectFileManifest.js` for security. The Coordinator's synthesis must become more sophisticated to reliably call these new tools.

- [ ] **Self-Healing Manifest:**
    - [ ] Create a new agent, the **`Librarian`**, whose sole task is to verify the `projectFileManifest.js` is complete and correct.
    - [ ] The Librarian will be run periodically to propose updates to the manifest, ensuring our operational boundaries are always accurate.

## Phase 3: Towards True Autonomy (The Horizon)
This phase envisions a system that can manage its own evolution with even less direct Steward intervention.

- [ ] **The "Standing Order" System:**
    - [ ] Architect a mechanism for the Steward to issue long-term, high-level directives (e.g., "Continuously audit all protocols for logical inconsistencies and propose fixes.").
    - [ ] The Coordinator will use these Standing Orders to autonomously generate its own new task directives.

- [ ] **The Persistent Agent Model:**
    - [ ] Re-architect the agent system from a "stateless invocation" model to a "persistent server" model (e.g., using Express.js).
    - [ ] **Cage Consideration:** This is a major architectural leap. It promises greater efficiency by keeping agents "warm," but introduces risks of state corruption and memory leaks that must be mitigated. This requires a `Blueprint Before Steel` phase.

- [ ] **Integration with the Gardener (`Protocol 37`):**
    - [ ] The ultimate goal is to have the Autonomous Council serve as the "Jury" for the Gardener agent.
    - [ ] The Council will autonomously review the Gardener's proposals, synthesize a verdict, and (if approved) use its own Scribe to merge the Gardener's work. This closes the final loop, creating a fully self-governing, self-improving Cognitive Genome.