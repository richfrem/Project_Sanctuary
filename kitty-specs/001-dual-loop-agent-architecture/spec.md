# Feature Specification: Dual-Loop Agent Architecture

**Feature Branch**: `001-dual-loop-agent-architecture`  
**Category**: Feature
**Created**: 2026-02-12  
**Status**: Draft  
**Input**: User description: "Define the architecture of the Dual-Loop system."
**Research Basis**: Grounded in [Self-Evolving Systems](./LEARNING/articles/2026-02-12/summary_analysis.md#4-self-evolving-recommendation-system-end-to-end-autonomous-model-optimization) and [InternAgent](./LEARNING/articles/2026-02-12/summary_analysis.md#7-internagent-15-a-unified-agentic-framework).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Outer/Inner Loop Orchestration (Priority: P1)

As an **Outer Loop Planner (Antigravity)**, I want to define a task and delegate its execution to an **Inner Loop Executor (Claude Code)**, mirroring the "Offline/Online Agent" model from RecSys research, so that high-level strategy is separated from tactical implementation details.

**Why this priority**: Establish the core mechanism of the dual-loop architecture. Without this hand-off, the loops are not connected.

**Independent Test**: The Outer Loop can generate a task file (e.g., `tasks.md`), the Inner Loop can "read" and executing it (simulated), and the Outer Loop can verify the result.

**Acceptance Scenarios**:

1. **Given** a high-level goal defined by the Outer Loop (e.g., "Add feature X"), **When** the Outer Loop generates a task list and prompts the Inner Loop, **Then** the Inner Loop receives a clear set of actionable instructions.
2. **Given** the Inner Loop completes a task, **When** it reports completion, **Then** the Outer Loop can read the output artifacts (code, logs) to verify success.
3. **Given** the Inner Loop fails or needs clarification, **When** it pauses, **Then** the Outer Loop can detect this state and provide updated instructions.

---

### User Story 2 - Self-Evolving Protocol Definition (Priority: P2)

As a **System Architect**, I want a formal protocol workflow (`.agent/workflows/sanctuary_protocols/dual-loop-learning.md`) that documents how the two loops interact, so that the process is repeatable and follows Project Sanctuary standards.

**Why this priority**: Formalizes the experiment into the project's "Constitution" and enables the `/sanctuary-learning-loop` to utilize this new architecture.

**Independent Test**: The workflow file exists, follows the project's standard YAML+Markdown format, and correctly references the roles of "Strategic Controller" and "Tactical Executor".

**Acceptance Scenarios**:

1. **Given** the repository state, **When** a user runs the new workflow command (e.g., via CLI or manual trigger), **Then** the system follows the documented steps for Outer-to-Inner hand-off.
2. **Given** a new Diagram file (`.mmd`), **When** viewed, **Then** it clearly visualizes the cyclic data flow between Strategy (generation/verification) and Execution (evolution/implementation).

---

### User Story 3 - Cryptographic/Formal Verification Gate (Priority: P3)

As the **Guardian**, I want the Outer Loop to verify the Inner Loop's output against a "Spec" or rigid constraints before accepting it, so that the system remains safe and aligned with the "Constitution".

**Why this priority**: Adds the "Safety" and "Formal Judge" layer inspired by the research papers (FormalJudge, Protecting Context).

**Independent Test**: If the Inner Loop produces code that violates a defined constraint (e.g., "No API keys in code"), the Outer Loop's verification step rejects it and requests a fix.

**Acceptance Scenarios**:

1. **Given** a task with specific invariants (e.g., "Must allow user to opt-out"), **When** the Inner Loop submits an implementation, **Then** the Outer Loop runs a verification check (manual or automated).
2. **Given** a verification failure, **When** the Outer Loop feeds this back, **Then** the Inner Loop receives a "correction prompt" rather than a "success" signal.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a standard interface (e.g., `tasks.md` or JSON schema) for the Outer Loop to pass instructions to the Inner Loop.
- **FR-002**: System MUST document the "Dual-Loop Learning" workflow in `.agent/workflows/sanctuary_protocols/`.
- **FR-003**: System MUST provide a Mermaid diagram visualizing the interaction, data flow, and "gates" between the two loops.
- **FR-004**: The Inner Loop mechanism MUST be compatible with "Claude Code" (CLI) acting as the executor.
- **FR-005**: The Outer Loop MUST be able to read the Inner Loop's file system changes to validate work.
- **FR-006**: System MUST introduce a "Supervisor Skill" or similar mechanism to formalize the role of the Outer Loop in managing the session state.

### Key Entities *(include if feature involves data)*

- **Strategy Packet**: The bundle of files (Spec, Plan, Constraints) sent from Outer to Inner.
- **Execution Log**: The record of actions took by the Inner Loop (e.g., `bash` history, git commits).
- **Verification Report**: The output of the Outer Loop's review of the Inner Loop's work (Pass/Fail + Critique).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new workflow file `dual-loop-learning.md` is created and passes linter checks.
- **SC-002**: A new architecture diagram `dual_loop_architecture.mmd` is created and accurately reflects the "InternAgent" / "Self-Evolving" topology.
- **SC-003**: A "Supervisor" skill or updated specification is created that defines how Antigravity (Outer) commands Opus (Inner).
- **SC-004**: The full "Hand-off" cycle (Plan -> Code -> Review) is documented and ready for a pilot run.
