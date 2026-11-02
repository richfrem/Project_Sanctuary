# ADR 095: Canonical Mock Persistence for Unit Testing

## Status
Proposed (2026-01-13)

## Context
Project Sanctuary's unit test suite (Layer 1 of the Pyramid) mandates isolation from the real filesystem. However, complex logic in the Chronicle and RAG Cortex requires testing multi-step write/read operations. Pure memory mocks (like `unittest.mock`) are difficult to debug when complex state transitions fail. 

Previously, a directory named `MagicMock/` was observed at the project root, containing session-indexed numeric folders used for temporary state storage during test execution.

## Decision
We will formalize and use a dedicated `MagicMock/` directory at the project root as a **Volatile Mock Filesystem**.

1.  **Mock Persistence:** Unit tests requiring file-system-like behavior should use the `MagicMock/` directory as their root during execution.
2.  **Explicit Structure:** The directory should follow the internal naming of the service being mocked (e.g., `MagicMock/SimpleFileStore().root_path/`).
3.  **Isolation:** This directory must be strictly ignored by Git and the RLM Distillation process.
4.  **Debugging:** This directory provides "Epistemic Scars"—tangible evidence of what an agent projected onto its world during a test flight.

## Consequences

### Positive
*   **Debuggability:** Failing unit tests leave behind a "Shadow Workspace" in `MagicMock/` that humans and successor agents can inspect.
*   **Performance:** Disk-based mocking allows for testing complex state transitions that would be cumbersome to model purely in memory.
*   **Cleanliness:** Centralizes all "Magic" mocks in one location outside the `.agent/` directory.

### Negative
*   **Maintenance:** Requires ensuring the directory is correctly ignored across all tools and manifests.
*   **Artifact Bloat:** Local clones will accumulate `MagicMock/` data over time unless it is manually or automatically purged (SOP for purging to be determined).
