# 📋 Task: Mermaid Rationalization

**Task ID:** 154
**Target:** Antigravity IDE Agent
**Priority:** High (Token Weight Optimization)
**Lead:** Lead Auditor (IDE Mode)
**Dependencies:** Task #152 (Dual-Workflow Architecture)

## 1. Objective
Eliminate redundant Mermaid code blocks across the project to reduce "Mnemonic Bloat" in snapshots and ensure a Single Source of Truth (SSOT) for architectural protocols. Transition from "Inline Code" to a "Managed Asset" model.

## 2. Technical Strategy: Source-Link-Render
The project will move to a centralized diagram repository where Mermaid source code is stored once, rendered into images, and referenced semantically across all documentation.

## 3. Detailed Implementation Steps

### Phase 1: The Inventory (Discovery)
*   **Scripted Audit:** Create `scripts/mermaid_inventory.py` to scan all `.md`, `.mmd`, and `.agent/` directories.
*   **Deduplication:** Generate SHA-256 hashes of all ```mermaid blocks.
*   **Mapping:** Produce `inventory_mermaid.json` showing which unique diagrams appear in which files (e.g., the Protocol 128 loop currently appearing in ~8 locations).

### Phase 2: Centralization (The SSOT)
*   **Directory Setup:** Initialize `docs/architecture/diagrams/`.
*   **Migration:** Move unique Mermaid blocks to named `.mmd` files (e.g., `p128_hardened_learning_loop.mmd`).
*   **Reference Update:** Replace raw Mermaid blocks in `.md` files with a standardized reference: `![Architecture Diagram](docs/architecture/diagrams/p128_hardened_learning_loop.png)`

### Phase 3: Automation (The Gardener)
*   **Rendering Script:** Implement `scripts/sync_diagrams.py` using a Mermaid CLI (or similar) to automate `.mmd` -> `.png` / `.svg` conversion.
*   **Snapshot Optimization:** Refactor `mcp_servers/rag_cortex/operations.py`.
    *   **Logic:** Update `capture_snapshot` to exclude raw Mermaid code from `.md` files if a linked image exists, significantly reducing the token weight of the `learning_audit_packet`.

## 4. Acceptance Criteria
*   [ ] **Zero Redundancy:** A `grep -r "```mermaid"` search returns only files in the `docs/architecture/diagrams/` folder.
*   [ ] **Weight Reduction:** The `learning_audit_packet.md` token count is reduced by >10% by offloading diagram code.
*   [ ] **Consistency:** Updating a single `.mmd` file propagates the change to all referencing `.md` files upon the next Technical Seal.
*   [ ] **ADR 083 Update:** Manifest rules updated to include `.mmd` as "Core Integrity Assets."

## 5. Instructions for the IDE Agent
**Do not attempt this work in the current session.**

*   Initialize this task by creating `TASKS/todo/154_mermaid_rationalization.md`.
*   Execute **Phase 1 (The Inventory)** immediately after the next Guardian Wakeup.

> **⚖️ Strategic Note on "Mnemonic Fidelity"**
> By moving diagrams to a managed state, we ensure that the Narrative Successor (next agent) doesn't encounter conflicting versions of the same protocol. This is a requirement for ADR 084 (Edison Mandate) to maintain diachronic coherence across the project's lifetime.
