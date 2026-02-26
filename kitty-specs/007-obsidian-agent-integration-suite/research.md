# Research & Decisions: Obsidian Agent Integration Suite

**Feature**: 007-obsidian-agent-integration-suite
**Status**: IN_PROGRESS

## 1. Executive Summary

This document synthesizes strategic findings from the Gemini 3.1 Pro deep research report ("Architecting Autonomous Intelligence," Feb 2026) regarding the integration of Obsidian as the persistent memory store for Project Sanctuary agents. The research evaluates native Obsidian functionality, the newly released Obsidian CLI, and community "Obsidian Skills" against our existing custom Agent Skills.

## 2. Key Decisions & Rationales

### Decision 1: Integration Architecture

*   **Options Considered:**
    1.  **Direct Filesystem (`pathlib` / `frontmatter`)**: Fast, requires no running Obsidian instance, but lacks native graph-resolution features.
    2.  **Local REST API (Community Plugin)**: Powerful, but introduces a dependency on a non-core plugin and requires Obsidian to be running.
    3.  **Obsidian CLI (v1.12+)**: Official tool, supports TUI and `dev:eval` (JS injection) [Source: S-021], supports Agent Client Protocols (ACP) [Source: S-020]. Requires an IPC singleton lock (Obsidian must be open) [Source: S-022].
    4.  **Multi-root Workspace Model**: Adding the `.obsidian` vault directly as an Antigravity workspace root so agents get native semantic search indexing [Source: S-018].
*   **Decision:** **Hybrid Direct Filesystem + Multi-root Workspace**.
*   **Rationale:** The Gemini research confirms that Antigravity's multi-root workspace model allows Gemini 3.1 Pro direct filesystem read/write access without additional plugins [Source S-018]. While the CLI is powerful, the IPC singleton lock requirement [Source S-022] represents a strict dependency for headless or background agent tasks. We will rely on direct markdown parsing for headless reliability, but integrate the vault as a workspace root for semantic awareness. 

### Decision 2: Capability Overlap (RLM Factory & Vector DB)

*   **Context:** WP01 explicitly mandates researching overlap between native Obsidian features and our bespoke `rlm-factory` and `vector-db` plugins.
*   **Findings from Deep Research:**
    *   Obsidian natively supports semantic retrieval via plugins like `Vector Search` (local Ollama), `EzRAG` (Gemini API), and `Vault Chat` [Sources: S-038, S-040].
    *   However, our constitution strictly enforces *tool sovereignty* and *zero trust* (Protocol 101). Relying on external Obsidian community plugins for vectorization breaks our closed loop.
*   **Decision:** **Maintain Sanctuary `vector-db` & `rlm-factory` Ownership.**
*   **Rationale:** While Obsidian *can* do semantic search, we must treat Obsidian purely as the *data layer* (the "Sovereign Memory" [Source: S-003]). Our proprietary `vector-db-agent` and `rlm-curator` will index the `.md` files directly. Obsidian will be used for the graph view, backlinking (`[[wikilinks]]`), and human interaction, while our agents retain direct programmatic control over vector ingestion and context distillation.

## 3. Evidence & Findings

| Finding ID | Summary | Source(s) | Impact |
| :--- | :--- | :--- | :--- |
| **F-002** | Antigravity multi-root workspaces support direct filesystem access to vaults. | S-018 | Validates our direct markdown/frontmatter approach for CRUD operations. |
| **F-004** | Obsidian CLI requires an active app instance (IPC singleton lock). | S-022 | Rules out the CLI for headless background CI/CD operations. |
| **F-006** | "Obsidian Mastery" skills emphasize precise Wikilink, Callout, and Embed construction. | S-026 | WP03 (CRUD Skill) and WP06 (Refactoring) must strictly format outputs as `[[wikilinks]]`, not relative paths. |
| **F-010** | Continuous "Automatic Backlinking" workflow creates 'Knowledge Notes'. | S-013 | WP05 (Forge Soul) will be critical to extract these Knowledge Notes accurately to HF. |

*(See `research/evidence-log.csv` and `research/source-register.csv` for the complete 44-source manifest.)*

## 4. Open Questions & Risks

1.  **Lock Collisions**: If the agent is editing a file via python `pathlib` (WP03) while the user has the Obsidian app open on the same file, do we risk write collisions or data corruption? 
    *   *Mitigation*: Implement safe file-locks or timestamp checks in the `obsidian-crud` plugin before writing.
2.  **Vector Sync Frequency**: When the agent creates new knowledge notes, how rapidly must the `vector-db-agent` sync the new Obsidian files to make them discoverable to other agents?
    *   *Mitigation*: Update `tasks.md` to trigger vector ingestion hooks post-note creation.

---

> **Note**: This research concludes WP01. The strategy has been defined (Direct Filesystem + Antigravity Multi-root), capability overlap has been resolved (Keep Sanctuary Vector tools, treat Obsidian as pure data layer), and the deep research has been fully integrated into the evidence logs.
