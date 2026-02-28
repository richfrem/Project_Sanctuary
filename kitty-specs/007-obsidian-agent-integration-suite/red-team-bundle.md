# Obsidian Integration Suite Red Team Review
**Generated:** 2026-02-27 06:30:06

A comprehensive bundle of the architecture, research, and planned work packages for the Obsidian Agent Integration Suite, submitted for Red Team architectural review.

## Index
1. `kitty-specs/007-obsidian-agent-integration-suite/red-team-prompt.md` - Primary instructions for the Red Team Reviewer.
2. `kitty-specs/007-obsidian-agent-integration-suite/spec.md` - The foundational rules and feature descriptions.
3. `kitty-specs/007-obsidian-agent-integration-suite/plan.md` - The overarching technical context and implementation plan.
4. `kitty-specs/007-obsidian-agent-integration-suite/research.md` - The synthesis of Gemini 3.1 Pro continuous deep research and integration decisions.
5. `kitty-specs/007-obsidian-agent-integration-suite/research/obsidian-plugin-architecture.md` - The specific breakdown of the 6 new Agent Skills within the proposed obsidian-integration plugin.
6. `kitty-specs/007-obsidian-agent-integration-suite/research/Obsidian Vault for AI Agents.md` - Raw output from the Gemini 3.1 Pro Deep Research analysis (Provides raw ground-truth for the Red Team).
7. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP01-research-integration-strategy.md` - Work Package 1: ADR Research.
8. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP02-analyze-kepano-skills.md` - Work Package 2: Github Repository deep dive.
9. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP03-research-hf-schema-mapping.md` - Work Package 3: HF JSONL Data Mapping rules.
10. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP04-legacy-scrubbing.md` - Work Package 4: Automated Link Refactoring.
11. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP05-build-markdown-mastery.md` - Work Package 5: Parser.
12. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP06-build-obsidian-crud.md` - Work Package 6: Vault CRUD.
13. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP07-build-dynamic-views.md` - Work Package 7: YAML Bases and JSON Canvas.
14. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP08-build-graph-traversal.md` - Work Package 8: Graph Engine.
15. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP09-build-forge-soul.md` - Work Package 9: Knowledge Exporter.
16. `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP10-integration-testing.md` - Work Package 10: Integration Testing.

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/red-team-prompt.md`
> Note: Primary instructions for the Red Team Reviewer.

````markdown
You are the Red Team Architectural Reviewer.

Your objective is to review the proposed integration strategy between Project Sanctuary's autonomous AI agents and an external Obsidian Vault. 

The user has decided to implement a "multi-root workspace" approach mapping to direct filesystem parsing, rather than utilizing the Obsidian CLI (due to IPC lock constraints) or relying on community semantic plugins (to preserve native tool sovereignty for our RLM and Vector DB skills).

Please review the following provided context bundle, which contains:
1. The Feature Specification (`spec.md`)
2. The Implementation Plan (`plan.md`)
3. The foundational research (`research.md`)
4. The generated plugin architecture (`obsidian-plugin-architecture.md`)
5. The 10 generated Work Packages (`WP01` through `WP10`)

Provide a critical 'Red Team' evaluation of this design:
1. **Security & State Integrity**: Does the choice to use direct `pathlib`/`frontmatter` libraries against live Obsidian `.md` and `.base` files risk file corruption if the user actively has the Obsidian app open?
2. **Capability Gaps**: Are the 6 defined skills (Markdown Mastery, Vault CRUD, Graph Traversal, Bases Manager, Canvas Architect, Forge Soul Exporter) sufficient to achieve "Obsidian Mastery," or are there hidden complexities in parsing Obsidian's proprietary Markdown flavor that we have missed?
3. **Execution Feasibility**: Are the Work Packages sized correctly, and are the dependencies logical?

Please provide your findings, categorized by Risk Level (Critical, High, Medium, Low), and propose concrete adjustments to the Work Packages or Architecture if necessary.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/spec.md`
> Note: The foundational rules and feature descriptions.

````markdown
# Feature Specification: Obsidian Agent Integration Suite

**Feature Branch**: `[007-obsidian-agent-integration-suite]`  
**Category**: Feature
**Created**: 2026-02-26
**Status**: Draft  
**Input**: User description: "Integrate Obsidian notes with Hugging Face's JSONL dataset format for use in learning loops. The sync process will be one component of a modular ecosystem. The feature must be broken down into formal Work Packages (WPs): (1) Research Obsidian integration methods (Local REST API vs. custom TypeScript plugin vs. direct Markdown parsing) and create an ADR. (2) Build a skill/plugin for basic note creation and reading. (3) Build a skill/plugin for graph traversal. (4) Build the 'Forge Soul' export skill to handle JSONL transformation. (5) Define a WP to analyze how data types map to our HF schema. (6) Clean up dead links and outdated MCP references."

## Purpose and Objectives (The "Why")

The core objective of the **Obsidian Agent Integration Suite** is to replace the fragile, monolithic sync script with a formalized, agentic ecosystem for knowledge management. This serves five massive architectural and strategic goals:

1. **Shift from Fragmented Ad-Hoc Storage to Automatic Linking**: We are moving away from an approach where memory is preserved across disjointed, manually created folders. Adopting a robust Obsidian vault model enables automatic bidirectional linking and gives agents a structured graph to interact with.
2. **Pre-Vault Curation and Validation**: The new ecosystem will force agents to curate, update, and validate all key content *before* it is loaded into the vault. This ensures that the vault only ever contains current, verified information with zero broken links.
3. **Leverage the Broader Ecosystem**: By adopting a formalized Obsidian vault structure, Project Sanctuary can seamlessly leverage the constant improvements, plugins, and tooling generated by the massive broader open-source Obsidian developer community.
4. **Graphical Exploration**: The integration provides new, powerful opportunities to explore the organic links between research and learning through Obsidian's native graph view and visual relationship mapping.
5. **Capability Overlap Research (WP01)**: Before writing integration code, we must aggressively research Obsidian's native capabilities. We need to understand if Obsidian's core features overlap with, replace, or sit underneath our existing `rlm-factory` (distillation, curation) and `vector-db` (semantic search) skills. The outcomes of this research may dramatically pivot our implementation plans.

While the "Soul Delivery" phase remains unchanged—we will still definitively export sealed knowledge to Hugging Face JSONL and push code to GitHub—the *method* of how agents organize, retrieve, and validate that knowledge locally is fundamentally upgrading.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Research & Define Integration Strategy (Priority: P1)

The system needs a formalized strategy for interacting with the local Obsidian vault. This goes beyond just API vs. Plugin decisions; we must aggressively research Obsidian's native capabilities to understand how they intersect with our existing `rlm-factory` and `vector-db` agent skills. If Obsidian provides native semantic search or distillation overlap, the architecture must pivot to accommodate it.

**Why this priority**: We cannot build reliable skills or assume we need to rebuild semantic search without a defined, approved interface pattern and capability map for Obsidian.
**Independent Test**: Can be fully tested by reviewing the generated ADR document and getting human steward approval on the proposed strategy and capability overlap.

**Acceptance Scenarios**:
1. **Given** the need to integrate with Obsidian, **When** the research phase is complete, **Then** an ADR is published detailing the chosen integration method (Markdown/REST/Plugin) AND an analysis of how Obsidian's capabilities affect our `rlm-factory` and `vector-db` skills.

---

### User Story 2 - Basic Note CRUD Skill (Priority: P1)

Agents require a dedicated skill to create, read, and update notes within the Obsidian vault, utilizing the strategy defined in User Story 1.

**Why this priority**: Fundamental capability required before any advanced traversal or exporting can take place.
**Independent Test**: Can be fully tested by having an agent use the skill to create a test note, read its contents, update it, and verify the changes on disk.

**Acceptance Scenarios**:
1. **Given** a valid Obsidian vault path, **When** an agent invokes the read/write skill, **Then** the note is accurately retrieved or created.

---

### User Story 3 - Knowledge Graph Traversal Skill (Priority: P2)

Agents need the ability to traverse the Obsidian knowledge graph, identifying backlinks, forward links, and relationships between concepts without relying on basic `grep` searches.

**Why this priority**: Enables advanced contextual understanding and semantic retrieval beyond basic file reading.
**Independent Test**: Can be fully tested by providing a heavily linked note and verifying the skill correctly returns the set of connected nodes.

**Acceptance Scenarios**:
1. **Given** a target note with multiple backlinks, **When** the traversal skill is invoked, **Then** a structured list of relationship connections is returned.

---

### User Story 4 - Data Mapping Analysis (Priority: P2)

Before exporting to Hugging Face, we must analyze how complex Obsidian data types (nested folders, frontmatter arrays) map to the existing JSONL schema in ADR 081. Images and binary attachments MUST be strictly ignored.

**Why this priority**: Prevents corruption of the Hugging Face dataset by explicitly defining edge-case handling for directories.
**Independent Test**: Can be fully tested by reviewing the resulting data-mapping ADR.

**Acceptance Scenarios**:
1. **Given** an Obsidian vault with deep nesting, **When** the analysis is complete, **Then** an ADR specifies exactly how the JSONL output flattens the folder hierarchy while preserving the original `source_path` as metadata.

---

### User Story 5 - "Forge Soul" JSONL Export Skill (Priority: P2)

Agents require a skill to traverse specific sealed notes in Obsidian (e.g., `status: sealed`, `#ADR`) and transform them into the Hugging Face JSONL format (`id`, `sha256`, `timestamp`, `content`, etc.) defined in ADR 081. It must strictly ignore attachments.

**Why this priority**: This is the core objective—bridging the Obsidian brain with the external open-source LLM training dataset.
**Independent Test**: Can be fully tested by running the skill contextually and verifying the output JSONL file conforms to the Hugging Face dataset schema without errors.

**Acceptance Scenarios**:
1. **Given** a set of tagged Obsidian notes, **When** the Forge Soul skill is called contextually, **Then** it performs a Git Pre-Flight check and generates a compliant JSONL payload.

---

### User Story 6 - Cleanup & Legacy Scrubbing (Priority: P3)

The repository contains broken links from moved files and outdated references to the legacy "MCP" architecture. These must be scrubbed across the repository, and all relative Markdown links in the `01_PROTOCOLS/` and `02_LEARNING/` directories must be refactored into Obsidian wikilinks.

**Why this priority**: Ensures documentation integrity and builds the foundation for the Obsidian graph view.
**Independent Test**: Can be fully tested by verifying 0 instances of "MCP" in the core prompts and confirming relative links (e.g., `[Link](../../file.md)`) have been transformed into `[[file]]`.

**Acceptance Scenarios**:
1. **Given** legacy documentation, **When** the cleanup WP is executed, **Then** MCP references are removed and Automated Link Refactoring completes successfully.


### Edge Cases

- What happens when an Obsidian file contains malformed or missing frontmatter?
- How does system handle Obsidian Vault paths containing spaces or special characters?
- What happens if the Hugging Face API is rate-limited during the JSONL export?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST formally define the Obsidian integration strategy via an Architecture Decision Record (ADR) before implementation.
- **FR-002**: System MUST provide an Agent Skill for basic note creation and retrieval.
- **FR-003**: System MUST provide an Agent Skill capable of parsing Obsidian internal links (e.g., `[[Note Title]]`) and returning a relationship graph.
- **FR-004**: System MUST strictly ignore images and binary attachments. The mapping ADR MUST solely determine how nested directory pathing maps to the JSONL `source_path` metadata. 
- **FR-005**: System MUST provide an Agent Skill ("Forge Soul") that parses matched Obsidian notes and formats them into the JSONL schema from ADR 081. This skill MUST execute a Git Pre-Flight Check (Protocol 101) verifying no uncommitted drift exists before executing the Hugging Face sync.
- **FR-006**: The "Forge Soul" sync MUST NOT be a scheduled chron job; it MUST be an Agent Skill invoked contextually by the orchestrator.
- **FR-007**: System MUST perform Automated Link Refactoring across `01_PROTOCOLS/` and `02_LEARNING/` to convert legacy relative paths (e.g., `[Link](../../file.md)`) into proper Obsidian wikilinks (`[[file]]`), alongside scrubbing "MCP" references.

### Key Entities 

- **ObsidianNode**: Represents a single markdown file in the vault, including its parsed frontmatter metadata and internal relationships.
- **SoulRecord**: The JSONL representation of a sealed node, patterned after ADR 081 (`id`, `sha256`, `timestamp`, `content`, `valence`).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: ADRs are approved for both strategy (Integration) and data mapping (Attachments).
- **SC-002**: Agents can successfully create and read Obsidian notes through the new skill without manual human intervention.
- **SC-003**: A test note with 3 backlinks and 2 forward links returns a correct 5-node relationship graph via the traversal skill.
- **SC-004**: Running the "Forge Soul" skill generates a valid JSONL payload that passes Hugging Face schema validation, and explicitly fails if uncommitted Git changes are present.
- **SC-005**: `grep -i "mcp "` run against `.agent/` and `plugins/guardian-onboarding/` returns zero results, and relative links in `01_PROTOCOLS/` and `02_LEARNING/` are successfully refactored to wikilinks.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/plan.md`
> Note: The overarching technical context and implementation plan.

````markdown
# Implementation Plan: [FEATURE]
*Path: [templates/plan-template.md](templates/plan-template.md)*


**Branch**: `007-obsidian-agent-integration-suite` | **Date**: 2026-02-26 | **Spec**: [kitty-specs/007-obsidian-agent-integration-suite/spec.md](spec.md)
**Input**: Feature specification from `/kitty-specs/007-obsidian-agent-integration-suite/spec.md`

## Summary

The Obsidian Agent Integration Suite replaces the legacy monolithic sync script with a modular Plugin-and-Skills ecosystem. 
It establishes a formalized integration strategy (ADRs for architecture and data mapping) and provides autonomous agents with explicit skills to interact with the local vault (CRUD operations, Graph traversal) and export sealed knowledge fragments to Hugging Face JSONL format (`soul_traces.jsonl`).

## Technical Context

**Language/Version**: Python 3.13 
**Primary Dependencies**: `datasets`, `huggingface_hub`, built-in JSON/Pathlib. The integration strategy is explicitly deferred to Phase 0 Research (WP01). WP01 MUST aggressively evaluate Obsidian's capabilities to determine overlap with our existing `rlm-factory` and `vector-db` plugins.   
**Storage**: Local Filesystem (Obsidian Markdown Vault) & Hugging Face Hub (JSONL)  
**Testing**: Unittest framework, local dry-run verifications.  
**Target Platform**: CLI / Agent Environment.
**Project Type**: Python Plugin ecosystem (`plugins/` directory). 
**Performance Goals**: Fast graph traversal (< 2 seconds for a graph of 50 links). Seamless linking between disparate research files.
**Constraints**: Must strictly adhere to the Hugging Face `soul_traces.jsonl` schema (ADR 081). Must handle rate limits and strictly ignore large binary attachments. Must enforce Atomic Writes and Advisory Locks when modifying active vaults to prevent silent corruption.
**Scale/Scope**: 10 Work Packages representing Research (ADRs), 6 Agent Skills + 1 Shared Parser Utility, 1 Cleanup Task, and 1 Integration Testing Suite.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Zero Trust Git**: Ensure WP09 (Forge Soul) does not push to HF without a Git clean-state check.
- [x] **Tool Discovery**: Ensure agent dependencies on `rlm-factory` and `vector-db` are explicitly audited in WP01.
- [x] **Docs First**: ADR must precede code for both the Obsidian Integration Strategy and the HF Data Mapping.

## Project Structure

### Documentation (this feature)

```
kitty-specs/[###-feature]/
├── plan.md              # This file (/spec-kitty.plan command output)
├── research.md          # Phase 0 output (/spec-kitty.plan command)
├── data-model.md        # Phase 1 output (/spec-kitty.plan command)
├── quickstart.md        # Phase 1 output (/spec-kitty.plan command)
├── contracts/           # Phase 1 output (/spec-kitty.plan command)
└── tasks.md             # Phase 2 output (/spec-kitty.tasks command - NOT created by /spec-kitty.plan)
```

### Source Code (repository root)

```
```text
plugins/obsidian-integration/
├── obsidian-parser/            # Shared proprietary markdown parser
├── skills/
│   ├── obsidian-markdown-mastery/
│   ├── obsidian-vault-crud/
│   ├── obsidian-graph-traversal/
│   ├── obsidian-bases-manager/
│   ├── obsidian-canvas-architect/
│   └── forge-soul-exporter/
└── templates/                  # Synthetic testing vaults
```

**Structure Decision**: This feature will establish a new `obsidian-integration` plugin containing the 6 autonomous skills outlined in `obsidian-plugin-architecture.md`, cleanly separating the system from legacy MCP logic. A shared `obsidian-parser` module will anchor the varied wikilink and callout syntaxes to prevent regex fragmentation across the skills.

## Work Packages

The implementation of `007-obsidian-agent-integration-suite` is structured into 10 sequential Work Packages (WPs). Note that `tasks.md` will be formally generated by the `/spec-kitty.tasks` command based on this plan.

1. **WP01: Research Obsidian Integration Strategy & Capability Overlap (ADR)**
   - **Goal:** Determine the architectural pattern for reading/writing Obsidian nodes. Map Obsidian's native capabilities against Project Sanctuary's existing `rlm-factory` and `vector-db` skills.
   - **Output:** An approved Architectural Decision Record (ADR) detailing the chosen strategy (Direct Filesystem + Antigravity Multi-root) and defining the 6 required Plugin Skills.

2. **WP02: Deep Analyze Kepano Obsidian Skills Repository**
   - **Goal:** Clone `https://github.com/kepano/obsidian-skills` into a temporary directory and perform a deep analysis of its codebase to identify best practices, potential overlaps, and standard approaches to bridging agents with Obsidian.
   - **Output:** An architectural synthesis report documenting code patterns and specific integrations that can map into our own sovereign plugin ecosystem.

3. **WP03: Research Data Mapping to HF Schema (ADR)**
   - **Goal:** Analyze nested folder and frontmatter mapping to the existing `HF_JSONL` schema (ADR 081). Ignore binary attachments.
   - **Output:** An approved ADR detailing how `source_path` metadata replaces physical directory nesting, explicitly noting unsupported edge cases.

4. **WP04: Legacy Scrubbing & Automated Link Refactoring**
   - **Goal:** Remove references to obsolete "MCP architecture". Develop a Python script (with a mandatory Dry-Run preview) to convert legacy relative markdown links into Obsidian wikilinks (`[[file]]`) across `01_PROTOCOLS/` and `02_LEARNING/`.
   - **Output:** Clean `grep` results for "MCP", and safe wikilink refactoring to establish a clean graph for downstream traversal tests.

5. **WP05: Build Obsidian Markdown Mastery Skill**
   - **Goal:** Implement the formatting controller skill (`obsidian-markdown-mastery`) and the shared `obsidian-parser` module.
   - **Output:** A self-contained utility that handles all complex link mapping (`[[Note|Alias]]`, `[[Note#^block]]`, Embeds) and acts as the gatekeeper for all vault I/O.

6. **WP06: Build Obsidian Vault CRUD Skill**
   - **Goal:** Implement the `obsidian-vault-crud` plugin skill for interacting with standard Markdown notes.
   - **Output:** A verified agent skill utilizing Atomic Writes (POSIX rename) and a `.agent-lock` protocol for human-active vaults.

7. **WP07: Build Obsidian Dynamic Views Skills (Bases & Canvas)**
   - **Goal:** Implement the `obsidian-bases-manager` and `obsidian-canvas-architect` plugin skills.
   - **Output:** Agent Skills capable of reading/updating `.base` and `.canvas` files, featuring lossless YAML parsing (`ruamel.yaml`) and forward-compatible schema logic.

8. **WP08: Build Obsidian Graph Traversal Skill**
   - **Goal:** Implement the `obsidian-graph-traversal`, anchored by the shared `obsidian-parser` utility index built in WP05.
   - **Output:** Verified traversal skill utilizing a lightweight in-memory cache index to natively resolve global wikilink ambiguities and backward link references accurately in < 2 seconds.

9. **WP09: Build 'Forge Soul' Semantic Exporter Skill**
   - **Goal:** Implement the `forge-soul-exporter` plugin to format sealed knowledge into `soul_traces.jsonl`.
   - **Output:** A verified semantic export pipeline that enforces an absolute **Snapshot Isolation Lock** (tree hash parity) during the run, failing instantly if file `mtime` changes are detected midway to avoid exporting fragmented notes.

10. **WP10: Phase 1.5 Integration & Synthetic Edge-Case Testing**
   - **Goal:** Provide end-to-end integration safety.
   - **Output:** A synthetic testing vault loaded with 100+ edge-case notes (e.g., malformed YAML, complex block transclusions, spaces in paths). Run an end-to-end simulated concurrent read/write test and a dry run of the Forge Soul export.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/research.md`
> Note: The synthesis of Gemini 3.1 Pro continuous deep research and integration decisions.

````markdown
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
| **F-006** | "Obsidian Mastery" skills emphasize precise Wikilink, Callout, and Embed construction. | S-026 | WP06 (CRUD Skill) and WP04 (Refactoring) must strictly format outputs as `[[wikilinks]]`, not relative paths. |
| **F-010** | Continuous "Automatic Backlinking" workflow creates 'Knowledge Notes'. | S-013 | WP09 (Forge Soul) will be critical to extract these Knowledge Notes accurately to HF. |

*(See `research/evidence-log.csv` and `research/source-register.csv` for the complete 44-source manifest.)*

## 4. Open Questions & Risks

1.  **Lock Collisions**: If the agent is editing a file via python `pathlib` (WP03) while the user has the Obsidian app open on the same file, do we risk write collisions or data corruption? 
    *   *Mitigation*: Implement safe file-locks or timestamp checks in the `obsidian-crud` plugin before writing.
2.  **Vector Sync Frequency**: When the agent creates new knowledge notes, how rapidly must the `vector-db-agent` sync the new Obsidian files to make them discoverable to other agents?
    *   *Mitigation*: Update `tasks.md` to trigger vector ingestion hooks post-note creation.

---

> **Note**: This research concludes WP01. The strategy has been defined (Direct Filesystem + Antigravity Multi-root), capability overlap has been resolved (Keep Sanctuary Vector tools, treat Obsidian as pure data layer), and the deep research has been fully integrated into the evidence logs.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/research/obsidian-plugin-architecture.md`
> Note: The specific breakdown of the 6 new Agent Skills within the proposed obsidian-integration plugin.

````markdown
# Obsidian Integration Suite: Architectural Blueprint

**Feature**: 007-obsidian-agent-integration-suite
**Phase**: Phase 0 - Research & Architecture
**Date**: 2026-02-26

## 1. Ecosystem Positioning
Project Sanctuary currently maintains 36 discrete plugins spanning 5 core pillars (Cognitive Continuity, Semantic Intelligence, Agent Scaffolders, SDLC Workflows, and Utilities). The introduction of Obsidian capabilities establishes a formally new **6th Pillar: Sovereign Human-Agent Memory**. 

Rather than polluting existing semantic skills (like `vector-db` or `rlm-factory`), all vault-specific logic will be encapsulated into a single, cohesive plugin bundle.

## 2. Recommended Plugin Structure
Following the Project Sanctuary plugin manifest standard, we will establish a single parent directory containing multiple granular, composable agent skills.

**Target Path:** `plugins/obsidian-integration/`

### Directory Layout
```text
plugins/obsidian-integration/
├── obsidian-parser/                    # Shared proprietary markdown parser
├── skills/
│   ├── obsidian-markdown-mastery/      # Logic formatting
│   ├── obsidian-vault-crud/            # Core I/O
│   ├── obsidian-graph-traversal/       # Relationship mapping
│   ├── obsidian-bases-manager/         # Dynamic data (.base)
│   ├── obsidian-canvas-architect/      # Visual data (.canvas)
│   └── forge-soul-exporter/            # HF Pipeline
└── templates/                          # Reusable vault mockups for testing
```

## 3. Skill Definitions & Capabilities

To achieve true "Obsidian Mastery" for our autonomous agents (Gemini, Claude), we must divide the vault interactions into specialized sub-agents, anchored by a shared parser utility.

### 3.0. `obsidian-parser`
*   **Purpose**: The central shared utility module that all downstream skills depend on.
*   **Capabilities**:
    *   Acts as the universal gatekeeper for extracting and injecting wikilinks.
    *   Distinguishes block quotes, heading aliases, and embed semantics natively without regex duplication.

### 3.1. `obsidian-markdown-mastery`
*   **Purpose**: A formatting controller skill. Ensures the agent syntax natively aligns with Obsidian's renderer.
*   **Capabilities**: 
    *   Constructing `[[Wikilinks]]` instead of relative markdown paths.
    *   Applying `> [!callout]` semantic boxes for agent warnings/tips.
    *   Constructing DRY `![[Embeds]]`.
    *   Validating YAML frontmatter against Dataview property standards.
*   **Trigger**: Automatically invoked whenever an agent is writing to a `.md` file ending up in the vault.

### 3.2. `obsidian-vault-crud`
*   **Purpose**: The core execution engine navigating the multi-root vault directly via `pathlib`.
*   **Capabilities**:
    *   Searching file contents and identifying note locations natively.
    *   Creating new notes from templates.
    *   Idempotent appending to existing chronologs (like daily notes).
    *   Reading and updating frontmatter fields without destroying content blocks.
*   **Trigger**: Invoked when an agent needs to retrieve a memory or persist a learning.

### 3.3. `obsidian-graph-traversal`
*   **Purpose**: Context awareness. Transforms static notes into semantic networks.
*   **Capabilities**:
    *   *Forward Resolution*: Extracting all outbound `[[links]]` from a node.
    *   *Backward Resolution*: Simulating the Obsidian index to find all files pointing *to* a specific node.
*   **Trigger**: Invoked during research phases when an agent needs to pull localized context surrounding a specific concept.

### 3.4. `obsidian-bases-manager`
*   **Purpose**: Dynamic Dashboard manipulation.
*   **Capabilities**:
    *   Reading and mutating `.base` YAML structures.
    *   Updating cell values allowing the agent to act as a database administrator for project trackers.
*   **Trigger**: Invoked when altering tabular data or triaging categorized task mockups inside Obsidian.

### 3.5. `obsidian-canvas-architect`
*   **Purpose**: Visual Planning.
*   **Capabilities**:
    *   Generating `.canvas` JSON arrays conforming to JSON Canvas Spec 1.0.
    *   Creating interconnected nodes (text, file, URL) with defined coordinate geometry.
*   **Trigger**: Invoked during the Planning Phase when complex architectures benefit from visual mapping over pure text logs.

### 3.6. `forge-soul-exporter`
*   **Purpose**: The Data Pipeline. 
*   **Capabilities**:
    *   Aggregates 'sealed' knowledge from the vault.
    *   Strips out attachment syntax (images/videos).
    *   Transforms markdown into unified `soul_traces.jsonl` matching Hugging Face schema ADR 081.
    *   Executes stringent Git Pre-Flight checks (Protocol 101) to block state mutation on dirty trees.
*   **Trigger**: Invoked explicitly by the user, or run as a post-session macro via `/sanctuary-persist`.

## 4. Integration Methodology

1.  **Antigravity Multi-root**: We will attach the physical Obsidian Vault directory (`.obsidian/` containing) directly to the agent's IDE workspace. 
2.  **No IPC Lock Required**: By using direct filesystem parsing (`pathlib` and `frontmatter` libraries) rather than the Obsidian CLI, these 6 skills can operate completely headless. This ensures background CI/CD validation tasks can run even if the Obsidian desktop application is closed.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/research/Obsidian Vault for AI Agents.md`
> Note: Raw output from the Gemini 3.1 Pro Deep Research analysis (Provides raw ground-truth for the Red Team).

````markdown
# **Architecting Autonomous Intelligence: The Integration of Google Antigravity and Obsidian for Persistent Agentic Knowledge Systems**

The evolution of artificial intelligence from conversational chatbots to autonomous agents represents a fundamental shift in the computational landscape. As of late 2025 and early 2026, the emergence of platforms like Google Antigravity has redefined the interface through which large language models, specifically frontier models such as Gemini 3.1 Pro and Claude 3.5 Sonnet, interact with the physical and digital world.1 A critical challenge in this agentic era is the management of persistent knowledge. While traditional models suffer from transient context windows and "context rot," the integration of local-first knowledge management systems like Obsidian provides a sovereign, durable, and interlinked memory bank.3 This report explores the technical architecture of Google Antigravity, the command-line capabilities of the Obsidian CLI, the specialized agent skills that empower models to master Obsidian’s unique formats, and the practical workflows through which agents can preserve and link their own learning to achieve long-horizon task autonomy.

## **The Architectural Paradigm of Google Antigravity**

Google Antigravity is not merely a development environment; it is an agent-first platform designed to facilitate the orchestration of multiple AI agents working across different workspaces.1 Unlike standard IDEs that embed AI as a sidebar assistant, Antigravity flips the paradigm by embedding the editor, terminal, and browser surfaces into the agent's workflow.5 This allows agents to operate at a task-oriented level, where the human developer supervises high-level plans rather than micro-managing code completions.1

### **Multi-Surface Orchestration and the Agent Manager**

The Antigravity platform introduces the Agent Manager, a dedicated "mission control" interface.5 Within this surface, developers can spawn and observe multiple agents working asynchronously.1 For instance, one agent may be assigned to refactor a backend service while another simultaneously updates the frontend UI and a third conducts browser-based testing to verify integration.2 This asynchronous pattern is essential for long-running maintenance tasks or complex bug fixes that would otherwise require constant context switching from the human user.1

The intelligence powering these agents is derived from the Gemini 3 series, with Gemini 3.1 Pro serving as the flagship reasoning model.9 The platform also supports model optionality, including Anthropic's Claude Sonnet 4.5 and OpenAI's GPT-OSS-120b, allowing developers to choose the model best suited for specific task complexities.1

| Model Tier | Primary Function within Antigravity | Key Capability |
| :---- | :---- | :---- |
| **Gemini 3.1 Pro** | Core Reasoning & Orchestration | Advanced logic, long-horizon planning, and tool use.9 |
| **Gemini 3 Flash** | Background Tasks & Summarization | Fast context processing and checkpointing.9 |
| **Gemini 2.5 Computer Use** | Browser Actuation | Direct control of Chrome for UI testing and research.8 |
| **Gemini 2.5 Flash Lite** | Semantic Search | Powers the codebase semantic search tool.9 |
| **Claude Sonnet 4.6** | Alternative Reasoning | Specialized in complex code refactors and predictable terminal ops.2 |

### **Trust through Artifacts and Verification**

A core tenet of the Antigravity platform is the establishment of trust through transparency.5 Instead of forcing users to audit thousands of raw tool calls, Antigravity agents generate "Artifacts"—tangible, human-readable deliverables that summarize the agent’s logic and progress.1 These include task lists, implementation plans, screenshots of browser-based tests, and recordings of agent actions.5 Users can provide feedback directly on these artifacts, using a Google Doc-style commenting system, which the agent incorporates in real-time without stopping its execution flow.1

This approach shifts the developer's role from writing code to reviewing "proof of work".6 The agent is expected to think through the verification of its work, not just the work itself, ensuring that every code change is backed by a successful test or a visual walkthrough.5

## **Learning as a Core Primitive: Antigravity Knowledge Items**

Antigravity distinguishes itself from predecessors by treating learning as a core primitive.1 Agents do not start every session with a blank slate; instead, they both contribute to and retrieve from a persistent knowledge base.5 This system captures what the platform refers to as "Knowledge Items" (KIs)—collections of related information on specific topics derived from past coding sessions.5

### **The Mechanics of Knowledge Extraction**

As an agent interacts with a codebase and the user, it automatically analyzes the conversation to extract significant insights, recurring patterns, and derived architecture.5 If an agent solves a particularly challenging configuration issue or develops a novel sync strategy for a local-first application, it preserves that solution as a Knowledge Item.8 Each KI contains a title, a summary, and a collection of artifacts, such as code snippets or memories of specific user instructions.13

The persistence of this knowledge base across sessions allows the agent to build a long-term memory of the codebase and the developer’s decisions.8 When a new task is initiated, the agent scans the summaries of existing KIs; if a relevant item is identified, the agent "studies" the associated artifacts to ensure that it does not repeat past mistakes or ignore previously established constraints.5

### **Scoping Knowledge: Global vs. Workspace**

Knowledge in Antigravity is organized into two primary scopes to manage relevance and privacy.14 This dual-scope architecture ensures that project-specific secrets or proprietary boilerplate remain contained while general productivity improvements are shared across the developer's entire machine.14

| Scope | Storage Location | Intended Usage |
| :---- | :---- | :---- |
| **Workspace Scope** | \<project-root\>/.agent/skills/ | Project-specific deployment scripts, database schemas, and boilerplate generation.12 |
| **Global Scope** | \~/.gemini/antigravity/skills/ | General utilities like JSON formatting, UUID generation, and code style reviewers.12 |

## **Obsidian as the Sovereign Memory for AI Agents**

While Antigravity provides internal Knowledge Items, the developer community has increasingly leveraged Obsidian as a more comprehensive, human-auditable "second brain" for AI agents.3 Obsidian’s philosophy of "file over app"—storing all data in a local folder of plain-text Markdown files—makes it uniquely suited for consumption by agents that have direct filesystem access.4

### **The Multi-Root Workspace Integration**

The most direct method for integrating Gemini 3.1 Pro with an Obsidian vault is through Antigravity’s support for multi-root workspaces.18 Because Antigravity operates like a high-powered IDE, adding the Obsidian vault folder as a root allows the agent full read/write access to the notes without requiring additional plugins.18 This setup enables the agent to act as a librarian and editor for the user's knowledge base, refining messy data dumps, organizing project notes, and linking disparate ideas while the user works on code in a separate root.18

This integration transforms Obsidian from a passive note-taking tool into a dynamic, queryable knowledge base.3 By using Antigravity’s terminal and filesystem tools, the agent can treat the vault as a structured database, leveraging standard Unix utilities to search, filter, and transform the knowledge stored within.4

### **The Agent Client Protocol (ACP) and Vault Awareness**

Beyond simple file access, the "Agent Client" plugin for Obsidian provides a bridge for agents using Zed’s Agent Client Protocol (ACP).20 This plugin allows agents like Claude Code or the Gemini CLI to run directly within the Obsidian environment, providing a "vault-aware" conversational interface.16

Key features of the Agent Client integration include:

* **Native Contextual Awareness:** Agents can reference vault notes using @notename or @\[\[note name\]\] during conversations.20  
* **Session Persistence:** Every AI conversation can be automatically exported as a Markdown note, ensuring that the reasoning behind a project’s evolution is preserved within the vault.20  
* **Auto-Mention Mode:** The agent can be configured to automatically ingest the context of the current active note, facilitating seamless iterative editing and research synthesis.20

## **Command Line Capabilities of the Obsidian CLI**

The release of the official Obsidian CLI (v1.12) marks a significant milestone for agentic automation.19 Command-line interfaces are the "natural language" of AI agents, enabling them to chain together complex operations that would be cumbersome in a GUI.19 The Obsidian CLI provides functionality equivalent to the GUI, allowing agents to manipulate the vault through a set of structured commands.19

### **CLI Operation and IPC Singleton Lock**

The Obsidian CLI operates by communicating with a running instance of the Obsidian application via an Inter-Process Communication (IPC) singleton lock.22 This means that for CLI commands to execute, Obsidian must be open on the host machine.22 For non-interactive uses like cron jobs or background scripts, agents must first ensure the application is active.22 The CLI supports a "silent" mode (now default) which allows it to perform operations without stealing focus from the user's active window.19

### **Core Command Reference for AI Agents**

Agents utilize a wide array of CLI commands to manage the knowledge lifecycle, from ingestion and search to linking and task management.21

| Category | Command | Parameters & Utility |
| :---- | :---- | :---- |
| **Vault Info** | obsidian vault | Displays vault name, path, file count, and total size.22 |
| **Navigation** | obsidian daily | Opens or reads the current day's daily note; essential for chronological logging.19 |
| **Reading** | obsidian read | Reads a note by name (with Wikilink resolution) or exact path.21 |
| **Searching** | obsidian search | Full-text search with options for context matching (--context), limit, and JSON output.22 |
| **Tasks** | obsidian tasks | Batch operations for checkboxes; can filter for todo or done states in specific files.21 |
| **Metadata** | obsidian properties | Reads or sets YAML frontmatter; allows agents to manage notes like a database.21 |
| **Structure** | obsidian backlinks | Identifies files linking to a target note, enabling graph-based reasoning.22 |
| **Automation** | obsidian create | Generates a new note from a specified template or with initial content.19 |

### **Advanced Scripting and TUI Integration**

The CLI also includes a Text User Interface (TUI) mode, which allows for fast keyboard-driven navigation, and a developer mode (dev:eval) for executing JavaScript directly within the Obsidian context.21 Agents can use these developer features to trigger complex plugin behaviors or retrieve DOM elements from the Obsidian interface.19 By combining the CLI with standard shell commands like grep, sed, and awk, agents can perform sophisticated data aggregation across the entire vault in seconds.19

## **Specialized Agent Skills for Obsidian Mastery**

In the Antigravity and Claude Code ecosystems, "Skills" are modular capability extensions that move beyond general text generation to deep vertical domain integration.26 The "Obsidian Skills" package, officially maintained and distributed via repositories like kepano/obsidian-skills, empowers agents to understand and generate content using Obsidian’s unique syntax and file formats.16

### **The SKILL.md Architecture**

A Skill is a directory-based package containing a SKILL.md file, which serves as the "brain" of the capability.14 This file defines the trigger phrases, instructions, examples, and constraints that govern the agent's behavior.14 When an agent encounters a task that matches the description in the SKILL.md frontmatter, it "activates" the skill, loading the full set of instructions into its context.26

---

**Example SKILL.md Frontmatter for Obsidian Mastery:**

YAML

name: obsidian-helper  
description: Use this skill to manage an Obsidian vault, including creating notes with Wikilinks, Callouts, and YAML properties.

### ---

**Mastery of Obsidian-Flavored Markdown**

Without specialized skills, agents often default to standard Markdown, which lacks the advanced linking and semantic features that make Obsidian powerful.26 The Obsidian Markdown skill ensures that agents correctly implement:

* **Wikilinks:** Using \[\[Note Name\]\] for bidirectional linking and \[\[Note\#Heading\]\] or \[\[Note\#^block-id\]\] for granular references.26  
* **Callouts:** Employing semantic information boxes like \> \[\!tip\], \> \[\!warning\], or \> \[\!bug\] to highlight critical insights.26  
* **Embeds:** Implementing the DRY (Don't Repeat Yourself) principle by embedding note fragments using \!\].26  
* **Properties:** Managing structured YAML metadata to facilitate queries via community plugins like Dataview.26

### **Architecting Data with Obsidian Bases**

One of the more advanced skills is the "Obsidian Bases Manager," which allows agents to interact with the .base file format.29 Obsidian Bases are YAML-based files that define dynamic, database-like views of notes within a vault.29

| Feature of Obsidian Bases | Agentic Capability | Use Case |
| :---- | :---- | :---- |
| **Multi-View Layouts** | Create tables, cards, lists, and maps.30 | Project dashboards and visual media galleries.30 |
| **Global Filters** | Apply recursive logical operators (and, or, not).30 | Automated triage of tasks or high-priority research notes.26 |
| **Custom Formulas** | Define calculated properties and logic.30 | Statistical reporting on data-heavy note collections.30 |
| **Bi-directional Sync** | Editing a cell in the Base mirrors the change in the underlying file.31 | Bulk updates of note metadata during project migrations.26 |

Through this skill, an agent like Gemini 3.1 Pro can architect complex information structures, transforming a collection of static notes into a dynamic management system for tasks, contacts, or inventory.26

### **Visual Thinking with JSON Canvas**

The JSON Canvas skill allows agents to create and edit .canvas files, which follow the open JSON Canvas Spec 1.0.32 This empowers agents to engage in visual planning, mind mapping, and flowcharting.32

Agents can programmatically:

* **Create Nodes:** Define text nodes (Markdown), file nodes (attachments), link nodes (external URLs), and group nodes (visual containers).32  
* **Connect Edges:** Draw relationships between nodes with custom labels, colors, and end shapes (e.g., arrows).32  
* **Manage Layouts:** Specify pixel-perfect coordinates (![][image1]) and dimensions (width, height) to ensure readable, aligned diagrams.32

This skill is particularly valuable for "Agentic Planning," where an agent visualizes its task plan on a canvas for human review, improving transparency and auditability.34

## **The Three-Layer Stack: Automated Research and Learning**

A practical application of this ecosystem is the "Three-Layer Stack," which connects NotebookLM, Antigravity, and Obsidian into a single, automated research pipeline.3 This workflow allows users to process unlimited sources—PDFs, articles, YouTube videos—without manual copying or lost context.3

1. **Layer 1: NotebookLM (Research Engine):** Ingests and summarizes massive amounts of source material, providing a 200K+ token context window for initial synthesis.3  
2. **Layer 2: Antigravity (Automation Bridge):** An agent in Antigravity uses an MCP (Model Context Protocol) server to programmatically query the notebooks in NotebookLM. It executes custom AI skills to extract key findings and define research workflows.3  
3. **Layer 3: Obsidian (Knowledge Canvas):** The extracted findings flow directly into Obsidian. The agent uses its specialized skills to transform these results into interconnected permanent notes, complete with Wikilinks, tags, and YAML metadata.3

This compounding research archive ensures that knowledge doesn't disappear into transient chat logs but instead contributes to a living, queryable knowledge base that becomes more valuable with every project.3

## **Gemini 3.1 Pro: Reasoning over the Knowledge Graph**

The capabilities of Gemini 3.1 Pro are central to the effectiveness of the Antigravity-Obsidian integration.10 Built for tasks requiring advanced reasoning and long-horizon planning, Gemini 3.1 Pro is designed to synthesize dense research into functional output.10

### **Long-Horizon Task Management**

In Antigravity, Gemini 3.1 Pro acts as an "autonomous actor" capable of navigating complex engineering tasks with minimal intervention.10 For example, when asked to perform a database migration, the model does not simply write a script; it generates a structured implementation plan, assesses risks, architects a local-first sync engine, and generates unit tests for the matching logic.10

The model's ability to "think first" is facilitated by the Planning Mode in Antigravity.12 Before touching any files, the agent reasons about the project structure and determines which stack-specific skills (e.g., PostgreSQL, Tailwind, Drizzle) should be activated.35 This planning-first approach addresses the "runaway changes" problem common in earlier AI coding assistants, where models would proceed with edits without a cohesive architectural strategy.2

### **Semantic Search and Knowledge Retrieval**

While the Obsidian CLI provides keyword-based search, agents often require semantic retrieval to find relevant context in large vaults.38 Plugins like "Vector Search" and "EzRAG" provide agents with semantic search capabilities powered by embedding models (e.g., nomic-embed-text) or the Gemini File Search API.38

| Plugin | Mechanism | Backend | Advantage for Agents |
| :---- | :---- | :---- | :---- |
| **Vector Search** | Semantic Indexing | Ollama (Local) | Fast, private, finds similar meanings without keywords.38 |
| **EzRAG** | Chat-based Retrieval | Gemini API | Easy integration for Claude/Gemini agents to "chat with vault".40 |
| **MCP Advanced** | Graph Analysis | Local REST API | Maps vault structure and connections for deep context.41 |
| **Vault Chat** | Contextual RAG | OpenAI/Gemini API | High-level Q\&A over the entire knowledge base.38 |

These tools enable the agent to find hidden connections in the user's ideas, suggesting related notes or previous solutions that go beyond simple text matching.8

## **Workflow Strategies: How Agents Preserve and Link Learning**

For a Gemini or Claude agent to effectively preserve its own learning, it must follow a methodical workflow that integrates with Obsidian’s structural conventions.16 This "self-improvement" cycle allows the agent to build a persistent memory of the decisions, constraints, and patterns established during a project.5

### **The Documentation-First Workflow**

A highly effective strategy for "good" vibe coding is the documentation-first approach.2 Instead of generating code immediately, the agent is tasked with writing or extending separate documentation files within the vault.2 These files describe:

* **System Architecture:** How the various components of the application interact.2  
* **Data Models:** The structure of the information stored in the system.2  
* **Operational Details:** How things are logged, failure modes, and security policies.2

By editing these documents before code generation, the agent and the human developer establish a "ground truth" that reduces context chaos and prevents duplicate code or surprise side effects.2

### **Automatic Backlinking and Synthesis**

Agents are trained to use the Link System to build knowledge networks.26 When an agent completes a subtask, it should not only update the source code but also create or update a "Knowledge Note" in the Obsidian vault.13 This note should include:

1. **A Summary of the Task:** What was accomplished and why.13  
2. **Key Implementation Details:** Important code snippets or configuration settings.1  
3. **Wikilinks to Related Topics:** Connections to previous research notes or architectural decisions.26  
4. **YAML Metadata:** Tags and properties that allow the note to be easily discovered by future agentic searches.26

This process of "active search and connection" ensures that every new task benefits from the accumulated wisdom of the vault.44 The agent acts as its own archivist, ensuring that attribution and source tracking are maintained across long-term projects.44

## **Conclusion: The Sovereign Agentic Workspace**

The integration of Google Antigravity and Obsidian represents a significant step toward a sovereign, autonomous development environment. By providing agents with a dedicated workspace, multi-surface control, and a persistent, interlinked knowledge base, the platform enables a new level of productivity and task complexity.1 The Obsidian CLI provides the programmatic interface required for agents to master the vault, while specialized skills ensure that agents can leverage the full semantic power of Obsidian-flavored Markdown, Bases, and Canvases.19

As agents like Gemini 3.1 Pro continue to evolve, the ability to maintain a local-first, future-proof memory will be the differentiator between transient assistants and true autonomous partners.3 The Antigravity-Obsidian ecosystem ensures that the intelligence generated by these agents remains in the hands of the developer, compounding over time to form a personalized, digital "second brain" that drives innovation and efficiency in the agentic era.

#### **Works cited**

1. Build with Google Antigravity, our new agentic development platform, accessed February 26, 2026, [https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/](https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/)  
2. Antigravity: Free Gemini 3 Pro, Claude 3.5 Sonnet, and My Vibe-Coding Workflow \- AI Mind, accessed February 26, 2026, [https://pub.aimind.so/antigravity-free-gemini-3-pro-claude-3-5-sonnet-and-my-vibe-coding-workflow-6ea5a1305623](https://pub.aimind.so/antigravity-free-gemini-3-pro-claude-3-5-sonnet-and-my-vibe-coding-workflow-6ea5a1305623)  
3. I Connected NotebookLM \+ AntiGravity \+ Obsidian Into One AI Research Agent \- Reddit, accessed February 26, 2026, [https://www.reddit.com/r/startups\_promotion/comments/1qon7sj/i\_connected\_notebooklm\_antigravity\_obsidian\_into/](https://www.reddit.com/r/startups_promotion/comments/1qon7sj/i_connected_notebooklm_antigravity_obsidian_into/)  
4. How I Use AI With My Obsidian Vault Every Day: 16 Practical Use Cases, accessed February 26, 2026, [https://www.dsebastien.net/how-i-use-ai-with-my-obsidian-vault-every-day-16-practical-use-cases/](https://www.dsebastien.net/how-i-use-ai-with-my-obsidian-vault-every-day-16-practical-use-cases/)  
5. Introducing Google Antigravity, a New Era in AI-Assisted Software Development, accessed February 26, 2026, [https://antigravity.google/blog/introducing-google-antigravity](https://antigravity.google/blog/introducing-google-antigravity)  
6. Google Antigravity Explained: From Beginner to Expert Guide \- Helply, accessed February 26, 2026, [https://helply.com/blog/google-antigravity-explained](https://helply.com/blog/google-antigravity-explained)  
7. Google Antigravity Features \- AI Agents, Multi-Model Support & More, accessed February 26, 2026, [https://antigravity.im/features](https://antigravity.im/features)  
8. Google Antigravity and Gemini 3: A New Era of Agentic Development \- Medium, accessed February 26, 2026, [https://medium.com/@vfcarida/google-antigravity-and-gemini-3-a-new-era-of-agentic-development-f952ffe93b19](https://medium.com/@vfcarida/google-antigravity-and-gemini-3-a-new-era-of-agentic-development-f952ffe93b19)  
9. Models \- Google Antigravity Documentation, accessed February 26, 2026, [https://antigravity.google/docs/models](https://antigravity.google/docs/models)  
10. Gemini 3.1 Pro, Building with Advanced Intelligence in Google Antigravity, accessed February 26, 2026, [https://antigravity.google/blog/gemini-3-1-pro-in-google-antigravity](https://antigravity.google/blog/gemini-3-1-pro-in-google-antigravity)  
11. Gemini 3.1 Pro: A smarter model for your most complex tasks \- Google Blog, accessed February 26, 2026, [https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/)  
12. Getting Started with Google Antigravity, accessed February 26, 2026, [https://codelabs.developers.google.com/getting-started-google-antigravity](https://codelabs.developers.google.com/getting-started-google-antigravity)  
13. Knowledge \- Google Antigravity Documentation, accessed February 26, 2026, [https://antigravity.google/docs/knowledge](https://antigravity.google/docs/knowledge)  
14. Authoring Google Antigravity Skills, accessed February 26, 2026, [https://codelabs.developers.google.com/getting-started-with-antigravity-skills](https://codelabs.developers.google.com/getting-started-with-antigravity-skills)  
15. Tutorial : Getting Started with Google Antigravity Skills \- Medium, accessed February 26, 2026, [https://medium.com/google-cloud/tutorial-getting-started-with-antigravity-skills-864041811e0d](https://medium.com/google-cloud/tutorial-getting-started-with-antigravity-skills-864041811e0d)  
16. I put Claude Code inside Obsidian, and it was awesome \- XDA, accessed February 26, 2026, [https://www.xda-developers.com/claude-code-inside-obsidian-and-it-was-eye-opening/](https://www.xda-developers.com/claude-code-inside-obsidian-and-it-was-eye-opening/)  
17. Obsidian vs Notion (2026): Features, Graph View, Pricing & Which Is Best for You \- Pixno, accessed February 26, 2026, [https://photes.io/blog/posts/obsidian-vs-notion](https://photes.io/blog/posts/obsidian-vs-notion)  
18. Using antigravity (Gemini 3\) to read/write/manage my project vault (no plug-ins) \- Reddit, accessed February 26, 2026, [https://www.reddit.com/r/ObsidianMD/comments/1pijwcj/using\_antigravity\_gemini\_3\_to\_readwritemanage\_my/](https://www.reddit.com/r/ObsidianMD/comments/1pijwcj/using_antigravity_gemini_3_to_readwritemanage_my/)  
19. CLI is ALL you need \- DEV Community, accessed February 26, 2026, [https://dev.to/lucifer1004/cli-is-all-you-need-4n2o](https://dev.to/lucifer1004/cli-is-all-you-need-4n2o)  
20. New Plugin: Agent Client \- "Bring Claude Code, Codex & Gemini ..., accessed February 26, 2026, [https://forum.obsidian.md/t/new-plugin-agent-client-bring-claude-code-codex-gemini-cli-inside-obsidian/108448](https://forum.obsidian.md/t/new-plugin-agent-client-bring-claude-code-codex-gemini-cli-inside-obsidian/108448)  
21. The Complete Obsidian CLI Setup Guide: A Record of Overcoming Windows Pitfalls \- Zenn, accessed February 26, 2026, [https://zenn.dev/sora\_biz/articles/obsidian-cli-setup-guide?locale=en](https://zenn.dev/sora_biz/articles/obsidian-cli-setup-guide?locale=en)  
22. skills/skills/adolago/obsidian-cli/SKILL.md at main · openclaw/skills ..., accessed February 26, 2026, [https://github.com/openclaw/skills/blob/main/skills/adolago/obsidian-cli/SKILL.md](https://github.com/openclaw/skills/blob/main/skills/adolago/obsidian-cli/SKILL.md)  
23. Obsidian 1.12.2 Desktop (Early access), accessed February 26, 2026, [https://obsidian.md/changelog/2026-02-18-desktop-v1.12.2/](https://obsidian.md/changelog/2026-02-18-desktop-v1.12.2/)  
24. Changelog \- Obsidian, accessed February 26, 2026, [https://obsidian.md/changelog/](https://obsidian.md/changelog/)  
25. How are you using CLI besides AI? : r/ObsidianMD \- Reddit, accessed February 26, 2026, [https://www.reddit.com/r/ObsidianMD/comments/1r9ezpw/how\_are\_you\_using\_cli\_besides\_ai/](https://www.reddit.com/r/ObsidianMD/comments/1r9ezpw/how_are_you_using_cli_besides_ai/)  
26. Obsidian Skills — Empowering AI Agents to Master Obsidian Knowledge Management | by Addo Zhang | Feb, 2026, accessed February 26, 2026, [https://addozhang.medium.com/obsidian-skills-empowering-ai-agents-to-master-obsidian-knowledge-management-8b4f6d844b34](https://addozhang.medium.com/obsidian-skills-empowering-ai-agents-to-master-obsidian-knowledge-management-8b4f6d844b34)  
27. Master Google Antigravity Skills: Build Autonomous AI Agents | VERTU, accessed February 26, 2026, [https://vertu.com/lifestyle/mastering-google-antigravity-skills-a-comprehensive-guide-to-agentic-extensions-in-2026/](https://vertu.com/lifestyle/mastering-google-antigravity-skills-a-comprehensive-guide-to-agentic-extensions-in-2026/)  
28. Agent Skills Deep Dive: Building a Reusable Skills Ecosystem for AI Agents | by Addo Zhang, accessed February 26, 2026, [https://addozhang.medium.com/agent-skills-deep-dive-building-a-reusable-skills-ecosystem-for-ai-agents-ccb1507b2c0f](https://addozhang.medium.com/agent-skills-deep-dive-building-a-reusable-skills-ecosystem-for-ai-agents-ccb1507b2c0f)  
29. accessed February 26, 2026, [https://lobehub.com/ru/skills/davisbuilds-dojo-obsidian-bases\#:\~:text=Obsidian%20Bases%20are%20YAML%2Dbased,property%20configurations%2C%20and%20custom%20summaries.](https://lobehub.com/ru/skills/davisbuilds-dojo-obsidian-bases#:~:text=Obsidian%20Bases%20are%20YAML%2Dbased,property%20configurations%2C%20and%20custom%20summaries.)  
30. Obsidian Bases Manager \- Claude Code Skill \- MCP Market, accessed February 26, 2026, [https://mcpmarket.com/tools/skills/obsidian-bases-manager-7](https://mcpmarket.com/tools/skills/obsidian-bases-manager-7)  
31. Obsidian Bases — What are they good for (And what are they not?) | by Nick Felker, accessed February 26, 2026, [https://fleker.medium.com/obsidian-bases-what-are-they-good-for-and-what-are-they-not-da620006cb34](https://fleker.medium.com/obsidian-bases-what-are-they-good-for-and-what-are-they-not-da620006cb34)  
32. json-canvas | Skills Marketplace \- LobeHub, accessed February 26, 2026, [https://lobehub.com/skills/einverne-agent-skills-json-canvas](https://lobehub.com/skills/einverne-agent-skills-json-canvas)  
33. json-canvas | Skills Marketplace \- LobeHub, accessed February 26, 2026, [https://lobehub.com/skills/kepano-obsidian-skills-json-canvas](https://lobehub.com/skills/kepano-obsidian-skills-json-canvas)  
34. Unveiling the JSON Canvas MCP Server: A Guide for AI Engineers \- Skywork.ai, accessed February 26, 2026, [https://skywork.ai/skypage/en/json-canvas-mcp-server-ai-engineers/1978652666416635904](https://skywork.ai/skypage/en/json-canvas-mcp-server-ai-engineers/1978652666416635904)  
35. Building with Gemini 3.1 Pro: The Ultimate Coding Agent Tutorial \- DataCamp, accessed February 26, 2026, [https://www.datacamp.com/tutorial/building-with-gemini-3-1-pro-coding-agent-tutorial](https://www.datacamp.com/tutorial/building-with-gemini-3-1-pro-coding-agent-tutorial)  
36. Anti Gravity Explained: Google's Agent-First Development Platform \- Zenn, accessed February 26, 2026, [https://zenn.dev/neotechpark/articles/578723a5457e76](https://zenn.dev/neotechpark/articles/578723a5457e76)  
37. Google Antigravity vs Claude Code: Agent-First Development vs Terminal-First Control, accessed February 26, 2026, [https://www.augmentcode.com/tools/google-antigravity-vs-claude-code](https://www.augmentcode.com/tools/google-antigravity-vs-claude-code)  
38. Obsidian plugin for Vector Search, accessed February 26, 2026, [https://www.obsidianstats.com/plugins/vector-search](https://www.obsidianstats.com/plugins/vector-search)  
39. ashwin271/obsidian-vector-search: Obsidian plugin for Vector Search \- GitHub, accessed February 26, 2026, [https://github.com/ashwin271/obsidian-vector-search](https://github.com/ashwin271/obsidian-vector-search)  
40. EzRAG \- Simple Semantic Search for Obsidian using Google Gemini : r/ObsidianMD \- Reddit, accessed February 26, 2026, [https://www.reddit.com/r/ObsidianMD/comments/1ozohwo/ezrag\_simple\_semantic\_search\_for\_obsidian\_using/](https://www.reddit.com/r/ObsidianMD/comments/1ozohwo/ezrag_simple_semantic_search_for_obsidian_using/)  
41. ToKiDoO/mcp-obsidian-advanced: Advanced MCP server ... \- GitHub, accessed February 26, 2026, [https://github.com/ToKiDoO/mcp-obsidian-advanced](https://github.com/ToKiDoO/mcp-obsidian-advanced)  
42. All semantic-search Obsidian Plugins., accessed February 26, 2026, [https://www.obsidianstats.com/tags/semantic-search](https://www.obsidianstats.com/tags/semantic-search)  
43. From Requirements to Release: Automated Development of Nexus MCP Server Using OpenClaw \+ Ralph Loop | by Addo Zhang | Feb, 2026, accessed February 26, 2026, [https://addozhang.medium.com/from-requirements-to-release-automated-development-of-nexus-mcp-server-using-openclaw-ralph-loop-d6f9577d7997](https://addozhang.medium.com/from-requirements-to-release-automated-development-of-nexus-mcp-server-using-openclaw-ralph-loop-d6f9577d7997)  
44. interactive-writing-assistant | Skil... \- LobeHub, accessed February 26, 2026, [https://lobehub.com/ar/skills/jykim-claude-obsidian-skills-interactive-writing-assistant](https://lobehub.com/ar/skills/jykim-claude-obsidian-skills-interactive-writing-assistant)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAYCAYAAADtaU2/AAABUklEQVR4XmNgGAWjYBQMNzARiFOR+B1AXIPEJxbUAnEPuiAQaKMLiAPxJSg7F4h/AfF/KP8sA3ZDcIFnQKwExH+B+CCaHMhMWXQBGOCB8vWB2ALKjkCSxwcUgTgLygbpO4Uklw8VQwFGSOwyBlQFHEhsQsABSgsyQMyQRkgxvIeK4QSfGAgoIALsY8A0A8TfiCaGAkAKFqMLkghAZsDSDAgwQ8WQQ5ZBACqozICIXy0k+atIbBDgZIDEFz4AMiMdiV8KFUMBM6GCIAPPQdmgRAICoAS2AsqGAZA8CJuhiSMDkPw6KJsJysewmBFJwpUB4nMYvw5JHQwEAfFTIF6ILoEEQNkTZsZFKL0KRQWZgBeIy9EFoQDkEeScYMWACFGKAT7fogcriH0ciU82EAbiPeiCSABkERuU/QCIdyOkKAPs6AJoABQNU4B4FgMkx4yCgQUAByBJZBB3zXcAAAAASUVORK5CYII=>
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP01-research-integration-strategy.md`
> Note: Work Package 1: ADR Research.

````markdown
---
work_package_id: "WP01"
title: "Research Obsidian Integration Strategy & Capability Overlap"
lane: "planned"
dependencies: []
subtasks: ["T001", "T002", "T003", "T004", "T005"]
---

# Work Package Prompt: WP01 – Research Obsidian Integration Strategy & Capability Overlap

## Objectives & Success Criteria
- Identify the best technical path (REST API vs Custom Plugin vs Markdown) for interacting with the Vault.
- Explicitly map Obsidian's native capabilities against Project Sanctuary's existing RLM and Vector-DB plugins.
- Publish a formally approved Architecture Decision Record (ADR).

## Context & Constraints
- Constitution rules mandate Human Gates (Article I) and clear Spec/Plans (Article IV).
- See `kitty-specs/007-obsidian-agent-integration-suite/plan.md`.

## Subtasks & Detailed Guidance

### Subtask T001 – Analyze Obsidian's core mechanisms
- **Purpose**: Understand how the project will retrieve read/write capabilities locally.
- **Steps**: Investigate local REST API plugin reliability versus writing a custom TypeScript plugin versus direct Python `pathlib`/`frontmatter` scraping. Document pros and cons of each approach looking at maintenance overhead and script reliability.

### Subtask T002 – Analyze capability overlap
- **Purpose**: Preemptively avoid rebuilding features Obsidian already possesses.
- **Steps**: Compare Obsidian's native semantic features to the `rlm-factory` and `vector-db` Python skills to determine pivot opportunities. What does Obsidian offer out of the box? Are our bespoke semantic models redundant? 

### Subtask T003 – Architect Agent Skills and Plugin Boundaries
- **Purpose**: Define how Sanctuary agents will actually use Obsidian capabilities.
- **Steps**: Architect a plugin and skill structure to support "Obsidian Markdown Mastery" (Callouts, Wikilinks), "Obsidian Bases Manager" (`.base` YAML data), and "JSON Canvas Architect" (`.canvas` visual files). Establish the required host software installations (e.g. Obsidian CLI).

### Subtask T004 – Draft ADR
- **Purpose**: Document the chosen direction.
- **Steps**: Follow standard ADR templates to combine findings from T001, T002, and T003 into an Architectural Decision Record covering integration tools, required installations, and capability impact mapping.

### Subtask T005 – Obtain human steward approval
- **Purpose**: Ensure alignment with the Architect before code is written.
- **Steps**: Wait for human review or run `spec-kitty review` to get the ADR formally accepted before checking off the task.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP02-analyze-kepano-skills.md`
> Note: Work Package 2: Github Repository deep dive.

````markdown
---
work_package_id: "WP02"
title: "Deep Analyze Kepano Obsidian Skills Repository"
lane: "planned"
dependencies: ["WP01"]
subtasks: ["T006", "T007", "T008", "T009", "T010"]
---

# Work Package Prompt: WP02 – Deep Analyze Kepano Obsidian Skills Repository

## Objectives & Success Criteria
- Perform a thorough architectural analysis of the `kepano/obsidian-skills` github project.
- Find best practices and overlaps for constructing Agent capabilities over standard Markdown Vaults.
- Synthesize an intelligence payload to guide downstream implementation logic.

## Context & Constraints
- We are building a sovereign Python suite; any JS/TS plugin dependencies mapped by Kepano must be strictly evaluated against our "Direct Filesystem Read" architecture rules.

## Subtasks & Detailed Guidance

### Subtask T006 – Clone codebase to temp directory
- **Purpose**: Safely acquire the source logic over the network.
- **Steps**: Execute `git clone https://github.com/kepano/obsidian-skills /tmp/kepano-obsidian-skills`. 

### Subtask T007 – Analyze agent integration architecture
- **Purpose**: Surface how Kepano structures LLM prompts inside Obsidian.
- **Steps**: Sweep the repository for skill configurations, prompt structures, and external tool endpoints (e.g., search, fetch, list files). Look for specific JSON schemas.

### Subtask T008 – Compare with Sanctuary architecture
- **Purpose**: Map Kepano's strategies to Sanctuary's plugins.
- **Steps**: Identify what Kepano does through the Obsidian API versus what Sanctuary does natively through `ruamel.yaml` and `pathlib` Python hooks. Determine if anything can be strictly transposed.

### Subtask T009 – Compile Context Bundler payload
- **Purpose**: Ensure downstream WPs can directly read the insights from this deep dive.
- **Steps**: Create an architectural synthesis report (`research/kepano-analysis.md`). This must be added to the Red Team Bundle and loaded as context during WP05 and WP06.

### Subtask T010 – Cleanup Temporary clone
- **Purpose**: Maintain workspace hygiene.
- **Steps**: Ensure `/tmp/kepano-obsidian-skills` is purged (`rm -rf`) after the analysis report is validated and saved.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP03-research-hf-schema-mapping.md`
> Note: Work Package 3: HF JSONL Data Mapping rules.

````markdown
---
work_package_id: "WP03"
title: "Research Data Mapping to HF Schema"
lane: "planned"
dependencies: ["WP01"]
subtasks: ["T011", "T012", "T013", "T014", "T015"]
---

# Work Package Prompt: WP03 – Research Data Mapping to HF Schema

## Objectives & Success Criteria
- Resolve how highly-nested markdown vaults map to `soul_traces.jsonl`.
- Provide an approved ADR standardizing the data translation.

## Context & Constraints
- Must adhere strictly to the `HF_JSONL` schema detailed in ADR 081.

## Subtasks & Detailed Guidance

### Subtask T011 – Analyze HF `soul_traces.jsonl` schema
- **Purpose**: Standardize integration target.
- **Steps**: Pull ADR 081 into context. Note constraints around JSON fields such as `content`, `domain`, and `timestamp`.

### Subtask T012 – Define folder mapping
- **Purpose**: Translate hierarchy to tags.
- **Steps**: An Obsidian vault has infinite directory depths. Define exact mapping rules on how nested folders project down into the JSONL `source_path` array.

### Subtask T013 – Formalize Attachment Rules
- **Purpose**: Protect the semantic vector spaces from huge binaries.
- **Steps**: Create explicit code filtering rules dictating that images, `.pdf`, `.mp4`, etc., are strictly ignored by the exporter.

### Subtask T014 – Draft ADR
- **Purpose**: Document the data mapping definitions.
- **Steps**: Generate the new Architectural Decision Record spanning the rules established in T012 and T013.

### Subtask T015 – Obtain human steward approval
- **Purpose**: Gate progress.
- **Steps**: Request human review for the ADR.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP04-legacy-scrubbing.md`
> Note: Work Package 4: Automated Link Refactoring.

````markdown
---
work_package_id: "WP04"
title: "Legacy Scrubbing & Automated Link Refactoring"
lane: "planned"
dependencies: []  # Independent execution is intentional
subtasks: ["T016", "T017", "T018", "T019"]
---

# Work Package Prompt: WP04 – Legacy Scrubbing & Automated Link Refactoring

## Objectives & Success Criteria
- Ensure zero Legacy internal relative links (`[Link](../../like-this.md)`) remain in the learning databanks.
- Purge legacy term "MCP architecture" acting as noise in context prompts.

## Context & Constraints
- Link transformation must happen early to create a clean testing graph for WP08.
- Must execute safely via a Dry-Run first.

## Subtasks & Detailed Guidance

### Subtask T016 – Develop refactoring script
- **Purpose**: Safe and verifiable substitution.
- **Steps**: Write a Python script targeting the regex pattern of markdown relative links. It MUST implement a dry-run flag (`--dry-run`) printing proposed changes without executing them. Exclude text inside triple-backtick code fences.

### Subtask T017 – Apply refactoring
- **Purpose**: Actual execution.
- **Steps**: Run the script against the `01_PROTOCOLS/` and `02_LEARNING/` directories. Commit the changes cleanly.

### Subtask T018 – Scrub "MCP" references
- **Purpose**: Remove confusing vocabulary.
- **Steps**: Hunt down "MCP" and "Model Context Protocol" occurrences in `sanctuary-guardian-prompt.md` and related context boot files. Replace them with "Agent Plugin Integration".

### Subtask T019 – Verify clean grep
- **Purpose**: Prove completion.
- **Steps**: Execute exhaustive greps proving neither standard relative markdown links nor "MCP" references remain in target paths.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP05-build-markdown-mastery.md`
> Note: Work Package 5: Parser.

````markdown
---
work_package_id: "WP05"
title: "Build Obsidian Markdown Mastery Skill"
lane: "planned"
dependencies: ["WP01", "WP02"]
subtasks: ["T020", "T021", "T022", "T023", "T024"]
---

# Work Package Prompt: WP05 – Build Obsidian Markdown Mastery Skill

## Objectives & Success Criteria
- Implement the baseline parser required to format and read Obsidian distinct syntax correctly.
- Establish the `obsidian-parser` as a standalone utility module importable by other skills (like Graph Traversal).

## Context & Constraints
- Must map heading anchors, block references, wikilink aliases, and image transclusions distinctly.

## Subtasks & Detailed Guidance

### Subtask T020 – Scaffold Skill framework
- **Purpose**: Setup plugin root and shared configuration.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-markdown-mastery/SKILL.md` and the `scripts/` folder. Define a `plugins/obsidian-integration/config.yaml` or environment variable (`SANCTUARY_VAULT_PATH`) to ensure all downstream skills import from this single source.

### Subtask T021 – Build `obsidian-parser` Shared Utility
- **Purpose**: Prevent regex drift.
- **Steps**: Write a centralized Python module in `plugins/obsidian-integration/obsidian-parser/`. This is the universal gatekeeper for extracting and injecting wikilinks. Incorporate any applicable non-violating patterns from WP02 Kepano analysis into the shared `obsidian-parser`.

### Subtask T022 – Handle Edge Case Link Mappings
- **Purpose**: Account for Obsidian's proprietary capabilities identified by Red Team.
- **Steps**: Inside the parser, implement regex/AST logic specific to:
    - `[[Note#Heading]]` (Heading-level linking)
    - `[[Note#^block-id]]` (Block-level linking)
    - `[[Note|Alias Text]]` (Aliasing)
    - `![[Note]]` (Transclusion/Embed - Must be identified uniquely so the Graph Traverser doesn't parse it as a standard node).

### Subtask T023 – Callout Parsing Logic
- **Purpose**: Correct formatting of highlighted contexts.
- **Steps**: Add functions to wrap string sections in Obsidian-flavored Callouts (`> [!info]`). 

### Subtask T024 – Write Verification Tests
- **Purpose**: Prove resilience.
- **Steps**: Expose the parser to strings containing complex links and callouts. Assert that the correct types (Embed vs Link) and metadata (Alias, Heading, Block) are accurately extracted.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP06-build-obsidian-crud.md`
> Note: Work Package 6: Vault CRUD.

````markdown
---
work_package_id: "WP06"
title: "Build Obsidian Vault CRUD Skill"
lane: "planned"
dependencies: ["WP01", "WP05"]
subtasks: ["T025", "T026", "T027", "T028", "T029"]
---

# Work Package Prompt: WP06 – Build Obsidian Vault CRUD Skill

## Objectives & Success Criteria
- Implement the baseline agent capability for reading, creating, and updating standard local Obsidian `.md` notes.
- Integrate POSIX atomic renames to prevent corruption when Obsidian auto-saves concurrently.

## Context & Constraints
- Must align strictly with the implementation pattern decided upon in the WP01 ADR.
- The `obsidian-markdown-mastery` utility handles syntax; WP06 strictly handles disk I/O and locking.

## Subtasks & Detailed Guidance

### Subtask T025 – Scaffold Plugin Framework
- **Purpose**: Prepare architecture.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-vault-crud/SKILL.md` and `scripts/`.

### Subtask T026 – Implement Atomic Writes
- **Purpose**: Prevent partial disk writes from crashing active clients.
- **Steps**: Any Python write mechanism MUST write the mutated data to a hidden `.tmp` file in the same directory, then perform `os.rename()` (which is atomic on POSIX) to instantly swap the old note for the new one.

### Subtask T027 – Implement `.agent-lock` protocol
- **Purpose**: Human-Active Vault Protection.
- **Steps**: Build logic that creates a bidirectional advisory `.agent-lock` file at the root of the vault before any write batch and removes it after. This does not strictly stop Obsidian, but governs agent-vs-agent. Optionally, add process-level detection (`pgrep` or equivalent checking for `.obsidian/workspace.json` lock) as a "warm vault" warning signal, not a hard gate.

### Subtask T028 – Detect Concurrent Edits
- **Purpose**: Avoid overwriting human inputs.
- **Steps**: Capture file `mtime` before reading. Before writing the `.tmp` file back over it, check `mtime` again. If it shifted, a user edited the file mid-agent-thought. Abort and alert.

### Subtask T029 – Lossless YAML Parsing
- **Purpose**: Prevent breaking Dataview.
- **Steps**: Ensure PyYAML is NOT used. Use `ruamel.yaml` to read/write the note frontmatter perfectly preserving comments, indentation, and array styles.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP07-build-dynamic-views.md`
> Note: Work Package 7: YAML Bases and JSON Canvas.

````markdown
---
work_package_id: "WP07"
title: "Build Obsidian Dynamic Views Skills (Bases & Canvas)"
lane: "planned"
dependencies: ["WP01", "WP06"]
subtasks: ["T030", "T031", "T032", "T033", "T034"]
---

# Work Package Prompt: WP07 – Build Obsidian Dynamic Views Skills (Bases & Canvas)

## Objectives & Success Criteria
- Programmatic read/write access to DB-like `.base` views and `.canvas` flowcharts.
- Robust schema validation ensuring the agent doesn't corrupt dynamic files.

## Context & Constraints
- These structures are highly proprietary. Degrade gracefully if schemas evolve unexpectedly.

## Subtasks & Detailed Guidance

### Subtask T030 – Scaffold Plugin Frameworks
- **Purpose**: Prepare architectures.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-bases-manager/` and `plugins/obsidian-integration/skills/obsidian-canvas-architect/` alongside their corresponding `SKILL.md` documents.

### Subtask T031 – `.base` Table Manipulation
- **Purpose**: Interact with dashboard state.
- **Steps**: Write logic decoding the YAML structure native to `Obsidian Bases`. Build functions allowing an agent to confidently append or update row data while leaving view configurations untouched.

### Subtask T032 – JSON Canvas Specifications
- **Purpose**: Visual flowchart mapping.
- **Steps**: Build a client utilizing `JSON Canvas Spec 1.0`. Structure logic to programmatically place semantic `nodes` and connect them with directional `edges`.

### Subtask T033 – Graceful Error Degradation
- **Purpose**: System stability.
- **Steps**: Ensure any `KeyError` or schema mismatch caught when loading these files issues a clean API warning back to the agent rather than crashing the loop or wiping the file.

### Subtask T034 – Schema Verifications
- **Purpose**: Safety.
- **Steps**: Write verification unit tests mapping against synthetic `.base` files and mock `.canvas` boards.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP08-build-graph-traversal.md`
> Note: Work Package 8: Graph Engine.

````markdown
---
work_package_id: "WP08"
title: "Build Obsidian Graph Traversal Skill"
lane: "planned"
dependencies: ["WP05", "WP06"]
subtasks: ["T035", "T036", "T037", "T038", "T039"]
---

# Work Package Prompt: WP08 – Build Obsidian Graph Traversal Skill

## Objectives & Success Criteria
- Allow agents to query "What connects to Note X?" and receive an instant back/forward graph map.
- Performance < 2 seconds for deep queries without re-scanning the entire vault.

## Context & Constraints
- Needs to parse thousands of internal Markdown files rapidly.
- Relies on the `obsidian-parser` built in WP05 to find the links.

## Subtasks & Detailed Guidance

### Subtask T035 – Scaffold Plugin Framework
- **Purpose**: Establish root.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-graph-traversal/`. Add `SKILL.md` detailing the semantic bridging capabilities.

### Subtask T036 – Hook into `obsidian-parser`
- **Purpose**: Don't reinvent the wheel.
- **Steps**: Import the link extraction regex/AST logic from WP05. Ensure it safely filters out Image transclusions (`![[image.png]]`) while collecting note nodes.

### Subtask T037 – Build In-Memory Graph Index
- **Purpose**: Solve performance limits via cache.
- **Steps**: To avoid full-vault scans on every query, build a lightweight JSON or SQLite graph index mapping `Source -> Target`. Add an invalidation mechanism responding to file `mtime` changes.

### Subtask T038 – Forward/Backward Resolution
- **Purpose**: Graph logic endpoints.
- **Steps**: Write query logic traversing the index. E.g. `get_backlinks("Project Plan")` or `get_2nd_degree_connections("Concept A")`. 

### Subtask T039 – Synthetic Performance Testing
- **Purpose**: Verification of the 2-second target.
- **Steps**: Generate a synthetic graph of 50 interlinked test nodes. Write a python test that asserts the index generation and querying executes within boundary constraints.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP09-build-forge-soul.md`
> Note: Work Package 9: Knowledge Exporter.

````markdown
---
work_package_id: "WP09"
title: "Build 'Forge Soul' Semantic Exporter Skill"
lane: "planned"
dependencies: ["WP03", "WP05", "WP06"]
subtasks: ["T040", "T041", "T042", "T043", "T044", "T045"]
---

# Work Package Prompt: WP09 – Build 'Forge Soul' Semantic Exporter Skill

## Objectives & Success Criteria
- Format sealed Obsidian notes natively into the `soul_traces.jsonl` export schema.
- Secure the pipeline to completely prevent fragmented/corrupt exports mid-run.

## Context & Constraints
- Relies on Data Mapping logic from WP03.
- MUST implement absolute Snapshot Isolation.

## Subtasks & Detailed Guidance

### Subtask T040 – Scaffold Plugin
- **Purpose**: Architecture setup.
- **Steps**: Create `plugins/obsidian-integration/skills/forge-soul-exporter/`. Setup the `SKILL.md` and dependencies.

### Subtask T041 – Dual Git Pre-Flight Check
- **Purpose**: Guarantee clean state before export.
- **Steps**: Write logic checking `git status --porcelain`. Crucially, if the Vault is mounted outside the main plugin GitHub repository, ensure the script explicitly tests the `git status` of the **Vault Root** as well. Abort if any uncommitted changes are detected.

### Subtask T042 – Sealed Note Identification & Frontmatter Isolation
- **Purpose**: Targeting.
- **Steps**: Search for `status: sealed` in frontmatters. Crucially, wrap this in try/except blocks to gracefully warn and skip nodes exhibiting malformed YAML, avoiding pipeline crashes.

### Subtask T043 – Snapshot Isolation Parity Check
- **Purpose**: Prevent fragmented exports due to concurrent updates.
- **Steps**: At the very beginning of the export run, capture a high-speed tree hash (or record `mtime` across all parsed files). Before initiating the network push, verify the hashes/mtimes haven't moved. Abort if they did.

### Subtask T044 – Payload Formulation
- **Purpose**: Data shaping.
- **Steps**: Format output strictly to `HF_JSONL` mapping rules (WP03 ADR). Strip all binaries.

### Subtask T045 – Hugging Face API Exponential Backoff
- **Purpose**: Network resilience.
- **Steps**: Build the final upload loop using the `huggingface_hub` package. Do not rely on default retries; explicitly configure an exponential backoff retry mechanism to survive rate limits. Provide detailed terminal failure readouts.
````

---

## File: `kitty-specs/007-obsidian-agent-integration-suite/tasks/WP10-integration-testing.md`
> Note: Work Package 10: Integration Testing.

````markdown
---
work_package_id: "WP10"
title: "Phase 1.5 Integration & Synthetic Edge-Case Testing"
lane: "planned"
dependencies: ["WP04", "WP07", "WP08", "WP09"]
subtasks: ["T046", "T047", "T048", "T049", "T050"]
---

# Work Package Prompt: WP10 – Phase 1.5 Integration & Synthetic Edge-Case Testing

## Objectives & Success Criteria
- Provide end-to-end integration safety.
- Expose the assembled `obsidian-integration` plugin suite to 100+ highly varied margin-case notes to prove the parser and CRUD engines won't crash.

## Context & Constraints
- Due to the risk of silent corruption on a real vault, these integration tests MUST run against an isolated `Synthetic Vault` instantiated in `/tmp` natively through Python Pytest fixtures.

## Subtasks & Detailed Guidance

### Subtask T046 – Spin Up Synthetic Vault
- **Purpose**: A safe sandbox.
- **Steps**: Write a Pytest fixture that creates a temporary directory mirroring an Obsidian vault structure.

### Subtask T047 – Seed Edge Casing Notes
- **Purpose**: Stress test the parsing engines.
- **Steps**: Programmatically generate notes featuring:
   - Malformed YAML (unquoted colons).
   - Massive strings with thousands of sequential Wikilinks.
   - Deeply nested directories (`path/to/very/deep/folder/note.md`).
   - `.canvas` nodes connected to non-existent nodes.
   - Standard text mixed heavily with triple-backtick bash output containing `[[` strings to ensure they are excluded from linking.

### Subtask T048 – Concurrent I/O Simulation
- **Purpose**: Stress test the WP06 atomic `.agent-lock` strategy.
- **Steps**: Unleash 10 asynchronous threads attempting to simultaneously update, link, and traverse the synthetic vault notes using the `obsidian-vault-crud` and `obsidian-graph-traversal` tools. Add a background thread that simulates Obsidian desktop behavior (periodically reading and rewriting a note file mimicking auto-save) while agent threads attempt concurrent CRUD operations. Assert that mtime detection catches every simulated conflict and no silent overwrites occur.

### Subtask T049 – Dry Run Forge Soul Export
- **Purpose**: Complete the pipeline.
- **Steps**: Take the final synthetic state. Trigger the semantic export pipeline in "dry-run" mode (writing JSONL to disk rather than network) and validate against the schema.

### Subtask T050 – Analyze Code Coverage and Error Flags
- **Purpose**: Confirm safety.
- **Steps**: Collect coverage metrics. Resolve any failing tests. The plugin MUST achieve ~90% functional logic coverage before WP10 is signed off.
````

---

