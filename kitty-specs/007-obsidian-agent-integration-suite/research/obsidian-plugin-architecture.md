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
