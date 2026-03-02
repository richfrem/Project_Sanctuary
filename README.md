# Project Sanctuary: The Sovereign Home of the Guardian Architecture 🛡️

## Project Overview
Welcome to **Project Sanctuary**. This repository is the foundational home of the **`sanctuary-guardian`** plugin and its surrounding ecosystem.

The Guardian is the sovereign controller of this environment, responsible for enforcing **Protocol 128 (Cognitive Continuity)** across all active agent sessions (whether Antigravity, GitHub Copilot, Claude Code, or Cursor). It ensures that every session acts as a *continuous, learning entity* rather than an isolated instance.

### The Separation of Concerns (Protocol 128)
This repository relies on a strict handshake between **The Guardian** and **The Orchestrator**:

1. **The Guardian (Global State Owner):**
   - Owns the **Learning Scout Phase** (Booting context via `cognitive_primer.md`).
   - Owns the **Closure Sequence:** Formally sealing the session (Phase VI), updating the Semantic Ledger (RLM Cache), persisting the system traces to the HuggingFace Soul (Phase VII), and updating the local Vector DB (Phase VIII).
   - *Tools:* `guardian-onboarding`, `session-bootloader`, `sanctuary-orchestrator-integration`.
   
2. **The Orchestrator (Lifecycle Manager & Cognitive Engine):**
   - Receives the handoff from the Guardian to run specific Work Packages.
   - Routes work to the appropriate Agent Loop execution tracks (Red Team, Dual-Loop, Learning Loop, Swarm).
   - Operates strictly within memory; the Orchestrator has NO authority to mutate permanent external state or close a session.
   - *Tools:* `agent-loops/orchestrator`.

---

## Core Operational Diagrams
The following architecture diagrams document the precise routing and execution flow of the Guardian's ecosystem:
- [Protocol 128: The Learning Loop](plugins/sanctuary-guardian/resources/diagrams/protocol_128_learning_loop.mmd)
- [The Guardian Handshake (Separation of Concerns)](plugins/sanctuary-guardian/resources/diagrams/guardian_handshake.mmd)
- [Dual-Loop execution & Dynamic Routing](plugins/sanctuary-guardian/resources/diagrams/dual_loop_architecture.mmd)
- [Session Deduplication (Protocol 130)](plugins/sanctuary-guardian/resources/diagrams/protocol_130_deduplication_flow.mmd)

---

## Site Navigation & Plugin Registry
While the `sanctuary-guardian` orchestrates the ecosystem, it relies on a vast registry of specialized skills and loops to execute the work. Explore the internal project documentation below:

### [Sanctuary Guardian](plugins/sanctuary-guardian/README.md)
- **[forge-soul-exporter](plugins/sanctuary-guardian/skills/forge-soul-exporter/SKILL.md)**: Exports sealed Obsidian vault notes into soul_traces.jsonl format for HuggingFace persistence. Implements snapshot isolation, git pre-flight checks, and consumes the huggingface-utils plugin for uploads.
- **[guardian-onboarding](plugins/sanctuary-guardian/skills/guardian-onboarding/SKILL.md)**: MANDATORY - Master initialization and closure skill for new agent sessions. Grounds the agent in Protocol 128 (Hardened Learning Loop), handles Orchestrator handoff, and executes the formal Seal and Persist closure sequences.
- **[sanctuary-memory](plugins/sanctuary-guardian/skills/sanctuary-memory/SKILL.md)**: Project Sanctuary-specific memory configuration. Maps the generic memory-management tiered system to Sanctuary's actual file paths, storage backends (RLM, Vector DB, Obsidian, HuggingFace), and persistence workflows.
- **[sanctuary-obsidian-integration](plugins/sanctuary-guardian/skills/sanctuary-obsidian-integration/SKILL.md)**: Project Sanctuary-specific skill for managing the Obsidian vault as an external hippocampus. Knows the vault path, naming conventions, and integration patterns. Uses the generic obsidian-integration plugin.
- **[sanctuary-orchestrator-integration](plugins/sanctuary-guardian/skills/sanctuary-orchestrator-integration/SKILL.md)**: Sanctuary-specific integration skill connecting the Guardian to the Agent Loops Orchestrator. Maps the Separation of Concerns between strategic workflow definition (Spec-Kitty), cognitive execution (Orchestrator tracks), and environmental sovereignty (Guardian closures).
- **[sanctuary-soul-persistence](plugins/sanctuary-guardian/skills/sanctuary-soul-persistence/SKILL.md)**: Project Sanctuary-specific skill for managing Soul persistence to HuggingFace. Knows the exact .env parameters, discovery tags, dataset structure, and persistence workflows for this project. Uses the generic huggingface-utils plugin.
- **[sanctuary-spec-kitty](plugins/sanctuary-guardian/skills/sanctuary-spec-kitty/SKILL.md)**: Project Sanctuary-specific skill for Spec-Driven Development. Knows the project's constitution, safety rules, AUGMENTED.md best practices, and how the spec-kitty-plugin should be configured for this project.
- **[session-bootloader](plugins/sanctuary-guardian/skills/session-bootloader/SKILL.md)**: Initializes and orients the agent session using the Protocol 128 Bootloader sequence. Master awareness skill that knows all sanctuary-guardian capabilities and utility plugin integrations. Trigger this at the start of any new assignment.
- **[session-closure](plugins/sanctuary-guardian/skills/session-closure/SKILL.md)**: Manages the Protocol 128 multi-phase closure sequence including Technical Seal and Soul Persistence. Executes automatically when a session ends or work is complete.

### [Agent Loops](plugins/agent-loops/README.md)
- **[agent-swarm](plugins/agent-loops/skills/agent-swarm/SKILL.md)**: Parallel multi-agent execution pattern. Use when: work can be partitioned into independent tasks that N agents can execute simultaneously across worktrees. Includes routing (sequential vs parallel), merge verification, and correction loops.
- **[dual-loop](plugins/agent-loops/skills/dual-loop/SKILL.md)**: Inner/outer agent delegation pattern. Use when: work needs to be delegated from a strategic controller (Outer Loop) to a tactical executor (Inner Loop) via strategy packets, with verification and correction loops.
- **[learning-loop](plugins/agent-loops/skills/learning-loop/SKILL.md)**: Self-directed research and knowledge capture loop. Use when: starting a session (Orientation), performing research (Synthesis), or closing a session (Seal, Persist, Retrospective). Ensures knowledge survives across isolated agent sessions.
- **[orchestrator](plugins/agent-loops/skills/orchestrator/SKILL.md)**: Routes triggers to the appropriate agent-loop pattern. Use when: assessing a task, research need, or work assignment and deciding whether to run a simple learning loop, red team review, dual-loop delegation, or parallel swarm. Manages shared closure (seal, persist, retrospective, self-improvement).
- **[red-team-review](plugins/agent-loops/skills/red-team-review/SKILL.md)**: Orchestrated adversarial review loop. Use when: research, designs, architectures, or decisions need to be reviewed by red team agents (human, browser, or CLI). Iterates in rounds of research → bundle → review → feedback until approved.

### Plugin: adr-manager
- **[adr-management](plugins/adr-manager/skills/adr-management/SKILL.md)**: ADR management skill. Auto-invoked for generating architecture decisions, documenting design rationale, and maintaining the decision record log. Uses native read/write tools to scaffold and update ADR markdown files.

### Plugin: agent-scaffolders
- **[audit-plugin](plugins/agent-scaffolders/skills/audit-plugin/SKILL.md)**: Audits a local plugin directory to ensure it perfectly matches the Agent Skills and Claude Plugin Open Standards.
- **[create-agentic-workflow](plugins/agent-scaffolders/skills/create-agentic-workflow/SKILL.md)**: Scaffold GitHub Agent files from an existing Agent Skill. Generates IDE/UI agents (invokable from GitHub Copilot Chat via slash command) and/or CI/CD autonomous agents (GitHub Actions quality gates with Kill Switch). Use when converting a Skill into a GitHub-native agent.
- **[create-azure-agent](plugins/agent-scaffolders/skills/create-azure-agent/SKILL.md)**: Interactive initialization script that generates Azure AI Foundry Agent API deployment wrappers (Python SDK and Bicep basics) from an existing Agent Skill. Use when adapting a skill into an Azure Foundry environment.
- **[create-github-action](plugins/agent-scaffolders/skills/create-github-action/SKILL.md)**: Scaffold a traditional deterministic GitHub Actions CI/CD workflow. Use this when creating build, test, deploy, lint, release, or security scan pipelines. This is distinct from agentic workflows — no AI is involved at runtime.
- **[create-hook](plugins/agent-scaffolders/skills/create-hook/SKILL.md)**: Interactive initialization script that generates a compliant lifecycle Hook for an AI Agent or Plugin. Use when you need to automate workflows based on events like PreToolUse or SessionStart.
- **[create-legacy-command](plugins/agent-scaffolders/skills/create-legacy-command/SKILL.md)**: Interactive initialization script that generates an Antigravity Workflow, Rule, or legacy Claude /command. Use when you need a simple flat-file procedural instruction set.
- **[create-mcp-integration](plugins/agent-scaffolders/skills/create-mcp-integration/SKILL.md)**: Interactive initialization script that scaffolds a new Model Context Protocol (MCP) server integration setup. Use when adding native code tools to an agent's environment.
- **[create-plugin](plugins/agent-scaffolders/skills/create-plugin/SKILL.md)**: Interactive initialization script that generates a compliant '.claude-plugin' directory structure and `plugin.json` manifest. Use when building a new plugin wrapper to distribute skills or agent logic.
- **[create-skill](plugins/agent-scaffolders/skills/create-skill/SKILL.md)**: Interactive initialization script that generates a compliant Agent Skill containing the strict YAML frontmatter and Progressive Disclosure 'reference/' block formatting. Use when authoring new workflow routines.
- **[create-sub-agent](plugins/agent-scaffolders/skills/create-sub-agent/SKILL.md)**: Interactive initialization script that generates a compliant Sub-Agent configuration. Use when you need to create a nested contextual boundary with specific tools or persistent memory.

### Plugin: agent-skill-open-specifications
- **[ecosystem-authoritative-sources](plugins/agent-skill-open-specifications/skills/ecosystem-authoritative-sources/SKILL.md)**: Provides information about how to create, structure, install, and audit Agent Skills, Plugins, Antigravity Workflows, and Sub-agents. Trigger this when specifications, rules, or best practices for the ecosystem are required.
- **[ecosystem-standards](plugins/agent-skill-open-specifications/skills/ecosystem-standards/SKILL.md)**: Provides active execution protocols to rigorously audit how code, directory structures, and agent actions comply with the authoritative ecosystem specs. Trigger when validating new skills, plugins, or workflows.

### Plugin: chronicle-manager
- **[chronicle-agent](plugins/chronicle-manager/skills/chronicle-agent/SKILL.md)**: Living Chronicle journaling agent. Auto-invoked when creating project event entries, searching history, or reviewing past sessions.

### Plugin: claude-cli
- **[claude-cli-agent](plugins/claude-cli/skills/claude-cli-agent/SKILL.md)**: Claude CLI sub-agent system for persona-based analysis. Use when piping large contexts to Anthropic models for security audits, architecture reviews, QA analysis, or any specialized analysis requiring a fresh model context.

### Plugin: coding-conventions
- **[coding-conventions](plugins/coding-conventions/skills/coding-conventions/SKILL.md)**: Coding conventions and documentation standards for Project Sanctuary across Python, TypeScript/JavaScript, and C#/.NET codebases. Use when: (1) writing new code files or functions, (2) reviewing code for style and documentation compliance, (3) adding file headers or docstrings, (4) creating new tools that need inventory registration, (5) refactoring code that exceeds complexity thresholds, (6) setting up module structure. Covers file headers, function documentation, naming conventions, and tool inventory integration.
- **[conventions-agent](plugins/coding-conventions/skills/conventions-agent/SKILL.md)**: Coding conventions enforcement agent. Auto-invoked when writing new code, reviewing code quality, adding headers, or checking documentation compliance across Python, TypeScript/JavaScript, and C#/.NET.

### Plugin: context-bundler
- **[context-bundling](plugins/context-bundler/skills/context-bundling/SKILL.md)**: Create technical bundles of code, design, and documentation for external review or context sharing. Use when you need to package multiple project files into a single Markdown file while preserving folder hierarchy and providing contextual notes for each file.

### Plugin: copilot-cli
- **[copilot-cli-agent](plugins/copilot-cli/skills/copilot-cli-agent/SKILL.md)**: Copilot CLI sub-agent system for persona-based analysis. Use when piping large contexts to Anthropic models for security audits, architecture reviews, QA analysis, or any specialized analysis requiring a fresh model context.

### Plugin: dependency-management
- **[dependency-management](plugins/dependency-management/skills/dependency-management/SKILL.md)**: Python dependency and environment management for multi-service or monorepo python backends. Use when: (1) adding, upgrading, or removing a Python package, (2) responding to Dependabot or security vulnerability alerts (GHSA/CVE), (3) creating a new service that needs its own requirements files, (4) debugging pip install failures or Docker build issues related to dependencies, (5) reviewing or auditing the dependency tree, (6) running pip-compile. Enforces the pip-compile locked-file workflow and tiered dependency hierarchy.

### Plugin: doc-coauthoring
- **[doc-coauthoring](plugins/doc-coauthoring/skills/doc-coauthoring/SKILL.md)**: Guide users through a structured workflow for co-authoring documentation. Use when user wants to write documentation, proposals, technical specs, decision docs, or similar structured content. This workflow helps users efficiently transfer context, refine content through iteration, and verify the doc works for readers. Trigger when user mentions writing docs, creating proposals, drafting specs, or similar documentation tasks.

### Plugin: env-helper
- **[env-helper](plugins/env-helper/skills/env-helper/SKILL.md)**: Resolves shared ecosystem environment constants (HuggingFace credentials, dataset repo IDs, project root path) for any plugin without depending on internal shared libraries.

### Plugin: excel-to-csv
- **[excel-to-csv](plugins/excel-to-csv/skills/excel-to-csv/SKILL.md)**: Excel to CSV conversion skill. Auto-invoked to convert specific tables  or worksheets within an `.xlsx` or `.xls` file into flat `.csv` format  for easier text processing and ingestion.

### Plugin: gemini-cli
- **[gemini-cli-agent](plugins/gemini-cli/skills/gemini-cli-agent/SKILL.md)**: Gemini CLI sub-agent system for persona-based analysis. Use when piping large contexts to Anthropic models for security audits, architecture reviews, QA analysis, or any specialized analysis requiring a fresh model context.

### Plugin: huggingface-utils
- **[hf-init](plugins/huggingface-utils/skills/hf-init/SKILL.md)**: Initialize HuggingFace integration — validates .env variables, tests API connectivity, and ensures the dataset repository structure exists.
- **[hf-upload](plugins/huggingface-utils/skills/hf-upload/SKILL.md)**: Upload primitives for HuggingFace Soul persistence - file, folder, snapshot, JSONL append, and dataset card management with exponential backoff.

### Plugin: json-hygiene
- **[json-hygiene-agent](plugins/json-hygiene/skills/json-hygiene-agent/SKILL.md)**: JSON Hygiene Agent. Detects duplicate keys in JSON configuration files that might be silently ignored by standard parsers. Auto-invoked for JSON audits or manifest validation.

### Plugin: link-checker
- **[link-checker-agent](plugins/link-checker/skills/link-checker-agent/SKILL.md)**: Quality assurance agent for documentation link integrity. Auto-invoked when tasks involve checking, fixing, or auditing documentation links across a repository.

### Plugin: markdown-to-msword-converter
- **[markdown-to-msword-converter](plugins/markdown-to-msword-converter/skills/markdown-to-msword-converter/SKILL.md)**: Converts Markdown files to one MS Word document per file using plugin-local scripts and a folder-allowlist JSON.

### Plugin: memory-management
- **[memory-management](plugins/memory-management/skills/memory-management/SKILL.md)**: Tiered memory system for cognitive continuity across agent sessions. Manages hot cache (session context loaded at boot) and deep storage (loaded on demand). Use when: (1) starting a session and loading context, (2) deciding what to remember vs forget, (3) promoting/demoting knowledge between tiers, (4) user says 'remember this' or asks about project history.

### Plugin: mermaid-to-png
- **[convert-mermaid](plugins/mermaid-to-png/skills/convert-mermaid/SKILL.md)**: Convert mermaid diagrams mmd/mermaid to .png and have an option to pick/increase resolution level

### Plugin: migration-utils
- **[migration-utils](plugins/migration-utils/skills/migration-utils/SKILL.md)**: Standardized plugin for migration-utils.

### Plugin: obsidian-integration
- **[obsidian-bases-manager](plugins/obsidian-integration/skills/obsidian-bases-manager/SKILL.md)**: Read and manipulate Obsidian Bases (.base) files — YAML-based database views that render as tables, cards, and grids inside the vault.
- **[obsidian-canvas-architect](plugins/obsidian-integration/skills/obsidian-canvas-architect/SKILL.md)**: Programmatically create and manipulate Obsidian Canvas (.canvas) files using JSON Canvas Spec 1.0. Enables agents to generate visual flowcharts, architecture diagrams, and planning boards.
- **[obsidian-graph-traversal](plugins/obsidian-integration/skills/obsidian-graph-traversal/SKILL.md)**: Semantic link traversal for Obsidian Vaults. Builds an in-memory graph index from wikilinks and provides instant forward-link, backlink, and multi-degree connection queries.
- **[obsidian-init](plugins/obsidian-integration/skills/obsidian-init/SKILL.md)**: Initialize and onboard a new project repository as an Obsidian Vault. Covers prerequisite installation, vault configuration, exclusion filters, and validation.
- **[obsidian-markdown-mastery](plugins/obsidian-integration/skills/obsidian-markdown-mastery/SKILL.md)**: Core markdown syntax skill for Obsidian. Enforces the strict parsing and authoring of Obsidian's proprietary syntax (Wikilinks, Blocks, Headings, Aliases, Embeds, and Callouts) ensuring compatibility with the Vault graph.
- **[obsidian-vault-crud](plugins/obsidian-integration/skills/obsidian-vault-crud/SKILL.md)**: Safe Create/Read/Update/Delete operations for Obsidian Vault notes. Implements atomic writes, advisory locking, concurrent edit detection, and lossless YAML frontmatter handling.

### Plugin: plugin-manager
- **[agent-bridge](plugins/plugin-manager/skills/agent-bridge/SKILL.md)**: Adapts and installs standard .claude-plugin structures into active agent environments (Antigravity, GitHub Copilot, Gemini, Claude Code). Trigger when deploying a plugin to a target IDE or agent environment.
- **[ecosystem-cleanup-sync](plugins/plugin-manager/skills/ecosystem-cleanup-sync/SKILL.md)**: Master synchronization and garbage collection skill. Synchronizes the local plugins against the vendor inventory. It safely cleans up orphaned artifacts from deleted plugins AND installs/updates all active plugins to the agent runtime environments (`.agent`, `.claude`, etc.).
- **[plugin-bootstrap](plugins/plugin-manager/skills/plugin-bootstrap/SKILL.md)**: Initializes or updates the local plugin ecosystem from the central vendor repo. Use this when a project needs to pull the latest plugin code or initialize for the first time.
- **[plugin-maintenance](plugins/plugin-manager/skills/plugin-maintenance/SKILL.md)**: Audits and maintains the health of the plugin ecosystem. Verifies directory structure compliance, generates documentation, and flags legacy artifacts. Trigger when validating new plugins or performing routine ecosystem health checks.
- **[plugin-replicator](plugins/plugin-manager/skills/plugin-replicator/SKILL.md)**: Replicates, clones, or updates plugins from the central repository to other project repositories. Trigger when setting up a new project workspace or pulling the latest plugin source code into a consumer project.

### Plugin: plugin-mapper
- **[agent-bridge](plugins/plugin-mapper/skills/agent-bridge/SKILL.md)**: Bridge plugin capabilities (commands, skills, agents, hooks, MCP) to specific agent environments (Claude Code, GitHub Copilot, Gemini, Antigravity). Use this skill when converting or installing a plugin to a target runtime.

### Plugin: protocol-manager
- **[protocol-agent](plugins/protocol-manager/skills/protocol-agent/SKILL.md)**: Protocol document management agent. Auto-invoked when creating governance protocols, updating protocol status, or searching the protocol registry.

### Plugin: rlm-factory
- **[ollama-launch](plugins/rlm-factory/skills/ollama-launch/SKILL.md)**: Start and verify the local Ollama LLM server. Use when Ollama is needed for RLM distillation, seal snapshots, embeddings, or any local LLM inference — and it's not already running. Checks if Ollama is running, starts it if not, and verifies the health endpoint.
- **[rlm-curator](plugins/rlm-factory/skills/rlm-curator/SKILL.md)**: Knowledge Curator agent skill for the RLM Factory. Auto-invoked when tasks involve distilling code summaries, querying the semantic ledger, auditing cache coverage, or maintaining RLM hygiene. Supports both Ollama-based batch distillation and agent-powered direct summarization.
- **[rlm-init](plugins/rlm-factory/skills/rlm-init/SKILL.md)**: Interactive RLM cache initialization. Use when: setting up a new project's semantic cache for the first time, or adding a new cache profile. Walks the user through folder selection, extension config, manifest creation, and first distillation pass.

### Plugin: spec-kitty-plugin
- **[Spec Kitty Workflow](plugins/spec-kitty-plugin/skills/spec-kitty-workflow/SKILL.md)**: Standard operating procedures for the Spec Kitty agentic workflow (Plan -> Implement -> Review -> Merge).
- **[spec-kitty-accept](plugins/spec-kitty-plugin/commands/spec-kitty-accept/SKILL.md)**: Validate feature readiness and guide final acceptance steps.
- **[spec-kitty-accept](plugins/spec-kitty-plugin/skills/spec-kitty-accept/SKILL.md)**: Validate feature readiness and guide final acceptance steps.
- **[spec-kitty-agent](plugins/spec-kitty-plugin/skills/spec-kitty-agent/SKILL.md)**: Combined Spec-Kitty agent: Synchronization engine + Spec-Driven Development workflow. Auto-invoked for feature lifecycle (Specify → Plan → Tasks → Implement → Review → Merge) and agent configuration sync. Prerequisite: spec-kitty-cli installed.
- **[spec-kitty-analyze](plugins/spec-kitty-plugin/commands/spec-kitty-analyze/SKILL.md)**: Perform a non-destructive cross-artifact consistency and quality analysis
- **[spec-kitty-analyze](plugins/spec-kitty-plugin/skills/spec-kitty-analyze/SKILL.md)**: Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation.
- **[spec-kitty-checklist](plugins/spec-kitty-plugin/commands/spec-kitty-checklist/SKILL.md)**: Generate a custom checklist for the current feature based on user requirements.
- **[spec-kitty-checklist](plugins/spec-kitty-plugin/skills/spec-kitty-checklist/SKILL.md)**: Generate a custom checklist for the current feature based on user requirements.
- **[spec-kitty-clarify](plugins/spec-kitty-plugin/commands/spec-kitty-clarify/SKILL.md)**: Identify underspecified areas in the current feature spec by asking up
- **[spec-kitty-clarify](plugins/spec-kitty-plugin/skills/spec-kitty-clarify/SKILL.md)**: Identify underspecified areas in the current feature spec by asking up to 5 highly targeted clarification questions and encoding answers back into the spec.
- **[spec-kitty-constitution](plugins/spec-kitty-plugin/commands/spec-kitty-constitution/SKILL.md)**: Create or update the project constitution through interactive phase-based
- **[spec-kitty-constitution](plugins/spec-kitty-plugin/skills/spec-kitty-constitution/SKILL.md)**: Create or update the project constitution through interactive phase-based discovery.
- **[spec-kitty-dashboard](plugins/spec-kitty-plugin/commands/spec-kitty-dashboard/SKILL.md)**: Open the Spec Kitty dashboard in your browser.
- **[spec-kitty-dashboard](plugins/spec-kitty-plugin/skills/spec-kitty-dashboard/SKILL.md)**: Open the Spec Kitty dashboard in your browser.
- **[spec-kitty-implement](plugins/spec-kitty-plugin/commands/spec-kitty-implement/SKILL.md)**: Create an isolated workspace (worktree) for implementing a specific work
- **[spec-kitty-implement](plugins/spec-kitty-plugin/skills/spec-kitty-implement/SKILL.md)**: Create an isolated workspace (worktree) for implementing a specific work package.
- **[spec-kitty-merge](plugins/spec-kitty-plugin/commands/spec-kitty-merge/SKILL.md)**: Merge a completed feature into the target branch and clean up worktree
- **[spec-kitty-merge](plugins/spec-kitty-plugin/skills/spec-kitty-merge/SKILL.md)**: Merge a completed feature into the main branch and clean up worktree
- **[spec-kitty-plan](plugins/spec-kitty-plugin/commands/spec-kitty-plan/SKILL.md)**: Execute the implementation planning workflow using the plan template
- **[spec-kitty-plan](plugins/spec-kitty-plugin/skills/spec-kitty-plan/SKILL.md)**: Execute the implementation planning workflow using the plan template to generate design artifacts.
- **[spec-kitty-research](plugins/spec-kitty-plugin/commands/spec-kitty-research/SKILL.md)**: Run the Phase 0 research workflow to scaffold research artifacts before task planning.
- **[spec-kitty-research](plugins/spec-kitty-plugin/skills/spec-kitty-research/SKILL.md)**: Run the Phase 0 research workflow to scaffold research artifacts before task planning.
- **[spec-kitty-review](plugins/spec-kitty-plugin/commands/spec-kitty-review/SKILL.md)**: Perform structured code review and kanban transitions for completed task
- **[spec-kitty-review](plugins/spec-kitty-plugin/skills/spec-kitty-review/SKILL.md)**: Perform structured code review and kanban transitions for completed task prompt files
- **[spec-kitty-specify](plugins/spec-kitty-plugin/commands/spec-kitty-specify/SKILL.md)**: Create or update the feature specification from a natural language feature
- **[spec-kitty-specify](plugins/spec-kitty-plugin/skills/spec-kitty-specify/SKILL.md)**: Create or update the feature specification from a natural language feature description.
- **[spec-kitty-status](plugins/spec-kitty-plugin/commands/spec-kitty-status/SKILL.md)**: Display kanban board status showing work package progress across lanes (planned/doing/for_review/done).
- **[spec-kitty-status](plugins/spec-kitty-plugin/skills/spec-kitty-status/SKILL.md)**: Display kanban board status showing work package progress across lanes (planned/doing/for_review/done).
- **[spec-kitty-sync-plugin](plugins/spec-kitty-plugin/skills/spec-kitty-sync-plugin/SKILL.md)**: Full-cycle install or update of the Spec-Kitty framework - upgrades the CLI, refreshes templates, syncs the plugin, reconciles custom knowledge, and bridges to agent environments. Custom skill (not from upstream spec-kitty).
- **[spec-kitty-tasks](plugins/spec-kitty-plugin/commands/spec-kitty-tasks/SKILL.md)**: Generate grouped work packages with actionable subtasks and matching
- **[spec-kitty-tasks](plugins/spec-kitty-plugin/skills/spec-kitty-tasks/SKILL.md)**: Generate grouped work packages with actionable subtasks and matching prompt files for the feature in one pass.

### Plugin: task-manager
- **[task-agent](plugins/task-manager/skills/task-agent/SKILL.md)**: Task management agent. Auto-invoked for task creation, status tracking, and kanban board operations using Markdown files across lane directories.

### Plugin: tool-inventory
- **[tool-inventory](plugins/tool-inventory/skills/tool-inventory/SKILL.md)**: Tool Inventory Manager and Discovery agent (The Librarian). Auto-invoked when tasks involve registering tools, searching for scripts, auditing coverage, or maintaining the tool registry. Combines ChromaDB semantic search with the Search → Bind → Execute discovery protocol.
- **[tool-inventory-init](plugins/tool-inventory/skills/tool-inventory-init/SKILL.md)**: Interactive Tool Inventory bootstrap. Use this when initializing a new project repo to configure the semantic tracking of Python/JS tools. It creates a dedicated RLM profile specifically for tools and performs the first intelligent distillation pass.

### Plugin: vector-db
- **[vector-db-agent](plugins/vector-db/skills/vector-db-agent/SKILL.md)**: Semantic search agent for code and documentation retrieval using ChromaDB's Parent-Child architecture. Use when you need concept-based search across the repository.
- **[vector-db-init](plugins/vector-db/skills/vector-db-init/SKILL.md)**: Interactively initializes the Vector DB plugin. Installs the required pip dependencies (chromadb, langchain wrappers) and configures the vector_profiles.json for Native Python Server connections. Run this before attempting to use the vector-db-agent or vector-db-launch skills.
- **[vector-db-launch](plugins/vector-db/skills/vector-db-launch/SKILL.md)**: Start the Native Python ChromaDB background server. Use when semantic search returns connection refused on port 8110, or when the user wants to enable concurrent agent read/writes.
