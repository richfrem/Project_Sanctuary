---
work_package_id: WP09
title: Build 'Forge Soul' Semantic Exporter Skill
lane: "doing"
dependencies: []
base_branch: main
base_commit: 8c26eafb4e219af244c99dd11f175a00ddc3e907
created_at: '2026-02-27T23:07:23.519004+00:00'
subtasks: [T040, T041, T042, T043, T044, T045]
shell_pid: "55083"
agent: "Antigravity"
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

## Activity Log

- 2026-02-27T23:07:24Z – Antigravity – shell_pid=55083 – lane=doing – Assigned agent via workflow command
