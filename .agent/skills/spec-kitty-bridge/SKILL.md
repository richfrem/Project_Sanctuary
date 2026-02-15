---
name: spec-kitty-bridge
description: "Manage the Spec Kitty Bridge to synchronize workflows, rules, and skills across all AI agents (Antigravity, Claude, Gemini, Copilot). Use when: (1) You have modified workflows or rules and need to propagate changes, (2) User asks to 'sync agents' or 'update constitution', (3) Verifying the integrity of the agent configuration files, (4) Distributing new skills to all agents."
---

# Spec Kitty Bridge

The **Spec Kitty Bridge** is the "Universal Adapter" that synchronizes the Single Source of Truth (`.windsurf` workflows and `.kittify` rules) to the native configuration formats of all supported AI agents.

## Core Operations

### 1. Universal Sync (The "Make it So" Command)
Run this command to synchronize **everything** (Workflows, Rules, Configs) to all agents.
**When to use**: After editing `spec.md`, `tasks.md`, `constitution.md`, or creating new workflows.

```bash
python3 tools/bridge/speckit_system_bridge.py
```

**What it does:**
-   Reads `.windsurf/workflows/*.md` -> Projects to `.agent/workflows`, `.claude/commands`, `.gemini/commands`, `.github/prompts`.
-   Reads `.kittify/memory/*.md` -> Projects to `.agent/rules`, `.claude/CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`.
-   Updates `.kittify/config.yaml` to register all agents.

### 2. Verify Integrity (The "Auditor")
Run this to check if the agent configurations match the Source of Truth.
**When to use**: If an agent is behaving weirdly or missing a command.

```bash
python3 tools/bridge/verify_bridge_integrity.py
```

### 3. Sync Supplemental Resources
Use these commands to sync specific resource types if you don't want a full bridge run (though full run is usually safer).

**Sync Rules**:
```bash
python3 tools/bridge/sync_rules.py --all
```

**Sync Skills**:
```bash
python3 tools/bridge/sync_skills.py --all
```

**Sync Workflows**:
```bash
python3 tools/bridge/sync_workflows.py --all
```

## Troubleshooting

### "Slash Command Missing"
If a user says "I typed /foo but it does nothing":
1.  Run `python3 tools/bridge/speckit_system_bridge.py` to regenerate the command files.
2.  **CRITICAL**: Tell the user to **RESTART THEIR IDE**. Slash commands are often loaded only at startup.

### "Agent Ignoring Rules"
1.  Check `.kittify/memory/constitution.md` to ensure the rule exists in the Source of Truth.
2.  Run `python3 tools/bridge/sync_rules.py --all`.
3.  Verify the output file for the specific agent (e.g., `GEMINI.md` or `.claude/CLAUDE.md`).

## Reference Architecture
For a deep dive into how the bridge transformations work, see:
-   [Architecture Overview](references/bridge_architecture_overview.md)
-   [Mapping Matrix](references/bridge_mapping_matrix.md)
