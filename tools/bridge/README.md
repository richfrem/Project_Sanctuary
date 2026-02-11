# Spec Kitty Bridge Tools

This directory contains the "Universal Bridge" tools for synchronizing Spec Kitty definitions across multiple AI agents (Antigravity, Claude, Gemini, Copilot).

## Scripts

### 1. [`speckit_system_bridge.py`](speckit_system_bridge.py)
**The Universal Sync Engine.**
-   **Purpose**: Reads workflows (`.windsurf`) and rules (`.kittify`) and projects them into the native configuration formats for all supported agents.
-   **Usage**: `python3 tools/bridge/speckit_system_bridge.py`
-   **Operation**: Idempotent. Cleans target directories before regenerating artifacts.

### 2. [`verify_bridge_integrity.py`](verify_bridge_integrity.py)
**The Auditor.**
-   **Purpose**: Verifies that the generated agent configurations match the Source of Truth.
-   **Usage**: `python3 tools/bridge/verify_bridge_integrity.py`
-   **Checks**: Existence of files, content integrity (e.g., correct `--actor` flags, valid arguments).

### 3. [`sync_rules.py`](sync_rules.py) & [`sync_skills.py`](sync_skills.py)
**Supplemental Syncs.**
-   **Purpose**: Sync rules from `.agent/rules/` and skills from `.agent/skills/` to all agent configs.
-   **Usage**: `python3 tools/bridge/sync_rules.py --all` and `python3 tools/bridge/sync_skills.py --all`

> [!WARNING]
> **Restart Required**: After running any sync scripts, you must **restart the IDE** for slash commands to appear in your AI agent.


## Post-Setup

After running the bridge script for the first time:

1.  **Create Constitution**: Run the `/spec-kitty.constitution` workflow to establish your project's technical standards and rules.
    *   *Note*: This saves to `.kittify/memory/constitution.md`.
2.  **Re-Sync**: Run `python3 tools/bridge/speckit_system_bridge.py` again.
    *   This copies the new `constitution.md` to `.agent/rules/`, `.claude/CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md`.

## Documentation

For detailed information on how the bridge works, see:

-   **[Architecture Overview](bridge_architecture_overview.md)**: Conceptual model of the single-pass "Universal Sync" logic.
-   **[Mapping Matrix](bridge_mapping_matrix.md)**: Detailed table of file transformations from Source to Target.
-   **[Process Diagram](bridge_process.mmd)**: Visual flowchart of the bridge execution.
