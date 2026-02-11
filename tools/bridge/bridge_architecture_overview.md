# Bridge Architecture Overview

## 1. Context: Spec Kitty Framework
**InvestmentToolkit** utilizes the **Spec Kitty Framework** for systematic AI agent interaction.
-   **Upstream Source**: The `.kittify` and `.windsurf` directories (initialized by `spec-kitty`) provide the *framework* for agent coordination.
-   **Role**: Spec Kitty provides the "operating system" for agents (Workflows, Missions, Memory), while the project code resides in `tools/investment-screener` and other directories.

The **Bridge Script** (`tools/bridge/speckit_system_bridge.py`) acts as the "Universal Adapter," reading the framework's configuration and projecting it into the native formats required by specific AI tools (Antigravity, Gemini, Copilot, Claude).

## 2. Key Principles (from `AGENTS.md`)
-   **Bring Your Own Agent (BYOA)**: Any developer can use their preferred assistant (Antigravity, Gemini, Copilot, Claude) and still access the same workflows and rules.
-   **Single Source of Truth**: `.kittify/memory` (Rules) and `.windsurf/workflows` (Workflows) are the masters.
-   **One-Way Sync**: Changes flow *from* the Source of Truth *to* the agent directories. Agent directories are ephemeral build artifacts.
-   **Security**: Agent directories (`.claude/`, `.gemini/`, `.github/copilot/`) must **NEVER** be committed to Git. The bridge script respects this.
-   **Pathing**: All paths in documentation/prompts must be absolute or relative to project root.

## 3. Architecture: Universal Sync (Single-Pass)

The script operates in a straightforward **Read -> Transform -> Write** loop, replacing the previous multi-phase approach.

### A. Read Source
The script reads from the initialized Spec Kitty directories:
1.  **Workflows**: `.windsurf/workflows/*.md`
2.  **Rules**: `.kittify/memory/*.md` (e.g., `constitution.md`)

*Note: If these directories are missing, the script advises running `spec-kitty init`.*

### B. Project to Targets
The script then generates the appropriate configuration for every supported agent simultaneously:

1.  **Antigravity** (`.agent/`)
    *   **Workflows**: Copied to `.agent/workflows/`. Actor set to `antigravity`.
    *   **Rules**: Copied to `.agent/rules/`.
2.  **Claude** (`.claude/`)
    *   **Commands**: Copied to `.claude/commands/`. Actor set to `claude`.
    *   **Context**: Rules concatenated into `.claude/CLAUDE.md`.
3.  **Gemini** (`.gemini/`)
    *   **Commands**: Wrapped in TOML at `.gemini/commands/`. Actor set to `gemini`.
    *   **Context**: Rules concatenated into `GEMINI.md` (Project Root).
4.  **Copilot** (`.github/`)
    *   **Prompts**: Copied to `.github/prompts/` as `.prompt.md`. Actor set to `copilot`.
    *   **Instructions**: Rules concatenated into `.github/copilot-instructions.md`.

## 4. Automation & Workflows
This bridge logic is encapsulated in the `tools/bridge/speckit_system_bridge.py` script.
-   **Usage**: Run `python3 tools/bridge/speckit_system_bridge.py` to sync all agents.
-   **Visual**: See `tools/bridge/bridge_process.mmd` for a process diagram.
