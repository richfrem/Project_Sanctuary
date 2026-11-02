# Sanctuary Core (`sanctuary-core`)

The consolidated APM-native core plugin for **Project Sanctuary**, providing governance, Living Chronicle journaling, and protocol lifecycle management.

## Directory Structure

```
sanctuary-core/
├── apm.yml
├── README.md
├── .claude-plugin/
│   └── plugin.json
└── .apm/
    └── skills/
        ├── managing-chronicles/
        │   ├── SKILL.md
        │   ├── scripts/chronicle_manager.py
        │   └── references/
        ├── managing-protocols/
        │   ├── SKILL.md
        │   ├── scripts/protocol_manager.py
        │   └── references/
        └── operating-sanctuary-guardian/
            ├── SKILL.md
            ├── scripts/
            │   ├── guardian_wakeup.py
            │   ├── capture_snapshot.py
            │   ├── learning_debrief.py
            │   ├── persist_soul.py
            │   └── forge_soul.py
            └── references/
```

## Features

- **`managing-chronicles`**: Comprehensive CRUD and search for the historical project journal (`chronicle/entries/`).
- **`managing-protocols`**: Formal governance document lifecycle management (`protocols/`).
- **`operating-sanctuary-guardian`**: Master Protocol 128 lifecycle controller (orientation boot, cognitive onboarding, technical seal verification, learning debrief, and soul persistence).

## Architecture

This plugin follows modern APM (Agent Package Manager) and late-2026 platform standards:
- **Zero Cross-Plugin Imports**: All resources, references, and scripts are self-contained.
- **Native Skill Tooling**: Capabilities are exposed as first-class skills without legacy `commands/` boilerplate.
- **Main-Thread Lifecycle**: Interactive governance is executed in the main conversation thread.
- **Global APM Interoperability**: Storage and execution layers cleanly interface with globally installed APM tools (`managing-obsidian-vaults`, `memory-management`, `orchestrating-agent-work`, `cli-agent`, `managing-huggingface-assets`).

## Installation

Install locally or globally via APM:

```bash
# Project installation
apm install plugins/sanctuary-core

# Global installation across targets (Claude, Antigravity, Copilot, Codex)
apm install -g --target claude,copilot,codex,antigravity plugins/sanctuary-core
```
