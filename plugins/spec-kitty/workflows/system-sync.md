---
description: Synchronize Antigravity configuration to Gemini CLI
---

# System Sync (Bridge Trigger)

**Purpose**: Updates the Gemini CLI configuration (`.gemini/`) to match the latest Antigravity Rules and Workflows (`.agent/`).

## 1. Execute Bridge Script
Run the synchronization script to mirror rules and generate command wrappers.

// turbo
```bash
python plugins/spec-kitty/scripts/speckit_system_bridge.py
```

## 2. Verification
Verify that a new command wrapper was generated (pick a random one).

```bash
ls -l .gemini/commands/system_sync.toml
```

## 3. Completion
Notify the user that the system is synced.
