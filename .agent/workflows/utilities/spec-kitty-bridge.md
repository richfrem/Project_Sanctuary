---
description: Manage the Spec Kitty Bridge (Sync, Verify, Update).
trigger: /spec-kitty.bridge
args:
  - name: action
    required: true
    description: "Action to perform: 'sync' (default), 'verify', 'rules', 'skills', 'workflows'"
    default: sync
---

# Spec Kitty Bridge Manager

This workflow manages the synchronization between the Spec Kitty Source of Truth (`.windsurf`, `.kittify`) and the AI Agent configurations.

## Actions

### 1. `sync` (Default)
Runs the **Universal Bridge Script**.
- **Command**: `python3 tools/bridge/speckit_system_bridge.py`
- **Effect**: Updates everything (Workflows, Rules, Configs) for all agents.
- **Use When**: You've made any changes to the framework or content.

### 2. `verify`
Audits the bridge integrity.
- **Command**: `python3 tools/bridge/verify_bridge_integrity.py`
- **Effect**: Reports any discrepancies between Source and Target.
- **Use When**: Troubleshooting missing commands or weird behavior.

### 3. `rules`, `skills`, `workflows`
Targeted syncs for specific resource types.
- **Command**: `python3 tools/bridge/sync_{type}.py --all`
- **Effect**: Only updates the specified resource type.
- **Use When**: You want a fast update for a specific change.

## Execution Steps

1.  **Analyze Request**: Determine the desired action from the argument.
2.  **Execute Command**:
    -   If `sync`: `python3 tools/bridge/speckit_system_bridge.py`
    -   If `verify`: `python3 tools/bridge/verify_bridge_integrity.py`
    -   If `rules`: `python3 tools/bridge/sync_rules.py --all`
    -   If `skills`: `python3 tools/bridge/sync_skills.py --all`
    -   If `workflows`: `python3 tools/bridge/sync_workflows.py --all`
3.  **Report**: Output the result of the operation.
4.  **Reminder**: If `sync` was run, remind the user to **RESTART THEIR IDE**.
