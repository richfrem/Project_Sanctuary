---
name: plugin-bridge
description: Universal Plugin Installer. Use this skill to install or update plugins in the current agent environment. It supports Antigravity, GitHub Copilot, Gemini, and generic Claude environments.
allowed-tools: Bash, Write, Read
---

# Plugin Bridge

## Overview

The **Plugin Bridge** allows you to install plugins that follow the standard `.claude-plugin` structure into your active agent environment. It handles the transformation of workflows/commands and the copying of skills and tools.

## Usage

### Installing a Single Plugin

To install or update a specific plugin:

```bash
python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin plugins/<plugin-name>
```

**Example:**
```bash
python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin plugins/plugin-creator
```

### Installing All Plugins

To install all detected plugins in the `plugins/` directory:

```bash
python3 plugins/plugin-bridge/scripts/install_all_plugins.py
```

## When to Use

-   **After creating a new plugin**: Run the installer to make the new skills available to the agent.
-   **After modifying a plugin**: Run the installer to propagate changes to the agent's environment.
-   **On a new environment**: Run `install_all_plugins.py` to set up all available tools.

## References

- [references/usage.md](references/usage.md): Detailed usage guide and troubleshooting.
