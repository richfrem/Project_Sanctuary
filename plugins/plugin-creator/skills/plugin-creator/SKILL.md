---
name: plugin-creator
description: Scaffold new Claude Code plugins. Use this skill when you need to create a new plugin project, set up the directory structure, or initialize the manifest file.
allowed-tools: Bash, Write
---

# Plugin Creator

## Overview

This skill helps you create new Claude Code plugins. It automates the setup of the required directory structure and manifests, ensuring that your new plugin follows best practices.

## Usage

### Creating a New Plugin

To create a new plugin, use the `init_plugin.py` script.

```bash
# syntax
python3 plugins/plugin-creator/skills/plugin-creator/scripts/init_plugin.py <plugin-name> --path <destination-path>

# example
python3 plugins/plugin-creator/skills/plugin-creator/scripts/init_plugin.py my-new-data-tool --path plugins/
```

This will:
1. Create a directory named `<plugin-name>` at the specified path.
2. Generate `.claude-plugin/plugin.json`.
3. Create a `skills/` directory with a sample skill.

### Next Steps

After initialization:
1.  **Edit the Manifest**: Update `.claude-plugin/plugin.json` with a description and author.
2.  **Define Skills**: Go to `skills/<plugin-name>/SKILL.md` and define your skill's logic.
3.  **Add Functionality**: Add scripts, references, or assets as needed (see `skill-creator` for guidance).

### Deployment with Plugin Bridge

After creating your plugin, use the **Plugin Bridge** to deploy it to your active agent environment.

```bash
# Deploy a single plugin
python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin plugins/<your-plugin-name>

# Deploy all plugins
python3 plugins/plugin-bridge/scripts/install_all_plugins.py
```

For more details, see [references/plugin_bridge_integration.md](references/plugin_bridge_integration.md).

## Reference

For more details on the plugin structure, see [references/plugin_architecture.md](references/plugin_architecture.md).
