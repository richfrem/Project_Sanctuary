---
<<<<<<< HEAD
description: Install a new plugin from the vendor collection.
args:
  plugin_name:
    description: "The name of the plugin to install (e.g., agency-swarm)."
    type: string
    required: true
---

# Install Plugin

This command installs a specific plugin from the vendor collection into your project and activates its capabilities.

```bash
PLUGIN_NAME="${plugin_name}"
VENDOR_PATH=".vendor/agent-plugins-skills/plugins/$PLUGIN_NAME"
TARGET_PATH="plugins/$PLUGIN_NAME"

# 1. Validation
if [ ! -d "$VENDOR_PATH" ]; then
    echo "Error: Plugin '$PLUGIN_NAME' not found in vendor collection."
    echo "Available plugins:"
    ls .vendor/agent-plugins-skills/plugins
    exit 1
fi

# 2. Install Code
if [ -d "$TARGET_PATH" ]; then
    echo "Plugin '$PLUGIN_NAME' is already installed. Updating..."
else
    echo "Installing '$PLUGIN_NAME'..."
    mkdir -p plugins
fi

cp -r "$VENDOR_PATH" "plugins/"

# 3. Activate (Inventory Sync)
echo "Activating plugin capabilities..."
python plugins/plugin-manager/scripts/sync_with_inventory.py
=======
description: "Install an Agent Plugin into the local environment(s)"
---

# /plugin-manager:install

Installs a portable agent plugin into Antigravity (`.agent/`), GitHub (`.github/`), and/or Gemini (`.gemini/`).

## Usage

```bash
python3 plugins/plugin-manager/scripts/bridge_installer.py --plugin <path-to-plugin> [--target <auto|antigravity|github|gemini>]
```

## Examples

### Install Agent Orchestrator (Auto-detect)
```bash
python3 plugins/plugin-manager/scripts/bridge_installer.py --plugin plugins/agent-orchestrator
```

### Force Install to GitHub
```bash
python3 plugins/plugin-manager/scripts/bridge_installer.py --plugin plugins/agent-orchestrator --target github
>>>>>>> origin/main
```
