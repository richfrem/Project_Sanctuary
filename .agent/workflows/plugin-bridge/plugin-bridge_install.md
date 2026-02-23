---
<<<<<<< HEAD
description: Install and bridge a plugin into agent environments.
args:
  plugin_name:
    description: "The name of the plugin to install/bridge (e.g., agency-swarm)."
    type: string
    required: true
  target_env:
    description: "The specific target environment to install for (e.g., antigravity, claude, github, gemini)."
    type: string
    required: true
---

# Install/Bridge Plugin

This command installs a plugin into your project (if not present) and bridges its capabilities to your specific agent environment. It prioritizes local plugins first. Note you MUST pass your own architecture name (e.g., `antigravity`) to the `--target_env` parameter.

```bash
PLUGIN_NAME="${plugin_name}"
TARGET_ENV="${target_env}"
LOCAL_PATH="plugins/$PLUGIN_NAME"
VENDOR_PATH=".vendor/agent-plugins-skills/plugins/$PLUGIN_NAME"

# 1. Check Local (Priority: User Created/Modified)
if [ -d "$LOCAL_PATH" ]; then
    echo "Processing local plugin: '$PLUGIN_NAME'..."

# 2. Check Vendor (Fallback: Downloadable)
elif [ -d "$VENDOR_PATH" ]; then
    echo "Plugin '$PLUGIN_NAME' found in vendor. Installing..."
    mkdir -p plugins
    cp -r "$VENDOR_PATH" "plugins/"
else
    echo "Error: Plugin '$PLUGIN_NAME' not found locally or in vendor collection."
    exit 1
fi

# 3. Bridge/Convert (The Core Action)
echo "Bridging '$PLUGIN_NAME' to environment '$TARGET_ENV'..."
python plugins/plugin-mapper/skills/agent-bridge/scripts/bridge_installer.py --plugin "plugins/$PLUGIN_NAME" --target "$TARGET_ENV"
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
