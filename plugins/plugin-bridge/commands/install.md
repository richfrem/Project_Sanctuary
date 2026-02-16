---
description: "Install an Agent Plugin into the local environment(s)"
---

# /plugin-bridge:install

Installs a portable agent plugin into Antigravity (`.agent/`), GitHub (`.github/`), and/or Gemini (`.gemini/`).

## Usage

```bash
python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin <path-to-plugin> [--target <auto|antigravity|github|gemini>]
```

## Examples

### Install Agent Orchestrator (Auto-detect)
```bash
python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin plugins/agent-orchestrator
```

### Force Install to GitHub
```bash
python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin plugins/agent-orchestrator --target github
```
