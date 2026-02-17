# Plugin Bridge Usage Guide

## Installer Script (`bridge_installer.py`)

The core script is `plugins/plugin-bridge/scripts/bridge_installer.py`.

### Arguments

-   `--plugin <path>`: **Required**. The path to the plugin directory (e.g., `plugins/my-plugin`).
-   `--target <auto|antigravity|github|gemini|claude>`: **Optional**. The target environment to install into. Defaults to `auto`.

### Auto-Detection

The installer automatically detects the environment by looking for specific directories in the project root:

-   `.agent/` -> **Antigravity**
-   `.github/` -> **GitHub Copilot**
-   `.gemini/` -> **Gemini**
-   `.claude/` -> **Claude**

## Bulk Installer (`install_all_plugins.py`)

Scans the `plugins/` directory for any folder containing a `.claude-plugin/plugin.json` manifest and installs it using `bridge_installer.py`.

## Troubleshooting

### "Plugin path not found"
Ensure you are running the command from the project root and that the path provided to `--plugin` is correct.

### "No compatible environments detected"
The installer couldn't find any of the supported configuration directories (`.agent`, `.github`, etc.). Ensure you are in the root of a supported project.

### "Validation Failed"
If the plugin is missing a `SKILL.md` or valid manifest, the installation might proceed but be incomplete. Always check the output logs.
