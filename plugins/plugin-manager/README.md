<<<<<<< HEAD
# Plugin Manager

**The Deployment & Synchronization Hub for Your Plugin Ecosystem**

The **Plugin Manager** is the core toolkit for maintaining a healthy plugin ecosystem. It connects your project to the central vendor repository and ensures your local environment (Copilot, Gemini, Antigravity) is always in sync.

## 🚀 Quick Start

### 1. Initial Setup
New to this repo? Run these commands to get started:
1.  **Clone Vendor Repo**: `git clone https://github.com/richfrem/agent-plugins-skills.git .vendor/agent-plugins-skills`
2.  **Bootstrap**: `python3 plugins/plugin-manager/scripts/plugin_bootstrap.py`

👉 **[Read the Full Setup Guide](../../INIT_SETUP.md)**

This is the **Master Source Repository** for agent plugins. 

To refresh your local agent environment (Antigravity, etc.) with the current plugin code:
1.  **Sync**: `python plugins/plugin-manager/scripts/sync_with_inventory.py`

## 🛠 Available Skills

| Skill | Purpose | Key Script |
| :--- | :--- | :--- |
| **[inventory-sync](skills/inventory-sync/SKILL.md)** | Deep sync between plugins folder and agent configs. | `sync_with_inventory.py` |
| **[plugin-replicator](skills/plugin-replicator/SKILL.md)** | Copies or links plugins to other repositories. | `plugin_replicator.py` |
| **[agent-bridge](skills/agent-bridge/SKILL.md)** | Adapts standard plugins to specific agent runtimes. | `bridge_installer.py` |
| **[plugin-maintenance](skills/plugin-maintenance/SKILL.md)** | Audits structure, generates READMEs, health checks. | `audit_structure.py` |

---

## core Capabilities

### 1. Inventory Sync (The Brain)
**"Keep my agents in sync with my plugins folder."**

The `sync_with_inventory.py` script is the heart of the system. It:
*   **Generates Inventory**: Creates `local-plugins-inventory.json` (your Bill of Materials).
*   **Safe Cleanup**: Identifies if you deleted a vendor plugin and removes its traces from `.agent`, `.github`, etc.
*   **Protection**: *Never* deletes your custom, project-specific plugins.

👉 **[Read the Maintenance & Cleanup Guide](../../CLEANUP.md)**

### 2. Agent Bridge (The Adapter)
**"Make my plugins work in GitHub Copilot or Gemini."**

The **Agent Bridge** adapts your standard `.claude-plugin` structure into the specific formats required by other AI agents. It is automatically run by the Sync process.

*   **GitHub Copilot**: Converts commands to `.prompt.md` files in `.github/prompts/`.
*   **Gemini**: Wraps commands in TOML for `.gemini/commands`.
*   **Antigravity**: Adapts workflows for the `.agent/workflows` structure.

### 3. Plugin Updates (The Refresher)
**"I want to get the latest code for my plugins."**

The **Update from Vendor** script handles synchronizing your existing plugins with the vendor source.
```bash
python3 plugins/plugin-manager/scripts/update_from_vendor.py
```
This is safer than manual copying as it only updates what you have installed.

---

## Directory Structure
This tool expects the following standard layout in your project root:

```
my-repo/
├── .vendor/          # Hidden source of truth (central repo)
├── .github/          # Target for Copilot prompts
├── .gemini/          # Target for Gemini commands
├── .agent/           # Target for Antigravity workflows
└── plugins/          # Your active plugins
    ├── plugin-manager/  <-- This tool
    └── ...
=======
# Plugin Bridge

**Universal Plugin Installer**

The **Plugin Bridge** allows you to write Agent Plugins *once* in the standard portable format (`.claude-plugin`, `commands/`, `skills/`) and deploy them automatically to:
- **Antigravity** (Project Sanctuary agents)
- **GitHub Copilot** (VS Code chat)
- **Gemini** (Windsurf / Codespaces)

## Features
- **Auto-Detection**: Scans your repo for `.agent`, `.github`, or `.gemini` folders.
- **Workflow Mapping**: Converts Markdown commands to `.prompt.md` (GitHub) or `.toml` (Gemini).
- **Skill Deployment**: Copies skills to the correct agent locations.
- **Resource Syncing**: Automatically deploys `resources/` (manifests, prompts) to the `tools/` mirror for path parity.

## Customization & Resources
Many plugins (like `rlm-factory`) use a `resources/` directory for configuration. If your plugin requires custom manifests or templates:
1.  **Edit Local JSONs**: Update the files in `plugins/your-plugin/resources/`.
2.  **Re-Run Bridge**: Execute the installer to synchronize these changes to the `tools/` fallback directory.
python3 plugins/plugin-manager/scripts/bridge_installer.py --plugin plugins/my-plugin
```

## Bulk Installation
To install **all detected plugins** at once (useful for new setups or full refreshes):
```bash
python3 plugins/plugin-manager/scripts/install_all_plugins.py
```

## Troubleshooting & Verification

### 1. Audit Inventory vs Filesystem
Run the audit tool to find scripts missing from `tool_inventory.json` or the RLM Cache:
```bash
python3 plugins/tool-inventory/scripts/audit_plugins.py
```

### 2. Register Missing Scripts
If the audit finds missing inventory entries, register them:
```bash
python3 plugins/tool-inventory/scripts/manage_tool_inventory.py add --path "plugins/path/to/script.py"
```

### 3. Fix "Missing from RLM Cache"
If the audit reports RLM Cache gaps, use the Distill Agent workflow to generate semantic summaries:
1. Open the workflow: `.agent/workflows/tool-inventory/tool-inventory_distill-agent.md`
2. Or run the distiller manually:
```bash
python3 plugins/tool-inventory/scripts/distiller.py --file plugins/path/to/missing_script.py
>>>>>>> origin/main
```
