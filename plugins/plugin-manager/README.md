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
```
