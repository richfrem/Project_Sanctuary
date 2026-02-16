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

## Usage
Run the install command:
```bash
python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin plugins/my-plugin
```
