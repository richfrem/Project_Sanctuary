# Config locations by tool

> Suggested filename: `mcp_servers/mcp_config_guide.md` or `mcp_servers/configuration_guide.md` —
> This document consolidates MCP-related config locations and helper scripts. Renaming may make its purpose clearer.

## relative path config file
### WINDOWS
- **File:** `%USERPROFILE%\mcp` (legacy / per-user)
### MACOS / LINUX
- **File:** `~/mcp` (legacy / per-user)

## Antigravity Config
### WINDOWS
- **File:** `%APPDATA%\Gemini\Antigravity\mcp_config.json`
- **Backup:** `%APPDATA%\Gemini\Antigravity\mcp_config.json.backup`
### MACOS / LINUX
- **File:** `~/.gemini/antigravity/mcp_config.json`
- **Backup:** `~/.gemini/antigravity/mcp_config.json.backup`

## Claude Desktop Config
### WINDOWS
- **File:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Backup:** `%APPDATA%\Claude\claude_desktop_config.json.backup`
### MACOS
- **File:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Backup:** `~/Library/Application Support/Claude/claude_desktop_config.json.backup`
## VS Code & GitHub Copilot
### WINDOWS
- **User settings:** `%APPDATA%\\Code\\User\\settings.json`
- **Workspace settings:** `<workspace>\\.vscode\\settings.json` (per-project overrides)
- **Extension storage (examples):** `%APPDATA%\\Code\\User\\globalStorage\\github.copilot\\` (extensions often store state under `globalStorage`) 

### MACOS / LINUX
- **User settings (macOS):** `~/Library/Application Support/Code/User/settings.json`
- **User settings (Linux):** `~/.config/Code/User/settings.json`
- **Workspace settings:** `<workspace>/.vscode/settings.json` (per-project overrides)
- **Extension storage (examples):** `~/.config/Code/User/globalStorage/github.copilot/` or `~/Library/Application Support/Code/User/globalStorage/github.copilot/`

### How VS Code / extensions will discover config
- **Workspace-first:** Extensions and workspace tasks commonly prefer a project-local config in the repository root (for example: `./mcp`, `./.mcp`, or a file under `.vscode/`). Place copies in the workspace to make them visible to collaborators.
- **User-scoped fallback:** If no workspace config is present, many tools/extensions fall back to user-scoped locations under `%APPDATA%` / `~/Library/Application Support/` / `~/.config/`.
- **Extension-specific storage:** Extensions may keep runtime state, caches, or tokens in their `globalStorage` folder; secrets and credentials are typically stored in the OS keychain/credential manager rather than plain files.
- **Explicit wiring:** If an extension supports a config path setting, add it to either the workspace or user `settings.json` to point to a canonical config. Example (illustrative):

```json
{
	"mcp.configPath": "${workspaceFolder}/.mcp"
}
```

This example requires the extension to support the `mcp.configPath` key; it shows the recommended pattern: prefer workspace-level settings for repository-specific configs.

## Notes and recommendations
- **User vs system scope:** Prefer user-scoped config (home directory or `%APPDATA%`) for per-user settings. Use system-wide locations (`C:\\ProgramData\\`, `/etc/`) only for defaults that require admin access.
- **Backups:** Keep a `.backup` copy next to the primary file to allow easy recovery.
- **Permissions:** Treat config files as sensitive; on Unix-like systems set directories to `700` and files to `600`. Use OS keychains/credential stores for secrets when available.
- **Directory creation (example, macOS/Linux):**

```bash
mkdir -p ~/.gemini/antigravity && chmod 700 ~/.gemini/antigravity
touch ~/.gemini/antigravity/mcp_config.json && chmod 600 ~/.gemini/antigravity/mcp_config.json
```

If you want PowerShell examples for Windows, additional extension-specific paths, or sample JSON schemas added, tell me and I'll expand this file accordingly.

- **Template file (env placeholders):** `.agent/mcp_config.json` — contains `${PYTHON_EXEC}` and `${PROJECT_SANCTUARY_ROOT}` placeholders for portable configs.

## How to set the environment variables required for the relative path configurations

### Helper launch scripts (launch with the project every time)
- **Windows (PowerShell):** `mcp_servers/start_sanctuary.ps1` — auto-sets `PROJECT_SANCTUARY_ROOT` and attempts to detect a `.venv` Python, then sets `PYTHON_EXEC`. Run from PowerShell in the repo root:

```powershell
.\mcp_servers\start_sanctuary.ps1
```

- **macOS / Linux (bash):** `mcp_servers/start_sanctuary.sh` — auto-sets `PROJECT_SANCTUARY_ROOT` and selects a Python interpreter (virtualenv or system). Make it executable and run from the repo root:

```bash
chmod +x mcp_servers/start_sanctuary.sh
./mcp_servers/start_sanctuary.sh
```

- **What they do:** Both scripts set the two recommended environment variables used by the template (`PROJECT_SANCTUARY_ROOT` and `PYTHON_EXEC`) so you don't have to export them manually each time. They then call `mcp-host --config "$PROJECT_SANCTUARY_ROOT/config.json"` (or the equivalent on Windows).

Add these scripts to your workflow to ensure the `${...}` placeholders in `.agent/mcp_config.json` resolve automatically when launching MCP servers.

### Updating configs from the env-template (cross-platform)

The previous shell and PowerShell helpers were replaced with a single cross-platform Python tool: `mcp_servers/deploy_mcp_config.py`.

-- **Python (cross-platform):** `mcp_servers/deploy_mcp_config.py`
	- Usage (dry-run):

```bash
python3 mcp_servers/deploy_mcp_config.py --target ClaudeDesktop --dry-run
```

	- Usage (real update with backup):

```bash
python3 mcp_servers/deploy_mcp_config.py --target ClaudeDesktop --backup
```

  - Options: `--target <ClaudeDesktop|Antigravity|RelativeMCP|VSCodeWorkspace|VSCodeUser>` `--template <path>` `--backup` `--dry-run` `--project-root <path>`
  - Behavior: expands `${VAR}` placeholders using environment variables, writes the expanded JSON to the platform-appropriate config location, optionally creates a timestamped backup, and sets secure file permissions on Unix-like systems.

- **Targets implemented:** `ClaudeDesktop`, `Antigravity`, `RelativeMCP`, `VSCodeWorkspace`, `VSCodeUser`.
- **VS Code behavior:** `VSCodeWorkspace` writes `<repo>/.vscode/mcp.json` and sets `.vscode/settings.json` key `"mcp.configPath": "${workspaceFolder}/.vscode/mcp.json"`. `VSCodeUser` writes a user-level `mcp.json` (under the Code User folder) and sets the user's `settings.json` `"mcp.configPath"` to point at that user `mcp.json`.
-- **Important:** Set `PROJECT_SANCTUARY_ROOT`, `PYTHON_EXEC`, and any other placeholders in your shell (or source `.agent/.env`) before running the tool. The launch scripts (`start_sanctuary.sh` / `.ps1`) set these automatically if you run them.

Example workflow to update the Claude config on macOS:

```bash
# set env (or source .agent/.env)
export PYTHON_EXEC="$HOME/Projects/Project_Sanctuary/.venv/bin/python"
export PROJECT_SANCTUARY_ROOT="$HOME/Projects/Project_Sanctuary"

# run update tool with backup
python3 mcp_servers/deploy_mcp_config.py --target ClaudeDesktop --backup
```

If you prefer, run the Python tool on Windows from PowerShell (PowerShell may use `python` or `python3` depending on your PATH) with `--backup` to create a timestamped backup of the current config before replacement.

### Persisting environment variables (make them permanent)

#### macOS / Linux (zsh, bash)
- Option 1 — add to your shell profile (zsh example):

```bash
# open your zsh profile in an editor
nano ~/.zshrc

# append these lines to the end of the file
export PYTHON_EXEC="$HOME/Projects/Project_Sanctuary/.venv/bin/python"
export PROJECT_SANCTUARY_ROOT="$HOME/Projects/Project_Sanctuary"

# save and source to apply immediately
source ~/.zshrc
```

- If you use `bash`, add the same lines to `~/.bashrc` or `~/.profile` instead and then `source` the file or open a new terminal.
- Option 2 — use a local `.env` file inside `.agent/` and source it when you start:

```bash
# .agent/.env (example)
PYTHON_EXEC="$HOME/Projects/Project_Sanctuary/.venv/bin/python"
PROJECT_SANCTUARY_ROOT="$HOME/Projects/Project_Sanctuary"

# usage
source .agent/.env
```

Note: prefer `.agent/.env` for per-repo settings (do not commit secrets). Use `chmod 600 .agent/.env` if it contains sensitive values.

#### Windows (User environment variables)
- Option 1 — GUI (recommended for non-technical users):
	1. Open **Settings → System → About → Advanced system settings** (or press Win and search "Environment Variables").
	2. Click **Environment Variables…**
	3. Under **User variables**, click **New…** and add `PROJECT_SANCTUARY_ROOT` and `PYTHON_EXEC` with their values (e.g. `C:\Users\you\Projects\Project_Sanctuary` and `C:\path\to\.venv\Scripts\python.exe`).
	4. Close and re-open your terminal/PowerShell to pick up the new variables.

- Option 2 — PowerShell (command-line):

```powershell
# sets user-level environment variables (take effect in new shells)
setx PROJECT_SANCTUARY_ROOT "C:\Users\you\Projects\Project_Sanctuary"
setx PYTHON_EXEC "C:\Users\you\Projects\Project_Sanctuary\.venv\Scripts\python.exe"

# or use the .NET API to set for current user and the current process
[System.Environment]::SetEnvironmentVariable('PROJECT_SANCTUARY_ROOT','C:\Users\you\Projects\Project_Sanctuary','User')
[System.Environment]::SetEnvironmentVariable('PYTHON_EXEC','C:\Users\you\Projects\Project_Sanctuary\.venv\Scripts\python.exe','User')
```

Note: `setx` writes to the registry and the new values will be visible in newly opened shells (existing shells keep the previous environment until restarted).

## Security and best practices
- Do not store long-lived secrets (API keys, tokens) in plaintext environment files checked into the repo. Use the OS keychain/credential manager or a secrets store.
- Restrict file permissions for local `.env` files (`chmod 600`) and avoid committing them.
- Prefer workspace-level `.agent/.env` for per-project settings and user-profile entries for global defaults.

