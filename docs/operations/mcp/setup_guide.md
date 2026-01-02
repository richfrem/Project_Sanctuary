# MCP Server Setup Guide

> [!IMPORTANT]
> **🚀 Unified Fleet Deployment (ADR 065)**  
> As of December 2025, Project Sanctuary uses a **unified Makefile-based deployment** for the Fleet of 8 MCP servers.  
> For deploying the complete fleet, see [PODMAN_STARTUP_GUIDE.md](../../PODMAN_STARTUP_GUIDE.md).
>
> This guide covers **individual MCP server development and standalone deployment** for advanced use cases.

This guide documents the standard process for creating, containerizing, and integrating MCP servers with Claude Desktop, based on the implementation of the Task MCP server.

## 1. Project Structure

Ensure your MCP server follows this structure to be importable as a module:

```
mcp_servers/
├── __init__.py          # CRITICAL: Required for python -m execution
└── server_name/
    ├── __init__.py      # Package init
    ├── server.py        # Main entry point (MCP server)
    ├── models.py        # Data models
    ├── operations.py    # Core logic (separation of concerns)
    ├── validator.py     # Input validation
    ├── Dockerfile       # Container definition
    ├── requirements.txt # Dependencies
    └── README.md        # Documentation
```

**Key Learning:** You MUST have an `__init__.py` in the root `mcp_servers/` directory, otherwise `python -m mcp_servers.task.server` will fail.

---

## 2. Configuration Template

A template configuration file is available at [`docs/architecture/mcp/claude_desktop_config_template.json`](claude_desktop_config_template.json).

**Important:** Claude Desktop **requires absolute paths**. You cannot use relative paths (like `./` or `../`) in the configuration file because Claude Desktop launches from its own application directory, not your project directory.

**Template Usage:**
1. Copy the content from the template.
2. Replace `<ABSOLUTE_PATH_TO_PROJECT>` with your full project path (e.g., `/Users/username/Projects/Project_Sanctuary`).
3. Paste into your `claude_desktop_config.json`.

Create a `Dockerfile` in your server directory.

**Build the Image:**
```bash
cd mcp_servers/task
podman build -t task-mcp:latest .
```

**Run the Container (Production):**
```bash
podman run -d \
  --name task-mcp \
  -v $(pwd)/tasks:/app/tasks:rw \
  -p 3004:8080 \
  task-mcp:latest
```

**Verify Running:**
```bash
# Check status (should show Up or Exited(0))
podman ps -a | grep task-mcp

# View logs
podman logs task-mcp
```
*Note: Stdio-based servers will exit immediately if no input is provided. This is normal behavior for stdio transport.*

---

## 3. Configuring Claude Desktop

To use the server locally (development mode), configure Claude Desktop to run the Python script directly.

**Config File Location:**
```bash
# Open in terminal editor
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Or open in VS Code (if installed)
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Configuration Format (CRITICAL):**
You **MUST** use absolute paths to the virtual environment's Python executable.
We recommend using **simplified keys** (e.g., `tasks`) combined with a `displayName` for a cleaner configuration.

```json
{
  "mcpServers": {
    "tasks": {
      "displayName": "Task MCP",
      "command": "/Users/username/Projects/Project_Sanctuary/.venv/bin/python",
      "args": [
        "-m",
        "mcp_servers.task.server"
      ],
      "env": {
        "PYTHONPATH": "/Users/username/Projects/Project_Sanctuary",
        "PROJECT_ROOT": "/Users/username/Projects/Project_Sanctuary"
      },
      "cwd": "/Users/username/Projects/Project_Sanctuary"
    }
  }
}
```

**Why Absolute Paths?**
Claude Desktop does not load your shell's `.bashrc` or `.zshrc`, so it doesn't know where `python` is or what virtual environment to use. Using the full path `/path/to/.venv/bin/python` ensures it uses the correct environment with all installed dependencies.

---

## 4. Verification

1.  **Restart Claude Desktop** (Quit completely via Cmd+Q).
2.  **Check Connection:** Look for the 🔌 icon or ask "What tools are available?".
3.  **Test with Natural Language:**
    > "Create a test task #099 to verify MCP integration."

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `spawn python ENOENT` | Claude can't find python executable | Use absolute path to `.venv/bin/python` |
| `ModuleNotFoundError` | Python can't find the module | Ensure `PYTHONPATH` is set and `__init__.py` exists |
| `Connection Refused` | Server crashed or not running | Check logs at `~/Library/Logs/Claude/` |

---

**Related Documentation:**
- [Task MCP README](../../mcp_servers/task/README.md)
- [Prerequisites](prerequisites.md)
