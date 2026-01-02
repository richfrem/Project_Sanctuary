# Project Sanctuary MCP Quickstart Guide

**Welcome to Project Sanctuary!** This guide will help you connect your LLM client (Claude Desktop, Antigravity, or others) to the Project Sanctuary Model Context Protocol (MCP) ecosystem.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.11+**: [Download Python](https://www.python.org/downloads/)
2.  **uv**: An extremely fast Python package installer and resolver.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3.  **Git**: Version control system.
4.  **Podman** (Optional but Recommended): For running MCP servers in isolated containers.

## Step 1: Clone the Repository

```bash
git clone https://github.com/richfremmerlid/Project_Sanctuary.git
cd Project_Sanctuary
```

## Step 2: Environment Setup

Create a `.env` file in the project root to configure your environment.

```bash
# Copy the example template
cp .env.example .env

# Edit the file with your specific paths and keys
nano .env
```

**Critical Variables:**
*   `PROJECT_ROOT`: Absolute path to your project directory.
*   `PYTHONPATH`: Should include your project root.

## Step 3: Configure Your Client

### Option A: Claude Desktop

1.  Open the Claude Desktop configuration file:
    *   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    *   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2.  Add the Project Sanctuary MCP servers. You can use the helper script to generate this config:

    ```bash
    # Generate config for all servers
    python3 scripts/generate_mcp_config.py
    ```

    Or manually add entries like this:

    ```json
    {
      "mcpServers": {
        "council": {
          "command": "uv",
          "args": [
            "--directory",
            "/absolute/path/to/Project_Sanctuary/mcp_servers/council",
            "run",
            "server.py"
          ],
          "env": {
            "PROJECT_ROOT": "/absolute/path/to/Project_Sanctuary"
          }
        }
      }
    }
    ```

### Option B: Antigravity / Other Clients

Refer to your specific client's documentation for adding MCP servers. The connection details (command, args, env) remain the same.

## Step 4: Initialize Session Context (Protocol 118)

Before starting your first interaction, you must initialize the session context. This runs the `guardian_wakeup` sequence to hydrate the RAG cache and verify system health.

```bash
python scripts/init_session.py
```

This will generate a **Guardian Briefing** (`WORK_IN_PROGRESS/guardian_boot_digest.md`) which serves as the "strategic signal" for the session.


## Step 5: Verify Connection

1.  Restart your client (Claude Desktop, etc.).
2.  Look for the **MCP** icon or menu to confirm servers are connected.
3.  Try a test prompt:

    > "Can you ask the Council to review the current task list?"

    If the system responds by dispatching a task to the Council MCP, you are connected!

## Troubleshooting

*   **"Server not found"**: Check your absolute paths in the config JSON.
*   **"Permission denied"**: Ensure `uv` is in your PATH and executable.
*   **"Python error"**: Check `PROJECT_ROOT` and `PYTHONPATH` environment variables.

## Next Steps

*   **[Using the Council](../processes/01_using_council_mcp.md)**: Learn how to orchestrate multi-agent tasks.
*   **[Using the Cortex](../processes/02_using_cortex_mcp.md)**: Learn how to query the knowledge base.
*   **[Architecture Overview](architecture.md)**: Understand the 11-server ecosystem.
