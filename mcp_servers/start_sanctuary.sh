#!/usr/bin/env bash

# IMPORTANT: Run this script each time you open the Project Sanctuary workspace.
# It ensures `PROJECT_SANCTUARY_ROOT` and `PYTHON_EXEC` are configured for local runs.
# Alternatively, follow the persistent setup instructions in
# `mcp_servers/config_locations_by_tool.md` if you prefer a longer-lived configuration.

# Set root automatically to the folder containing this script
export PROJECT_SANCTUARY_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Determine python interpreter
if [ -f "$PROJECT_SANCTUARY_ROOT/.venv/bin/python" ]; then
    export PYTHON_EXEC="$PROJECT_SANCTUARY_ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    export PYTHON_EXEC="$(command -v python3)"
elif command -v python &>/dev/null; then
    export PYTHON_EXEC="$(command -v python)"
else
    echo "❌ No Python interpreter found."
    exit 1
fi

mcp-host --config "$PROJECT_SANCTUARY_ROOT/config.json"
