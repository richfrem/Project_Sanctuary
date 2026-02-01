#!/bin/bash
# workflow-end.sh - Wrapper for Python CLI
# Part of the Universal Closure Protocol

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
CLI_PATH="$PROJECT_ROOT/tools/cli.py"

# Handover to Python
exec python3 "$CLI_PATH" workflow end "$@"
