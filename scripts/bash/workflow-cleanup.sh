#!/bin/bash
# workflow-cleanup.sh - Post-Merge Cleanup Wrapper
# Part of the Universal Closure Protocol

# ---------------------------------------------------------
# RED TEAM REMEDIATION: CV-01 (Human Gate)
# ---------------------------------------------------------
echo "=============================================="
echo "🛡️  HUMAN GATE REQURIED: CLEANUP 🛡️"
echo "=============================================="
echo "You are about to SWITCH to 'main' and DELETE the current branch."
echo "This is a DESTRUCTIVE operation."
echo ""
echo "Type 'PROCEED' to execute."
read -p "> " approval

if [[ "$approval" != "PROCEED" ]]; then
    echo "❌ Approval not given. Aborting."
    exit 1
fi

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
CLI_PATH="$PROJECT_ROOT/tools/cli.py"

# Handover to Python
exec python3 "$CLI_PATH" workflow cleanup "$@"
