#!/bin/bash
# workflow-end.sh - Wrapper for Python CLI
# Part of the Universal Closure Protocol

# ---------------------------------------------------------
# RED TEAM REMEDIATION: CV-01 (Human Gate) & SW-02 (Protocol 128)
# ---------------------------------------------------------

# 1. Protocol 128 Enforcement (SW-02)
# Check if a learning snapshot exists in the audit trail (created by learning loop)
AUDIT_DIR="$PWD/.agent/learning/learning_audit"
if [ ! -d "$AUDIT_DIR" ] || [ -z "$(ls -A $AUDIT_DIR 2>/dev/null)" ]; then
    echo "⚠️  WARNING: No learning audit trail found in $AUDIT_DIR"
    echo "Protocol 128 requires cognitive continuity."
    echo "Did you run /workflow-learning-loop?"
    # We warn but do not block, to allow emergency closure.
    echo "Proceeding with CAUTION..."
fi

# 2. explicit Human Gate (CV-01)
# State-Changing Operation: Persisting memory and closing session.
echo "=============================================="
echo "🛡️  HUMAN GATE REQURIED: SESSION CLOSURE 🛡️"
echo "=============================================="
echo "You are about to SEAL and PERSIST this session."
echo "This is a STATE-CHANGING operation."
echo ""
echo "Type 'PROCEED' to execute closure."
read -p "> " approval

if [[ "$approval" != "PROCEED" ]]; then
    echo "❌ Approval not given. Aborting closure."
    exit 1
fi

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
CLI_PATH="$PROJECT_ROOT/tools/cli.py"

# Handover to Python
exec python3 "$CLI_PATH" workflow end "$@"
