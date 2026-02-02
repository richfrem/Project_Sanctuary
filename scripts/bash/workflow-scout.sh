#!/bin/bash
# Shim for /workflow-scout (Protocol 128 Phase I)
# Purpose: Orientation and context acquisition for new sessions
# Reference: .agent/workflows/workflow-scout.md, workflow-learning-loop.md

set -e

source "$(dirname "$0")/common.sh"

echo "=============================================="
echo "🦉 PHASE I: THE LEARNING SCOUT 🦉"
echo "=============================================="
echo ""

# Step 1: Guardian Wakeup (Iron Check + HMAC)
echo "Step 1/3: Running Guardian Wakeup..."
python3 tools/cli.py guardian --mode TELEMETRY

# Step 2: Run the debrief workflow
echo ""
echo "Step 2/3: Running Debrief..."
python3 tools/cli.py workflow run --name workflow-scout "$@"

# Step 3: Reminder to read truth anchor
echo ""
echo "Step 3/3: Truth Anchor"
echo "=============================================="
echo "📜 REMINDER: Read the learning_package_snapshot.md"
echo "   Location: .agent/learning/learning_package_snapshot.md"
echo "   This is your Cognitive Hologram - your memory from prior sessions."
echo "=============================================="

