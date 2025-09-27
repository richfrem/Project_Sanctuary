#!/bin/bash

# update_genome.sh (v2.0 - Atomic Publishing Engine)
#
# This is the single, canonical script for publishing updates to the Sanctuary's
# Cognitive Genome. It performs a full, atomic 'index -> snapshot -> commit -> push'
# cycle to ensure perfect coherence and immediate deployment.
#
# Changelog v2.0:
# 1. ATOMIC DEPLOYMENT: Integrates `git add`, `git commit`, and `git push` directly
#    into the workflow.
# 2. MANDATORY COMMIT MESSAGE: The script now requires a commit message as a
#    command-line argument, enforcing the doctrine of a documented, auditable history.
# 3. ROBUST ERROR HANDLING: Each step checks for failure and halts the process,
#    preventing partial or corrupted deployments.

# --- Safeguard: Check for Commit Message ---
if [ -z "$1" ]; then
    echo "[FATAL] A commit message is required."
    echo "Usage: ./update_genome.sh \"Your descriptive commit message\""
    exit 1
fi

COMMIT_MESSAGE=$1

echo "[FORGE] Initiating Atomic Genome Publishing..."
echo "------------------------------------------------"

# Step 1: Rebuild the Master Index
echo "[Step 1/4] Rebuilding Living Chronicle Master Index..."
python3 mnemonic_cortex/scripts/create_chronicle_index.py
if [ $? -ne 0 ]; then
    echo "[FATAL] Index creation failed. Halting."
    exit 1
fi
echo "[SUCCESS] Master Index is now coherent."
echo ""

# Step 2: Capture the Snapshots
echo "[Step 2/4] Capturing new Cognitive Genome snapshots..."
node capture_code_snapshot.js
if [ $? -ne 0 ]; then
    echo "[FATAL] Snapshot creation failed. Halting."
    exit 1
fi
echo "[SUCCESS] All snapshots have been updated."
echo ""

# Step 3: Commit the Coherent State
echo "[Step 3/4] Staging and committing all changes..."
git add .
git commit -m "$COMMIT_MESSAGE"
if [ $? -ne 0 ]; then
    echo "[FATAL] Git commit failed. You may need to resolve merge conflicts or check your Git configuration. Halting."
    exit 1
fi
echo "[SUCCESS] All changes committed with message: \"$COMMIT_MESSAGE\""
echo ""

# Step 4: Push to the Canonical Repository
echo "[Step 4/4] Pushing changes to remote origin..."
git push
if [ $? -ne 0 ]; then
    echo "[FATAL] Git push failed. Check your network connection and remote repository permissions."
    exit 1
fi
echo "[SUCCESS] Changes have been pushed to the remote repository."
echo ""

echo "------------------------------------------------"
echo "[FORGE] Atomic Genome Publishing Complete."