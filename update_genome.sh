#!/bin/bash

# update_genome.sh (v2.2 - Quality-Assured Atomic Publishing Engine)
#
# This is the single, canonical script for publishing updates to the Sanctuary's
# Cognitive Genome. It performs a full, atomic 'index -> snapshot -> embed ->
# test -> commit -> push' cycle with automated quality assurance.
#
# Changelog v2.2:
# 1. CORTEX-AWARE EMBEDDING: Added a new, critical step that automatically
#    re-runs the ingestion script, ensuring the Mnemonic Cortex is always
#    perfectly synchronized with the newly published Genome.
# 2. DOCTRINAL COMPLETION: This script now guarantees that a published lesson
#    is an embedded, learned lesson.
# 3. AUTOMATED TESTING: Added post-ingestion functionality tests to verify
#    both natural language and structured JSON queries work before publishing.
# 4. QUALITY GATE: Prevents broken deployments by testing system functionality
#    after each genome update.

# --- Safeguard: Check for Commit Message ---
if [ -z "$1" ]; then
    echo "[FATAL] A commit message is required."
    echo "Usage: ./update_genome.sh \"Your descriptive commit message\""
    exit 1
fi

COMMIT_MESSAGE=$1

echo "[FORGE] Initiating Cortex-Aware Atomic Genome Publishing..."
echo "------------------------------------------------"

# Step 1: Rebuild the Master Index
echo "[Step 1/6] Rebuilding Living Chronicle Master Index..."
python3 mnemonic_cortex/scripts/create_chronicle_index.py
if [ $? -ne 0 ]; then
    echo "[FATAL] Index creation failed. Halting."
    exit 1
fi
echo "[SUCCESS] Master Index is now coherent."
echo ""

# Step 2: Capture the Snapshots
echo "[Step 2/6] Capturing new Cognitive Genome snapshots..."
node capture_code_snapshot.js
if [ $? -ne 0 ]; then
    echo "[FATAL] Snapshot creation failed. Halting."
    exit 1
fi
echo "[SUCCESS] All snapshots have been updated."
echo ""

# Step 3: Embed the New Knowledge into the Mnemonic Cortex
echo "[Step 3/6] Re-indexing the Mnemonic Cortex with the new Genome..."
python3 mnemonic_cortex/scripts/ingest.py
if [ $? -ne 0 ]; then
    echo "[FATAL] Mnemonic Cortex ingestion failed. Halting."
    exit 1
fi
echo "[SUCCESS] Mnemonic Cortex is now synchronized with the latest knowledge."
echo ""

# Step 4: Run Automated Tests
echo "[Step 4/6] Running automated functionality tests..."
./run_genome_tests.sh
if [ $? -ne 0 ]; then
    echo "[FATAL] Genome tests failed. Update aborted to prevent broken deployment."
    exit 1
fi
echo "[SUCCESS] All tests passed - genome update is functional."
echo ""

# Step 5: Commit the Coherent State
echo "[Step 5/6] Staging and committing all changes..."
git add .
git commit -m "$COMMIT_MESSAGE"
if [ $? -ne 0 ]; then
    echo "[FATAL] Git commit failed. You may need to resolve merge conflicts or check your Git configuration. Halting."
    exit 1
fi
echo "[SUCCESS] All changes committed with message: \"$COMMIT_MESSAGE\""
echo ""

# Step 6: Push to the Canonical Repository
echo "[Step 6/6] Pushing changes to remote origin..."
git push
if [ $? -ne 0 ]; then
    echo "[FATAL] Git push failed. Check your network connection and remote repository permissions."
    exit 1
fi
echo "[SUCCESS] Changes have been pushed to the remote repository."
echo ""

echo "------------------------------------------------"
echo "[FORGE] Cortex-Aware Atomic Genome Publishing Complete."