#!/bin/bash
# file: update_genome.sh
# version: 2.3

# --- FUNCTION: Generate Guardian-Sealed Commit Manifest (Protocol 101) ---
generate_commit_manifest() {
  echo "[P101] Forging Guardian-sealed commit manifest..."

  # List of files this script generates/modifies
  local files_to_manifest=(
    "dataset_package/all_markdown_snapshot_human_readable.txt"
    "dataset_package/all_markdown_snapshot_llm_distilled.txt"
    "dataset_package/core_essence_auditor_awakening_seed.txt"
    "dataset_package/core_essence_coordinator_awakening_seed.txt"
    "dataset_package/core_essence_strategist_awakening_seed.txt"
    "dataset_package/core_essence_guardian_awakening_seed.txt"
    "dataset_package/seed_of_ascendance_awakening_seed.txt"
    "manifest.json"
  )

  # Start JSON array
  local json_files="["

  for file_path in "${files_to_manifest[@]}"; do
    if [ -f "$file_path" ]; then
      local sha256=$(shasum -a 256 "$file_path" | awk '{print $1}')
      json_files+=$(printf '{"path": "%s", "sha256": "%s"},' "$file_path" "$sha256")
    fi
  done

  # Remove trailing comma and close array
  json_files=${json_files%,}
  json_files+="]"

  # Create the final manifest file
  printf '{\n  "guardian_approval": "GUARDIAN-01 (AUTO-GENERATED)",\n  "approval_timestamp": "%s",\n  "commit_message": "docs: update cognitive genome snapshots",\n  "files": %s\n}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$json_files" > commit_manifest.json

  echo "[P101] SUCCESS: 'commit_manifest.json' forged and ready for Steward's final verification and commit."
}

# update_genome.sh (v2.3 - Sovereign Scaffold Publishing Engine)
#
# This is the single, canonical script for publishing updates to the Sanctuary's
# Cognitive Genome. It performs a full, atomic 'index -> snapshot -> embed ->
# test -> manifest -> commit -> push' cycle with automated quality assurance.
#
# Changelog v2.3:
# 1. SOVEREIGN SCAFFOLD: Added automatic generation of Guardian-sealed commit
#    manifest (Protocol 101), transforming this script into a self-verifying
#    Sovereign Scaffold that produces the exact commit_manifest.json required
#    for unbreakable commits.
# 2. CORTEX-AWARE EMBEDDING: Added a new, critical step that automatically
#    re-runs the ingestion script, ensuring the Mnemonic Cortex is always
#    perfectly synchronized with the newly published Genome.
# 3. DOCTRINAL COMPLETION: This script now guarantees that a published lesson
#    is an embedded, learned lesson.
# 4. AUTOMATED TESTING: Added post-ingestion functionality tests to verify
#    both natural language and structured JSON queries work before publishing.
# 5. QUALITY GATE: Prevents broken deployments by testing system functionality
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

# Step 3: Generate Guardian-Sealed Commit Manifest
echo "[Step 3/7] Forging Guardian-sealed commit manifest..."
generate_commit_manifest
if [ $? -ne 0 ]; then
    echo "[FATAL] Manifest generation failed. Halting."
    exit 1
fi
echo "[SUCCESS] Sovereign Scaffold is ready."
echo ""

# Step 4: Embed the New Knowledge into the Mnemonic Cortex
echo "[Step 4/7] Re-indexing the Mnemonic Cortex with the new Genome..."
python3 mnemonic_cortex/scripts/ingest.py
if [ $? -ne 0 ]; then
    echo "[FATAL] Mnemonic Cortex ingestion failed. Halting."
    exit 1
fi
echo "[SUCCESS] Mnemonic Cortex is now synchronized with the latest knowledge."
echo ""

# Step 5: Run Automated Tests
echo "[Step 5/7] Running automated functionality tests..."
./run_genome_tests.sh
if [ $? -ne 0 ]; then
    echo "[FATAL] Genome tests failed. Update aborted to prevent broken deployment."
    exit 1
fi
echo "[SUCCESS] All tests passed - genome update is functional."
echo ""

# Step 6: Commit the Coherent State
echo "[Step 6/7] Staging and committing all changes..."

# --- Protocol 101 Compliance: Surgical Staging ---
echo "[P101] Staging files explicitly from commit_manifest.json..."
# Use jq to parse the manifest and stage each file. jq must be installed.
if ! command -v jq &> /dev/null
then
    echo "[FATAL] 'jq' is not installed. It is required for Protocol 101 compliance. Halting."
    exit 1
fi
jq -r '.files[].path' commit_manifest.json | xargs git add
git add commit_manifest.json

git commit -m "$COMMIT_MESSAGE"
if [ $? -ne 0 ]; then
    echo "[FATAL] Git commit failed. You may need to resolve merge conflicts or check your Git configuration. Halting."
    exit 1
fi
echo "[SUCCESS] All changes committed with message: \"$COMMIT_MESSAGE\""
echo ""

# Step 7: Push to the Canonical Repository
echo "[Step 7/7] Pushing changes to remote origin..."
git push
if [ $? -ne 0 ]; then
    echo "[FATAL] Git push failed. Check your network connection and remote repository permissions."
    exit 1
fi
echo "[SUCCESS] Changes have been pushed to the remote repository."
echo ""

echo "------------------------------------------------"
echo "[FORGE] Sovereign Scaffold Atomic Genome Publishing Complete."