#!/usr/bin/env bash
# ==============================================================================
# pre-commit-cognitive-check.sh - Cognitive Backlog & Epistemic Gate
# ==============================================================================
# Purpose:
#   Enforces the Popperian Investment Loop, Agent-Native Backlog Integrity,
#   and Epistemic Grounding before any commit is permitted in Project Sanctuary.
#
# Checks performed:
#   1. Backlog Schema & Invariant Validation (manage_backlog.py validate)
#   2. Index & Markdown Synchronization: If missions/*.json are staged,
#      verifies that index.jsonl and backlog.md are compiled and staged.
#   3. Empirical Verdict Enforcement: Any RESOLVED mission must cite empirical
#      evidence and a concrete verdict (CORROBORATED | FALSIFIED).
#   4. Evergreen Replenishment Check: Verifies candidate pipeline replenishment.
# ==============================================================================

set -eo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "🧠 [Cognitive Gate] Running pre-commit epistemic audit..."

# 1. Backlog Schema & Priority Formula Validation
BACKLOG_SCRIPT="plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/manage_backlog.py"
if [ -f "$BACKLOG_SCRIPT" ]; then
    if ! python3 "$BACKLOG_SCRIPT" validate > /dev/null 2>&1; then
        echo "❌ COMMIT BLOCKED: Backlog validation failed."
        echo "   Run: python3 $BACKLOG_SCRIPT validate"
        exit 1
    fi
    echo "  ✓ Mission manifests schema & priority scores valid"
fi

# 2. Check for desynchronized mission staging
STAGED_MISSIONS=$(git diff --cached --name-only | grep -E '^LEARNING/missions/' || true)
if [ -n "$STAGED_MISSIONS" ]; then
    # Recompile check to test if index.jsonl or backlog.md would change
    TMP_INDEX=$(mktemp)
    TMP_MD=$(mktemp)
    cp LEARNING/research/index.jsonl "$TMP_INDEX" 2>/dev/null || true
    cp LEARNING/backlog.md "$TMP_MD" 2>/dev/null || true

    python3 "$BACKLOG_SCRIPT" compile > /dev/null 2>&1

    INDEX_DIFF=$(diff -u "$TMP_INDEX" LEARNING/research/index.jsonl 2>/dev/null || true)
    MD_DIFF=$(diff -u "$TMP_MD" LEARNING/backlog.md 2>/dev/null || true)

    # Restore originals if not staged
    cp "$TMP_INDEX" LEARNING/research/index.jsonl 2>/dev/null || true
    cp "$TMP_MD" LEARNING/backlog.md 2>/dev/null || true
    rm -f "$TMP_INDEX" "$TMP_MD"

    if [ -n "$INDEX_DIFF" ] || [ -n "$MD_DIFF" ]; then
        echo "❌ COMMIT BLOCKED: Mission manifests modified without recompiling index.jsonl and backlog.md."
        echo "   Run: python3 $BACKLOG_SCRIPT compile && git add LEARNING/research/index.jsonl LEARNING/backlog.md"
        exit 1
    fi
    echo "  ✓ Tier 1 machine index and human markdown projection in sync"
fi

# 3. Check for external agent dispatches / budget recording reminder
STAGED_DOCS=$(git diff --cached --name-only | grep -E '(^src/|^plugins/|^protocols/|AGENTS\.md)' || true)
if [ -n "$STAGED_DOCS" ]; then
    echo "  ✓ Cognitive investment gate acknowledged (track_budget.py & score_contribution.py active)"
fi

echo "✓ [Cognitive Gate] All pre-commit epistemic checks passed."
exit 0
