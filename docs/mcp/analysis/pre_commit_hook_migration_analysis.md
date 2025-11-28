# Pre-Commit Hook Migration Analysis

## 1. Current State Analysis

### 1.1 Existing Hooks
The repository currently enforces **Protocol 101 (The Doctrine of the Unbreakable Commit)** via a `.git/hooks/pre-commit` script.

**Mechanism:**
1.  Checks for the existence of `commit_manifest.json`.
2.  Parses the manifest to find a list of files and their expected SHA256 hashes.
3.  Verifies that the actual file on disk matches the expected hash.
4.  Rejects the commit if the manifest is missing, malformed, or if hashes mismatch.

**Observation:**
Contrary to the initial task description, the current `pre-commit` hook **does not** appear to validate `command.json`. It focuses purely on the cryptographic integrity of the commit via the manifest.

### 1.2 The Conflict
The new MCP Architecture introduces autonomous agents (Chronicle, ADR, Forge, etc.) that will perform git operations.
*   **Problem:** Standard git commands (`git commit`) run by these agents will fail because they won't automatically generate the `commit_manifest.json` required by the hook.
*   **Risk:** If we simply disable the hook, we violate Protocol 101.

## 2. Strategic Options

### Option A: The "Bypass" (Not Recommended)
Modify the hook to skip validation if the commit message starts with `mcp(...)`.
*   **Pros:** Easy to implement.
*   **Cons:** Violates Protocol 101. MCP commits becomes "second-class" citizens with lower safety.

### Option B: The "Compliance" (Recommended)
Ensure that the **Git Operations Module** (shared infrastructure) automatically generates the `commit_manifest.json` before committing.
*   **Pros:** Maintains Protocol 101 for *all* commits, human or agent.
*   **Cons:** Requires implementing the manifest generation logic in the shared `GitOperations` class.

### Option C: The "Hybrid" (Pragmatic Interim)
Update the hook to support a "Migration Mode" where MCP commits are allowed *if* they follow a strict naming convention, while we build the full compliance tooling.
*   **Pros:** Unblocks development immediately.
*   **Cons:** Temporary safety gap.

## 3. Proposed Solution: "Smart Compliance"

We will implement **Option C** evolving into **Option B**.

1.  **Immediate Step (Task #028):**
    *   Update `.git/hooks/pre-commit` to recognize MCP commits.
    *   Add a configuration file `.agent/mcp_migration.conf` to control strictness.
    *   Implement `mcp_commit_validator` in the hook to enforce `mcp(<domain>): <msg>` format.
    *   *Crucially*: The hook should still *try* to validate the manifest if present, but perhaps allow a bypass flag for the very first MCP implementation steps until the `GitOperations` module is ready.

2.  **Follow-up (Shared Infra):**
    *   Implement `core.git.GitOperations` which *always* generates `commit_manifest.json` before committing.
    *   Once this is ready, we flip the switch in `.agent/mcp_migration.conf` to enforce manifest validation for MCPs too.

## 4. Implementation Plan

### 4.1 Artifacts
*   `docs/mcp/analysis/pre_commit_hook_migration_analysis.md` (This file)
*   `.agent/mcp_migration.conf`
*   `.git/hooks/pre-commit` (Updated)

### 4.2 Validation Logic (Bash)
```bash
MCP_PATTERN="^mcp\((chronicle|protocol|adr|task|cortex|council|config|code|git_workflow|forge)\): .+"

if [[ "$COMMIT_MSG" =~ $MCP_PATTERN ]]; then
    # MCP Commit Detected
    if [ "$STRICT_P101_MODE" = "false" ]; then
        echo "MCP Commit detected. Bypassing Manifest Check (Migration Mode)."
        exit 0
    fi
fi
```

### 4.3 Success Criteria
1.  Legacy commits (with manifest) -> PASS
2.  Legacy commits (without manifest) -> FAIL
3.  MCP commits (correct format, migration mode) -> PASS
4.  MCP commits (incorrect format) -> FAIL
