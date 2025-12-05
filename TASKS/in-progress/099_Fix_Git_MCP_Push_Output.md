# Tactical Mandate T095: Fix Git MCP Push Output Assertions

## 🎯 Goal
Resolve the assertion failures in the Git MCP's safety test suite (`test_tool_safety.py`). This requires modifying the `git_push_feature` operation to return descriptive status messages (`Verified push` and `WARNING` for hash mismatch) as required by the tests, ensuring the Git MCP adheres to **Protocol 101 (Functional Coherence Gate)**.

## 🛠️ Server & Operation
* **Primary Server:** MCP_SERVER_GIT
* **Operation:** GIT_PUSH_FEATURE (Fix Assertions)

## 📋 Task Details

### A. Code Refactoring (Git MCP)

**Mandate:** Modify the Python logic within the Git MCP that handles the result of the underlying `git push` command.

**Target File (Primary):** `mcp_servers/git/operations.py` (or the file where `git_push_feature` is defined)

**Required Changes:**
1.  **Successful Push:** If the push succeeds and the local commit hash (`HEAD`) matches the remote branch hash (`origin/feature/...`), the function **must** include the string `"Verified push"` in its final return result.
2.  **Hash Mismatch Warning:** If the push succeeds but the local commit hash does **not** match the remote branch hash, the function **must** include the string `"WARNING"` in its final return result.

The current simple return of `"Push successful"` is insufficient for the safety checks.

### B. Verification Checklist

1.  [ ] `git_push_feature` logic updated to return descriptive messages.
2.  [ ] Run `pytest tests/mcp_servers/git/ -v` to confirm **32/32 tests now pass**.

## 🕰️ Effort Estimate (T-Shirts)
**Size:** S (Small - Highly targeted code fix in a single function)
