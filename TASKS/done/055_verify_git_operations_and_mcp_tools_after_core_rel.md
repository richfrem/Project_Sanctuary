# TASK: Verify Git Operations and MCP Tools After Core Relocation

**Status:** done
**Priority:** High
**Lead:** Antigravity
**Dependencies:** None
**Related Documents:** tests/test_git_ops.py, mcp_servers/lib/git/git_ops.py, mcp_servers/system/git_workflow/server.py

---

## 1. Objective

Ensure all git operations work correctly after relocating core/ to mcp_servers/lib/ and adding new force/no_verify parameters. Create comprehensive test suite for both direct GitOperations class and Git MCP tools, with clear documentation and test results.

## 2. Deliverables

1. Test suite for all GitOperations class methods (add, commit, push with force/no_verify, branch operations, status, diff, log)
2. Updated documentation (README or test guide) explaining how to run the test suite and outlining all tests
3. Test results report showing all tests passing
4. Replicate same tests using Git MCP tools (git_start_feature, git_add, git_smart_commit, git_push_feature with no_verify, git_create_pr, git_finish_feature)
5. MCP test results report
6. Any bug fixes discovered during testing

## 3. Acceptance Criteria

- All existing unit tests in tests/test_git_ops.py pass
- New tests for force and no_verify parameters added and passing
- Integration test successfully creates temp branch, commits with manifest, pushes with no_verify=True, and cleans up
- Documentation clearly explains how to run test suite and what each test covers
- Test results documented and shared with user
- All Git MCP tools verified working with same test scenarios
- MCP test results documented and shared with user

## Notes

This is critical after the core relocation to ensure git operations are stable. Should be completed before any new feature work.

**Status Change (2025-11-28):** todo → in-progress
Moving to in-progress to work on tomorrow along with 022A, 022B, and 022C
