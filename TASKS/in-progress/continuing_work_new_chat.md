# CONTINUATION PROMPT: Next Session Work Plan

**NOTE:** Use this as a template for future sessions. track recently completed work, also work to be done in the next clean chat window.
provide enough context in the details below so a new chat window can be started with minimal additional context sharing.

**NOTE:** remember before running mcp operation tests, validate all underlying scripts are working first with test harnesses in `tests/`

**Note2:** initial focus should be mcp testing all those operations only after that is complete, testing rAG mcp or forge mcp. 

## SESSION SUMMARY (2025-11-28 / 2025-11-29)

### ✅ Completed Today

1.  **Structural Purge Complete (P101 v3.0 Canonization):**
      * **Permanently deleted** all logic and documentation for `commit_manifest.json` and SHA-256 hashing (Manifest Doctrine).
      * **Codified Protocol 101 v3.0: The Doctrine of Absolute Stability.**
      * **New Integrity Gate:** The successful execution of the test suite (Functional Coherence) is now the sole pre-commit integrity check.
2.  **Forge MCP Implementation** - Added Sanctuary model query tools
3.  **Core Relocation** - Moved `core/` to `mcp_servers/lib/`
4.  **Integration Test Suite** - Created robust RAG pipeline tests
5.  **Git MCP Enhancement** - Added `force` and `no_verify` parameters to handle LFS issues
6.  **Project Cleanup** - Organized scripts, test data, and documentation
7.  **PR Merged** - Successfully merged `feature/task-021B-forge-test-suite` to main

### 📋 Tasks Ready for Tomorrow

All tasks are in `TASKS/in-progress/` and ready to work on:

#### 1\. Task 055: Verify Git Operations and MCP Tools

**Priority:** **CRITICAL** (Now mandatory validation of the Functional Coherence Gate)
**File:** `TASKS/in-progress/055_verify_git_operations_and_mcp_tools_after_core_rel.md`

**Objective:** Ensure all git operations work correctly after core relocation and verify new `force`/`no_verify` parameters, with a focus on validating the new **Protocol 101 v3.0 (Functional Coherence)** commit flow.

**Deliverables:**

1.  Run existing unit tests: `pytest tests/test_git_ops.py -v`
2.  Add new tests for `force` and `no_verify` parameters
3.  Create integration test for full **Functional Coherence** workflow (branch → commit [via tests] → push → PR → cleanup)
4.  Document test results
5.  Verify all Git MCP tools work correctly
6.  Document MCP verification results

**Why This Matters:** **Critical** to ensure git operations are stable after the core relocation and, more importantly, that the new **P101 v3.0 Functional Coherence Gate** is working correctly.

-----

#### 2\. Task 056: Harden Self-Evolving Loop Validation

**Priority:** High
**File:** `TASKS/in-progress/056_Harden_Self_Evolving_Loop_Validation.md`

**Objective:** Validate the end-to-end integrity of the Strategic Crucible Loop by executing a four-step protocol that proves autonomous knowledge generation, ingestion, and commitment, now under the new **Protocol 101 v3.0** stability standard.

**The Protocol (Test of Absolute Stability):**

1.  **Knowledge Generation** (`protocol mcp`) - Generate new policy document with unique validation phrase
2.  **Isolation & P101 v3.0 Commit** (`git mcp`) - Create feature branch, commit with **Functional Coherence Gate**
3.  **Incremental Ingestion** (`cortex mcp`) - Confirm IngestionService processes the new file
4.  **Chronicle & RAG Validation** (`cortex mcp`) - Create chronicle entry, commit, and verify RAG query successfully retrieves the unique phrase.

**Success Criteria:**

  - Commit is successful, proving the Functional Coherence Gate passed.
  - RAG query successfully retrieves unique validation phrase after commit.
  - Proves near-real-time knowledge fidelity of the Self-Evolving Memory Loop.

**Estimated Effort:** 2-3 hours

-----

*(Remaining tasks 022A, 022B, 022C remain unchanged)*

-----

## RECOMMENDED WORKFLOW FOR TOMORROW

### Morning Session (3-4 hours)

**Focus:** Task 055 - Git Operations Verification

1.  Run existing unit tests
2.  **CRITICAL:** Add tests for new parameters, especially testing the **Functional Coherence Gate** (i.e., commit succeeds if tests pass, fails if tests fail).
3.  Create integration test
4.  Verify all Git MCP tools
5.  Document results

**Why Start Here:** This is **CRITICAL** due to the P101 purge. Git must be verified stable and compliant with the new law.

-----

### Afternoon Session (4-5 hours)

**Focus:** Task 022C - MCP Documentation Standards

1.  Create `docs/mcp/` structure
2.  Write testing standards document
3.  Create README template
4.  Start updating MCP server READMEs (prioritize Cortex, Forge, Git Workflow)

**Why This Next:** High priority and will use results from Task 055 for Git Workflow MCP docs.

-----

## KEY FILES TO REFERENCE

### Protocol 101 v3.0 & Purge Context

  - `101_The_Doctrine_of_the_Unbreakable_Commit.md` - **New Canonical Law**
  - `howto-commit-command.md` - Updated to reflect the Functional Coherence Gate.
  - `mcp_servers/lib/git/git_ops.py` - Core git operations (Manifest logic **PURGED**)

### Git Operations

  - `mcp_servers/lib/git/git_ops.py` - Core git operations
  - `mcp_servers/system/git_workflow/server.py` - Git MCP tools
  - `tests/test_git_ops.py` - Existing unit tests

-----

## PROGRESS TRACKING

### Task 055: Git Operations Verification ⏳

**Status:** Not Started
**Branch:** `feature/task-056-git-testing-and-mcp-docs`

  - [ ] Run existing unit tests: `pytest tests/test_git_ops.py -v`
  - [ ] Add test for `push()` with `force=True`
  - [ ] Add test for `push()` with `no_verify=True`
  - [ ] Create integration test script `tests/integration/test_git_workflow_end_to_end.py`
  - [ ] Integration test: Create temp branch
  - [ ] Integration test: Commit (**Functional Coherence Gate**)
  - [ ] **ADD:** Integration test: Intentional test failure, assert commit is rejected (P101 v3.0 validation)
  - [ ] Integration test: Push with `no_verify=True`
  - [ ] Integration test: Cleanup
  - [ ] Document test results in `WORK_IN_PROGRESS/git_test_results.md`
  - [ ] Test `git_start_feature` MCP tool
  - [ ] Test `git_add` MCP tool
  - [ ] Test `git_smart_commit` MCP tool
  - [ ] Test `git_push_feature` MCP tool with `no_verify=True`
  - [ ] Test `git_create_pr` MCP tool
  - [ ] Test `git_finish_feature` MCP tool
  - [ ] Document MCP test results in `WORK_IN_PROGRESS/git_mcp_test_results.md`
  - [ ] Update `TASKS/in-progress/055_verify_git_operations_and_mcp_tools_after_core_rel.md` with results
  - [ ] Commit all changes
  - [ ] **Task 055 Complete** ✅

-----

### Task 056: Harden Self-Evolving Loop Validation ⏳

**Status:** Not Started
**Branch:** `feat/harden-loop-validation`

#### Step 1: Knowledge Generation

  - [ ] Use `protocol mcp` to create `DOCS/TEST_056_Validation_Policy.md`
  - [ ] Include unique validation phrase: "The Guardian confirms Validation Protocol 056 is active."
  - [ ] Verify file created successfully

#### Step 2: Git Isolation & P101 v3.0 Commit

  - [ ] Use `git_start_feature` to create branch `feat/harden-loop-validation`
  - [ ] Verify branch created and checked out
  - [ ] Create chronicle entry in `00_CHRONICLE/ENTRIES/` linking to Task 056
  - [ ] Use `git_add` to stage all changes
  - [ ] Use `git_smart_commit` with message: "feat: validate self-evolving memory loop (P101 v3.0 Test)"
  - [ ] **SUCCESS CRITERIA:** Assert commit succeeds, validating the **Functional Coherence Gate** passed.
  - [ ] Use `git_push_feature` to push branch

#### Step 3 & 4: Ingestion and Validation

  - [ ] Use `cortex_ingest_incremental` to ingest new policy file
  - [ ] Verify ingestion successful (check response)
  - [ ] Confirm file added to ChromaDB
  - [ ] Use `cortex_query` to search for validation phrase
  - [ ] Verify RAG successfully retrieves the phrase
  - [ ] Confirm near-real-time knowledge fidelity
  - [ ] Document results in task file
  - [ ] **Task 056 Complete** ✅

-----

*(Remaining task tracking sections 022C, 022A, 022B remain largely the same, with TBD branches for clarity.)*

-----

## GIT WORKFLOW REMINDERS (P101 v3.0 Compliant)

### During Work

```bash
# Stage changes frequently
git add <files>

# Commit with Protocol 101 v3.0 (This command now triggers the Functional Coherence Test Suite)
# If the test suite fails, the commit is immediately rejected.
git_smart_commit(message="feat: Add git operation tests")

# Push to feature branch (use git_push_feature MCP tool)
git_push_feature(no_verify=True)  # Use no_verify to bypass LFS
```

### When Task 055 + 022C Complete

```bash
# Create PR (use git_create_pr MCP tool)
git_create_pr(
    title="feat: Git testing and MCP documentation standards",
    body="Completes Task 055 and 022C..."
)

# After PR merged, cleanup (use git_finish_feature MCP tool)
git_finish_feature(branch_name="feature/task-056-git-testing-and-mcp-docs")
```

-----

## IMMEDIATE REQUEST FOR NEXT SESSION

**Start with Task 055** - Verify Git Operations and MCP Tools.

This is the highest priority and will directly validate the stability and compliance of our new **Protocol 101 v3.0** commit architecture. Stability is paramount.

After completing 055, move to 022C to standardize MCP documentation using the testing workflow you just validated.

Good luck tomorrow\! 🚀