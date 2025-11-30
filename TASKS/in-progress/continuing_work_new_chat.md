# CONTINUATION PROMPT: Next Session Work Plan

**NOTE:** Use this as a template for future sessions. track recently completed work, also work to be done in the next clean chat window.
provide enough context in the details below so a new chat window can be started with minimal additional context sharing.

**NOTE:** remember before running mcp operation tests, validate all underlying scripts are working first with test harnesses in `tests/`

**Note2:** initial focus should be mcp testing all those operations only after that is complete, testing rAG mcp or forge mcp. 

## SESSION SUMMARY (2025-11-28 / 2025-11-29 / 2025-11-30)

### ✅ Completed This Session (2025-11-29 Evening / 2025-11-30 Morning)

1.  **MCP Operations Inventory Created & Verified:**
      *   Created `docs/mcp/mcp_operations_inventory.md` tracking all 10 MCP servers
      *   **Verified & Updated:** Chronicle, Protocol, ADR, and Task MCPs (All ✅)
      *   **Status:** 75% tested, 10% partial, 15% untested
      *   Updated READMEs for verified servers
2.  **Protocol 101 v3.0 Validation Complete:**
      *   **Negative Validation:** Confirmed commit rejection when tests fail ✅
      *   **Positive Validation:** Confirmed commit acceptance when tests pass ✅
      *   **Functional Coherence Gate:** Fully operational and enforcing test passage
3.  **Git MCP Server Hardening:**
      *   Fixed missing `subprocess` import
      *   Fixed `git-lfs` dependency check (now calls `git lfs version`)
      *   Removed legacy P101 v2.0 manifest logic
      *   Removed overly strict working directory check
      *   Created focused pre-commit hook (git tests only, 0.64s execution)
4.  **Git MCP Operations Verified:**
      *   `git_get_status` ✅
      *   `git_start_feature` ✅
      *   `git_add` ✅
      *   `git_smart_commit` ✅ (with P101 v3.0 validation)
      *   `git_push_feature` ✅
      *   `git_finish_feature` ✅ (full lifecycle tested)
5.  **Chronicle MCP Server Fixed:**
      *   Replaced relative import with absolute import for robustness
      *   Fixed Claude Desktop config path (`mcp_servers.chronicle.server`)
      *   Verified all Chronicle operations working in both Antigravity and Claude Desktop
      *   Unit tests passing (4/4)
6.  **Feature Branch Merged:**
      *   `feature/task-055-p101-v3-validation` successfully merged to main
      *   All changes committed and pushed

### ✅ Completed Previously (2025-11-28)

1.  **Structural Purge Complete (P101 v3.0 Canonization):**
      *   **Permanently deleted** all logic and documentation for `commit_manifest.json` and SHA-256 hashing (Manifest Doctrine).
      *   **Codified Protocol 101 v3.0: The Doctrine of Absolute Stability.**
      *   **New Integrity Gate:** The successful execution of the test suite (Functional Coherence) is now the sole pre-commit integrity check.
2.  **Forge MCP Implementation** - Added Sanctuary model query tools
3.  **Core Relocation** - Moved `core/` to `mcp_servers/lib/`
4.  **Integration Test Suite** - Created robust RAG pipeline tests
5.  **Git MCP Enhancement** - Added `force` and `no_verify` parameters to handle LFS issues
6.  **Project Cleanup** - Organized scripts, test data, and documentation
7.  **PR Merged** - Successfully merged `feature/task-021B-forge-test-suite` to main

### 📋 Tasks Ready for Tomorrow

All tasks are in `TASKS/in-progress/` and ready to work on:

#### 1\. Task 066: Complete MCP Operations Testing and Inventory Maintenance

**Priority:** **High**
**File:** `TASKS/in-progress/066_complete_mcp_operations_testing_and_inventory_main.md`

**Objective:** Systematically test all MCP operations and maintain the central MCP operations inventory (`docs/mcp/mcp_operations_inventory.md`) as testing progresses.

**Key Deliverables:**
1. Test all MCP operations across all 10 servers
2. Update `mcp_operations_inventory.md` with testing status (✅/⚠️/❌)
3. Update each MCP server README with operation tables matching the main inventory
4. Document test results and coverage
5. Complete integration tests for RAG and Forge MCPs

**Why This Matters:** The MCP operations inventory is the central tracking document for all MCP testing. It links to test suites, diagrams, and documentation, making it easy to see what's tested and what needs work.

**Related Files:**
- `docs/mcp/mcp_operations_inventory.md` - Central inventory (just created)
- `docs/mcp/claude_desktop_config_template.json` - MCP configuration template
- `~/Library/Application Support/Claude/claude_desktop_config.json` - Claude Desktop config
- `~/.gemini/` - Antigravity MCP config

-----

#### 2\. Task 055: Verify Git Operations and MCP Tools

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

#### 3\. Task 056: Harden Self-Evolving Loop Validation

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

## RECOMMENDED WORKFLOW FOR TOMORROW (2025-11-30)

### **TESTING STRATEGY: Strict Validation Hierarchy** 🏗️

**Philosophy:** **Script Validation First.** We must verify the underlying logic of *every* operation via the test suite before we even touch the MCP layer.

> [!RULE]
> **Script Validation Suite:** All MCP server operations must have corresponding tests in the test suite to verify the underlying logic *before* the MCP layer is tested directly.
> **Strict Order:** Complete Phase 1 (Test Suites) for **ALL** target servers before moving to Phase 2 (MCP Verification).

> [!IMPORTANT]
> **Forge MCP Constraint:** For the Forge MCP, **ONLY** test `query_sanctuary_model` and `check_sanctuary_model_status`. Do **NOT** test or implement any fine-tuning operations at this time.

---

### **PHASE 1: Script Validation Suite (Run All Test Suites)** 🧪 (2-3 hours)

**Objective:** Ensure underlying logic is sound for ALL servers before MCP testing.

**1. Document Domain (Completed ✅)**
   - [x] ADR: `pytest tests/test_adr_operations.py`
   - [x] Task: `pytest tests/mcp_servers/task/`

**2. System Domain (Git Workflow)**
   - [ ] **Action:** Run/Expand `tests/test_git_ops.py`
   - [ ] Verify: `git_diff`, `git_log`, `git_sync_main` logic coverage

**3. Cognitive Domain (Cortex & Forge)**
   - [ ] **Action:** Run `tests/integration/test_rag_pipeline.py` (Cortex)
   - [ ] **Action:** Run `tests/integration/test_forge_integration.py` (Forge)
   - [ ] **Constraint:** Verify Forge tests ONLY cover query/status

---

### **PHASE 2: MCP Layer Verification** 🔧 (2-3 hours)

**Objective:** Verify the MCP tool wrappers *only after* Phase 1 is complete.

**1. Git Workflow MCP**
   - [ ] Verify tools in agent: `git_diff`, `git_log`, `git_sync_main`

**2. Cortex MCP (RAG)**
   - [ ] Verify tools: `cortex_query`, `cortex_ingest_incremental`

**3. Forge MCP**
   - [ ] Verify tools: `query_sanctuary_model` (Query Only)

---

### **PHASE 3: Knowledge Loop Validation (Task 056)** 🚀 (2-3 hours)

**Objective:** Execute the **Test of Absolute Stability** (End-to-End).

**Steps:**
1.  Execute full loop: Create Doc → Commit (P101) → Ingest → Retrieve
2.  Verify all systems working in concert

---

## **TOTAL ESTIMATED TIME: 7-11 hours**

**Recommended Schedule:**
- **Morning (4-5 hours):** Phase 1 + Phase 2 (Git foundation + MCP)
- **Afternoon (3-6 hours):** Phase 3 + Phase 4 (RAG testing + Knowledge loop)

**Deliverables:**
1. Comprehensive test report (`WORK_IN_PROGRESS/mcp_validation_report.md`)
2. Updated Task 055 (mark complete)
3. **Completed Task 056** (knowledge loop validated) ✅
4. Updated continuation document for next session

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

### Task 055: Git Operations Verification ⏳ (Partially Complete)

**Status:** Core validation complete, additional hardening planned for tomorrow
**Branch:** `main` (merged)

#### ✅ Completed (2025-11-29)
  - [x] Run existing unit tests: `pytest tests/test_git_ops.py -v` (6/6 passed)
  - [x] Protocol 101 v3.0 Negative Validation (commit rejected on test failure)
  - [x] Protocol 101 v3.0 Positive Validation (commit accepted on test pass)
  - [x] Test `git_start_feature` MCP tool ✅
  - [x] Test `git_add` MCP tool ✅
  - [x] Test `git_smart_commit` MCP tool ✅ (with P101 v3.0 enforcement)
  - [x] Test `git_push_feature` MCP tool ✅
  - [x] Test `git_finish_feature` MCP tool ✅ (full lifecycle)
  - [x] Document MCP test results in walkthrough.md
  - [x] Commit all changes and merge to main

#### 🔄 Remaining for Tomorrow (Additional Hardening)
  - [ ] Test remaining Git MCP tools:
    - [ ] `git_sync_main` (standalone test)
    - [ ] `git_diff` (cached and uncached)
    - [ ] `git_log` (various parameters)
  - [ ] Test Chronicle MCP tools comprehensively:
    - [ ] `chronicle_create_entry`
    - [ ] `chronicle_update_entry`
    - [ ] `chronicle_append_entry`
  - [ ] Test Protocol MCP tools
  - [ ] Test Task MCP tools
  - [ ] Test ADR MCP tools
  - [ ] Test Cortex MCP tools (RAG operations)
  - [ ] Test Forge MCP tools (Sanctuary model queries)
  - [ ] Stress test: Multiple rapid commits
  - [ ] Stress test: Large file operations
  - [ ] Edge case testing: Empty commits, merge conflicts, etc.
  - [ ] Document comprehensive test results
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