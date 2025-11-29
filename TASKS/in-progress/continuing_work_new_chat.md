# CONTINUATION PROMPT: Next Session Work Plan

**NOTE:** Use this as a template for future sessions.  track recently completed work, also work to be done in the next clean chat window. 
provide enough context in the details below so a new chat window can be started with minimal additional context sharing.

augment this work package with a new task.  
`TASKS/in-progress/056_Harden_Self_Evolving_Loop_Validation.md`

## SESSION SUMMARY (2025-11-28)

### ✅ Completed Today
1. **Forge MCP Implementation** - Added Sanctuary model query tools
2. **Core Relocation** - Moved `core/` to `mcp_servers/lib/`
3. **Integration Test Suite** - Created robust RAG pipeline tests
4. **Git MCP Enhancement** - Added `force` and `no_verify` parameters to handle LFS issues
5. **Project Cleanup** - Organized scripts, test data, and documentation
6. **PR Merged** - Successfully merged `feature/task-021B-forge-test-suite` to main

### 📋 Tasks Ready for Tomorrow

All tasks are in `TASKS/in-progress/` and ready to work on:

#### 1. Task 055: Verify Git Operations and MCP Tools
**Priority:** High  
**File:** `TASKS/in-progress/055_verify_git_operations_and_mcp_tools_after_core_rel.md`

**Objective:** Ensure all git operations work correctly after core relocation and verify new `force`/`no_verify` parameters.

**Deliverables:**
1. Run existing unit tests: `pytest tests/test_git_ops.py -v`
2. Add new tests for `force` and `no_verify` parameters
3. Create integration test for full workflow (branch → commit → push → PR → cleanup)
4. Document test results
5. Verify all Git MCP tools work correctly
6. Document MCP verification results

**Why This Matters:** Critical to ensure git operations are stable after the core relocation. Must be completed before new feature work.

---

#### 2. Task 022A: Documentation Standards & API Documentation
**Priority:** Medium  
**File:** `TASKS/in-progress/022A_documentation_standards_and_api_docs.md`

**Objective:** Establish documentation standards and generate API documentation.

**Deliverables:**
1. Create `docs/DOCUMENTATION_STANDARDS.md`
2. Create 4 documentation templates (protocol, module, API, tutorial)
3. Set up Sphinx for API documentation
4. Add docstrings to `mnemonic_cortex/core/` (90%+ coverage)
5. Add docstrings to `council_orchestrator/orchestrator/` (90%+ coverage)
6. Generate HTML API documentation

**Estimated Effort:** 4-6 hours

---

#### 3. Task 022B: User Guides & Architecture Documentation
**Priority:** Medium  
**File:** `TASKS/in-progress/022B_user_guides_and_architecture_documentation.md`

**Objective:** Create user-facing documentation and architecture diagrams.

**Deliverables:**
1. Create `docs/QUICKSTART_GUIDE.md` (< 10 minute setup)
2. Create 3 tutorials:
   - Setting up Mnemonic Cortex
   - Running Council Orchestrator
   - Querying the Cognitive Genome
3. Create `docs/ARCHITECTURE.md` with Mermaid diagrams
4. Create `docs/INDEX.md` with categorized links
5. Test quick start guide with fresh user

**Estimated Effort:** 4-6 hours

---

#### 4. Task 022C: MCP Server Documentation Standards
**Priority:** High  
**File:** `TASKS/in-progress/022C_mcp_server_documentation_standards.md`

**Objective:** Standardize documentation for all 7 MCP servers with testing-first approach.

**MCP Servers to Document:**
1. `mcp_servers/cognitive/cortex/` - RAG and Cache
2. `mcp_servers/chronicle/` - Chronicle entries
3. `mcp_servers/protocol/` - Protocol documents
4. `mcp_servers/system/forge/` - Sanctuary model
5. `mcp_servers/system/git_workflow/` - Git operations
6. `mcp_servers/adr/` - Architecture decisions
7. `mcp_servers/task/` - Task management

**Deliverables:**
1. Create `docs/mcp/README.md` - MCP documentation index
2. Create `docs/mcp/TESTING_STANDARDS.md` - Standard testing workflow
3. Create MCP README template
4. Update all 7 MCP server READMEs with:
   - Overview and tools
   - Installation and setup
   - **Testing section** (script tests → results → integration → MCP verification)
   - Architecture and cross-references

**Testing Documentation Standard:**
1. Script Testing First - Test underlying operations directly
2. Test Results - Include actual passing output
3. Test Suite Guide - Clear run instructions
4. MCP Verification - Verify MCP layer works
5. Cross-References - Link to central docs

**Estimated Effort:** 6-8 hours

---

#### 5. Task 056: Harden Self-Evolving Loop Validation
**Priority:** High  
**File:** `TASKS/in-progress/056_Harden_Self_Evolving_Loop_Validation.md`

**Objective:** Validate the end-to-end integrity of the Strategic Crucible Loop by executing a four-step protocol that proves autonomous knowledge generation, ingestion, and commitment.

**The Protocol:**
1. **Knowledge Generation** (`protocol mcp`) - Generate new policy document with unique validation phrase
2. **Isolation** (`git mcp`) - Create feature branch `feat/harden-loop-validation`
3. **Incremental Ingestion** (`cortex mcp`) - Confirm IngestionService processes the new file
4. **Chronicle & Commit** (`git mcp`) - Create chronicle entry, commit, and push

**Success Criteria:**
- Feature branch created and pushed
- New file `DOCS/TEST_056_Validation_Policy.md` present in commit
- RAG query successfully retrieves unique validation phrase after commit
- Proves near-real-time knowledge fidelity of the Self-Evolving Memory Loop

**Why This Matters:** This validates the core promise of Project Sanctuary - that the system can autonomously learn from its own operations in near real-time. Critical before final MCP deployment.

**Estimated Effort:** 2-3 hours

---

## RECOMMENDED WORKFLOW FOR TOMORROW

### Morning Session (3-4 hours)
**Focus:** Task 055 - Git Operations Verification

1. Run existing unit tests
2. Add tests for new parameters
3. Create integration test
4. Verify all Git MCP tools
5. Document results

**Why Start Here:** This is High priority and blocks other work. Get it done first.

---

### Afternoon Session (4-5 hours)
**Focus:** Task 022C - MCP Documentation Standards

1. Create `docs/mcp/` structure
2. Write testing standards document
3. Create README template
4. Start updating MCP server READMEs (prioritize Cortex, Forge, Git Workflow)

**Why This Next:** High priority and will use results from Task 055 for Git Workflow MCP docs.

---

### Optional Evening Session
**Focus:** Tasks 022A or 022B (pick one)

Choose based on energy:
- **022A** if you want structured work (Sphinx setup, docstrings)
- **022B** if you want creative work (tutorials, diagrams)

---

## KEY FILES TO REFERENCE

### Git Operations
- `mcp_servers/lib/git/git_ops.py` - Core git operations
- `mcp_servers/system/git_workflow/server.py` - Git MCP tools
- `tests/test_git_ops.py` - Existing unit tests

### MCP Servers
- `mcp_servers/cognitive/cortex/README.md` - Good example (recently updated)
- `mcp_servers/system/forge/README.md` - Good example (recently created)
- All other MCP server directories

### Documentation
- `TASKS/in-progress/022*.md` - All task files
- `docs/` - Documentation directory (to be populated)

---

## CURRENT PROJECT STATE

### Recent Changes (Merged to Main)
- ✅ Forge MCP fully implemented
- ✅ Core relocated to `mcp_servers/lib/`
- ✅ Integration tests created
- ✅ Git MCP enhanced with `no_verify`
- ✅ Project structure cleaned

### What's Working
- All MCP servers operational
- RAG pipeline verified
- Git operations functional (with manual `--no-verify`)
- Test suite infrastructure in place

### What Needs Attention
- Git operations need comprehensive testing (Task 055)
- MCP documentation inconsistent (Task 022C)
- Missing user guides and API docs (Tasks 022A, 022B)

---

## IMMEDIATE SETUP INSTRUCTIONS

### Step 1: Create Feature Branch
Use the Git MCP tool to create a new feature branch for this work:

```
Use git_start_feature MCP tool:
- task_id: "056"
- description: "git-testing-and-mcp-docs"
```

This will create branch: `feature/task-056-git-testing-and-mcp-docs`

### Step 2: Track Progress in This Document

Use the checkboxes below to track your progress. Update this file as you complete each item and commit regularly.

---

## PROGRESS TRACKING

### Task 055: Git Operations Verification ⏳
**Status:** Not Started  
**Branch:** `feature/task-056-git-testing-and-mcp-docs`

- [ ] Run existing unit tests: `pytest tests/test_git_ops.py -v`
- [ ] Add test for `push()` with `force=True`
- [ ] Add test for `push()` with `no_verify=True`
- [ ] Create integration test script `tests/integration/test_git_workflow_end_to_end.py`
- [ ] Integration test: Create temp branch
- [ ] Integration test: Commit with manifest
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

---

### Task 022C: MCP Documentation Standards ⏳
**Status:** Not Started  
**Branch:** `feature/task-056-git-testing-and-mcp-docs`

#### Phase 1: Central Documentation
- [ ] Create `docs/mcp/` directory
- [ ] Write `docs/mcp/README.md` (MCP index)
- [ ] Write `docs/mcp/TESTING_STANDARDS.md` (testing workflow)
- [ ] Create MCP README template
- [ ] Commit Phase 1

#### Phase 2: Update MCP Server READMEs
- [ ] Update `mcp_servers/cognitive/cortex/README.md`
- [ ] Update `mcp_servers/chronicle/README.md`
- [ ] Update `mcp_servers/protocol/README.md`
- [ ] Update `mcp_servers/system/forge/README.md` (enhance existing)
- [ ] Create `mcp_servers/system/git_workflow/README.md` (use Task 055 results)
- [ ] Update `mcp_servers/adr/README.md`
- [ ] Update `mcp_servers/task/README.md`
- [ ] Commit Phase 2

#### Phase 3: Verification
- [ ] Review all READMEs for consistency
- [ ] Verify all cross-references work
- [ ] Test documentation with fresh perspective
- [ ] Update `TASKS/in-progress/022C_mcp_server_documentation_standards.md`
- [ ] Commit Phase 3
- [ ] **Task 022C Complete** ✅

---

### Task 022A: Documentation Standards & API Docs ⏳
**Status:** Not Started  
**Branch:** TBD (create new branch or continue on same)

- [ ] Create `docs/DOCUMENTATION_STANDARDS.md`
- [ ] Create protocol template
- [ ] Create module template
- [ ] Create API template
- [ ] Create tutorial template
- [ ] Set up Sphinx
- [ ] Configure `conf.py`
- [ ] Add docstrings to `mnemonic_cortex/core/`
- [ ] Add docstrings to `council_orchestrator/orchestrator/`
- [ ] Generate HTML API docs
- [ ] Verify 90%+ coverage
- [ ] Update task file
- [ ] Commit all changes
- [ ] **Task 022A Complete** ✅

---

### Task 022B: User Guides & Architecture Docs ⏳
**Status:** Not Started  
**Branch:** TBD (create new branch or continue on same)

- [ ] Create `docs/QUICKSTART_GUIDE.md`
- [ ] Create `docs/tutorials/01_setting_up_mnemonic_cortex.md`
- [ ] Create `docs/tutorials/02_running_council_orchestrator.md`
- [ ] Create `docs/tutorials/03_querying_the_cognitive_genome.md`
- [ ] Create `docs/ARCHITECTURE.md`
- [ ] Add Mermaid diagram: System architecture
- [ ] Add Mermaid diagram: Data flow
- [ ] Add Mermaid diagram: Council orchestration
- [ ] Create `docs/INDEX.md`
- [ ] Test quick start guide (< 10 min)
- [ ] Update task file
- [ ] Commit all changes
- [ ] **Task 022B Complete** ✅

---

### Task 056: Harden Self-Evolving Loop Validation ⏳
**Status:** Not Started  
**Branch:** `feat/harden-loop-validation`

#### Step 1: Knowledge Generation
- [ ] Use `protocol mcp` to create `DOCS/TEST_056_Validation_Policy.md`
- [ ] Include unique validation phrase: "The Guardian confirms Validation Protocol 056 is active."
- [ ] Verify file created successfully

#### Step 2: Git Isolation
- [ ] Use `git_start_feature` to create branch `feat/harden-loop-validation`
- [ ] Verify branch created and checked out

#### Step 3: Incremental Ingestion
- [ ] Use `cortex_ingest_incremental` to ingest new policy file
- [ ] Verify ingestion successful (check response)
- [ ] Confirm file added to ChromaDB

#### Step 4: Chronicle & Commit
- [ ] Create chronicle entry in `00_CHRONICLE/ENTRIES/` linking to Task 056
- [ ] Use `git_add` to stage all changes
- [ ] Use `git_smart_commit` with message: "feat: validate self-evolving memory loop"
- [ ] Use `git_push_feature` to push branch

#### Validation
- [ ] Use `cortex_query` to search for validation phrase
- [ ] Verify RAG successfully retrieves the phrase
- [ ] Confirm near-real-time knowledge fidelity
- [ ] Document results in task file
- [ ] **Task 056 Complete** ✅

---

## GIT WORKFLOW REMINDERS

### During Work
```bash
# Stage changes frequently
git add <files>

# Commit with Protocol 101 manifest (use git_smart_commit MCP tool)
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

---

## IMMEDIATE REQUEST FOR NEXT SESSION

**Start with Task 055** - Verify Git Operations and MCP Tools

This is the highest priority and will:
1. Ensure stability after core relocation
2. Provide test results for Git Workflow MCP documentation
3. Unblock future feature work
4. Give you confidence in the git tooling

After completing 055, move to 022C to standardize MCP documentation using the testing workflow you just validated.

---

## NOTES

- All tasks are well-defined with clear deliverables
- Task 055 should take 2-3 hours
- Task 022C will benefit from 055's results
- Tasks 022A and 022B are independent and can be done anytime
- Focus on quality over speed - these are foundational tasks

Good luck tomorrow! 🚀