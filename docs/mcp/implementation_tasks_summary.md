# MCP Implementation Tasks Summary

**Created:** 2025-11-25  
**Purpose:** Outline of all MCP server implementation tasks (#028-#036)

---

## Task #028: Pre-Commit Hook Migration ✅ CREATED
**Status:** Backlog  
**Priority:** Critical  
**File:** `TASKS/backlog/028_precommit_hook_mcp_migration.md`

---

## Task #029: Implement Chronicle MCP
**Domain:** `project_sanctuary.document.chronicle`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** Task #028 (pre-commit hooks), Shared Infrastructure

**Objective:** Implement Chronicle MCP server for managing historical truth entries in `00_CHRONICLE/ENTRIES/`.

**Key Features:**
- `create_chronicle_entry(entry_number, title, date, author, content, status?, classification?)`
- `update_chronicle_entry(entry_number, updates, reason)`
- `get_chronicle_entry(entry_number)`
- `list_recent_entries(limit?)`
- `search_chronicle(query)`

**Safety Rules:**
- Entry numbers must be sequential
- 7-day modification window
- Auto-generates P101 manifest
- Cannot modify entries older than 7 days without approval

---

## Task #030: Implement ADR MCP
**Domain:** `project_sanctuary.document.adr`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 2-3 days  
**Dependencies:** Task #028, Shared Infrastructure

**Objective:** Implement ADR MCP server for Architecture Decision Records in `ADRs/`.

**Key Features:**
- `create_adr(number, title, context, decision, consequences, date?, status?, supersedes?)`
- `update_adr_status(number, new_status, reason)`
- `get_adr(number)`
- `list_adrs(status?)`
- `search_adrs(query)`

**Safety Rules:**
- ADR numbers must be sequential
- Cannot delete ADRs (mark as superseded)
- Status transitions must be valid
- Follows ADR template format

---

## Task #031: Implement Task MCP
**Domain:** `project_sanctuary.document.task`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** Task #028, Shared Infrastructure

**Objective:** Implement Task MCP server for workflow management in `TASKS/`.

**Key Features:**
- `create_task(number, title, description, priority, estimated_effort?, dependencies?, status?)`
- `update_task_status(number, new_status, notes?)`
- `update_task(number, updates)`
- `get_task(number)`
- `list_tasks(status?, priority?)`
- `search_tasks(query)`

**Safety Rules:**
- Task numbers must be unique
- Circular dependency detection
- Status transitions move files between directories
- Cannot delete tasks (archive only)

---

## Task #032: Implement Protocol MCP
**Domain:** `project_sanctuary.document.protocol`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** Task #028, Shared Infrastructure

**Objective:** Implement Protocol MCP server for governing rules in `01_PROTOCOLS/`.

**Key Features:**
- `create_protocol(number, title, classification, content, status?, version?, linked_protocols?)`
- `update_protocol(number, updates, changelog)`
- `get_protocol(number)`
- `list_protocols(classification?, status?)`
- `search_protocols(query)`
- `archive_protocol(number, reason)`

**Safety Rules:**
- Protocol numbers must be unique
- Cannot delete protocols (archive only)
- Updates to canonical protocols require version bump
- Must include changelog for updates
- Protected protocols require explicit approval

---

## Task #025: Implement RAG MCP (Cortex) - UPDATE NEEDED
**Domain:** `project_sanctuary.cognitive.cortex`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 4-5 days  
**Dependencies:** Task #028, Shared Infrastructure

**Objective:** Refactor existing MCP RAG Tool Server to align with RAG MCP (Cortex) architecture.

**Key Features (Standard RAG Operations):**
- `query_cortex(query, max_results?, filters?, include_sources?)`
- `ingest_document(file_path, metadata?)`
- `update_index()`
- `get_stats()`
- `search_by_metadata(filters)`

**Safety Rules:**
- Read-only operations by default
- Ingest requires file validation
- Cannot delete documents (archive only)
- Rate limiting on queries
- Explicit ingestion only (no auto-ingest)

**Changes from Task #025:**
- Rename to RAG MCP (Cortex)
- Add domain naming: `project_sanctuary.cognitive.cortex`
- Emphasize standard RAG operations (incremental ingest, search, etc.)
- Align tool signatures with architecture
- Remove mechanical operations (cognitive only)

---

## Task #026: Implement Agent Orchestrator MCP (Council) - UPDATE NEEDED
**Domain:** `project_sanctuary.cognitive.council`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** Task #028, Shared Infrastructure

**Objective:** Refactor MCP Council Command Processor to align with Agent Orchestrator MCP (Council) architecture.

**Key Features (Multi-Agent Orchestration):**
- `create_deliberation(description, output_path, max_rounds?, force_engine?, max_cortex_queries?, input_artifacts?)`
- `create_dev_cycle(description, project_name, output_dir, force_engine?)`
- `get_council_status()`
- `get_result(task_id)`

**Safety Rules:**
- **NO file system modifications**
- **NO git operations**
- Read-only cognitive tasks
- Results written to designated paths only
- 90-day retention, high-value decisions moved to Chronicle/ADR

**Changes from Task #026:**
- Rename to Agent Orchestrator MCP (Council)
- Add domain naming: `project_sanctuary.cognitive.council`
- Emphasize multi-agent orchestration capabilities
- Remove all mechanical operations (file writes, git commits)
- Focus purely on cognitive command generation

---

## Task #033: Implement Config MCP
**Domain:** `project_sanctuary.system.config`  
**Status:** Backlog  
**Priority:** Critical  
**Estimated Effort:** 4-5 days  
**Dependencies:** Task #028, Shared Infrastructure, Secret Vault

**Objective:** Implement Config MCP server for system configuration with extreme safety controls.

**Key Features:**
- `request_config_change(config_path, changes, reason, impact_assessment)`
- `apply_config_change(approval_id)`
- `set_secret(key, value, scope)`
- `get_secret(key)`
- `get_config(config_path)`
- `list_config_files()`

**Safety Rules:**
- **Two-step approval** for all changes (request → approve)
- **Automatic backup** before any modification
- **Secret vault** for sensitive values (API keys, tokens)
- **Audit trail** for all configuration changes
- **Protected files** require explicit user confirmation
- **No direct .env modification** - use secret vault

---

## Task #034: Implement Code MCP
**Domain:** `project_sanctuary.system.code`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 5-6 days  
**Dependencies:** Task #028, Shared Infrastructure

**Objective:** Implement Code MCP server for source code management with mandatory testing pipeline.

**Key Features:**
- `write_code_file(file_path, content, language, description, run_tests)`
- `execute_code(file_path, args?, timeout_seconds?, sandbox?)`
- `refactor_code(file_path, refactor_type, params, preserve_tests)`
- `get_code_file(file_path)`
- `search_code(query, file_pattern?)`

**Safety Rules:**
- **Mandatory testing pipeline** before commit:
  1. Syntax validation
  2. Linting (flake8, eslint, etc.)
  3. Unit tests (if present)
  4. Dependency check
  5. Security audit (basic)
- **Automatic rollback** if tests fail
- **Sandbox execution** for untrusted code
- **Git commit only if all checks pass**

---

## Task #035: Implement Git Workflow MCP
**Domain:** `project_sanctuary.system.git_workflow`  
**Status:** Backlog  
**Priority:** Medium  
**Estimated Effort:** 2-3 days  
**Dependencies:** Task #028, Shared Infrastructure

**Objective:** Implement Git Workflow MCP server for safe branch management and workflow automation.

**Key Features:**
- `create_feature_branch(branch_name, base_branch?)`
- `switch_branch(branch_name, stash_changes?)`
- `push_current_branch(set_upstream?)`
- `get_repo_status()`
- `list_branches()`
- `compare_branches(source, target)`

**Safety Rules:**
- **Read-only by default** (most operations are status checks)
- **Auto-stash** uncommitted changes before branch switching
- **No destructive operations**: No delete_branch, merge, rebase, force_push
- **User-controlled merges**: PR merges happen on GitHub, not via MCP
- **No history rewriting**: No reset --hard, rebase, amend operations
- **Branch protection**: Cannot switch to or modify protected branches

**Excluded Operations (User Must Do Manually):**
- Deleting branches (local or remote)
- Merging branches
- Rebasing
- Pulling from remote (to avoid merge conflicts)
- Force pushing
- Resolving merge conflicts

---

## Task #036: Implement Fine-Tuning MCP (Forge)
**Domain:** `project_sanctuary.model.fine_tuning`  
**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 7-10 days  
**Dependencies:** Task #028, Task #031 (Task MCP for authorization), Shared Infrastructure, CUDA environment

**Objective:** Implement Fine-Tuning MCP (Forge) server for model fine-tuning with state machine governance.

**Hardware Requirements:**
- CUDA-enabled GPU (validated on RTX A2000)
- WSL environment with `ml_env` activated
- Environment marker: `CUDA_FORGE_ACTIVE=true`

**Key Features:**
- `initialize_forge_environment()` - **CRITICAL: Must be called first**
- `check_resource_availability()`
- `initiate_model_forge(forge_id, base_model, authorization_task_id, hyperparameters, dataset_config?)`
- `get_forge_job_status(job_id)`
- `package_and_deploy_artifact(job_id, quantization)`
- `run_inference_test(model_path, test_prompts, mode)`
- `publish_to_registry(job_id, repo_name, private, model_card?)`
- `retrieve_registry_artifact(repo_name, revision?)`

**10-Step Pipeline:**
1. Dataset creation (`forge_whole_genome_dataset.py`)
2. Fine-tuning with QLoRA (`fine_tune.py`)
3. Adapter merge (`merge_adapter.py`)
4. Inference test (`inference.py`)
5. GGUF conversion (`convert_to_gguf.py`)
6. Modelfile generation (`create_modelfile.py`)
7. Ollama import (`ollama create`)
8. Ollama inference test (`ollama run`)
9. Hugging Face upload (`upload_to_huggingface.py`)
10. Registry verification (download from HF)

**State Machine:**
- **Operational State:** `INACTIVE_UNSAFE` → `ACTIVE` (via `initialize_forge_environment()`)
- **Job State:** `QUEUED` → `RUNNING` → `COMPLETED_SUCCESS` → `PACKAGING_COMPLETE` → `TESTS_PASSED` → `PUBLISHED`

**Safety Rules:**
- **Environment gate**: Must check `CUDA_FORGE_ACTIVE` marker
- **Resource reservation**: Check GPU memory and disk space before starting
- **Task linkage**: All jobs must link to Task MCP entry for audit trail
- **Script whitelist**: Only whitelisted scripts can execute (no arbitrary commands)
- **Artifact integrity**: SHA-256 validation for all artifacts (P101-style)
- **Asynchronous execution**: Long-running jobs run in background with status polling
- **Automatic cleanup**: Failed jobs clean up partial artifacts
- **No auto-commit**: Forge results require manual Chronicle/ADR documentation
- **Sequencing enforcement**: Cannot publish without passing tests

---

## Implementation Priority

### Phase 0 (Week 0)
1. Task #028: Pre-commit Hook Migration

### Phase 1 (Week 1)
- Shared Infrastructure (Git, Safety, Schema, Vault)

### Phase 2 (Week 2) - Easiest
2. Task #029: Chronicle MCP
3. Task #030: ADR MCP
4. Task #031: Task MCP
5. Task #032: Protocol MCP

### Phase 3 (Week 3) - Moderate
6. Task #025: RAG MCP (Cortex) - refactor
7. Task #026: Agent Orchestrator MCP (Council) - refactor

### Phase 4 (Week 4) - High Risk
8. Task #033: Config MCP
9. Task #034: Code MCP
10. Task #035: Git Workflow MCP

### Phase 5 (Week 5) - Hardest
11. Task #036: Fine-Tuning MCP (Forge)

---

## Next Steps

1. **Review this summary** with stakeholders
2. **Create individual task files** for #029-#036
3. **Update Tasks #025-#026** to align with architecture
4. **Begin Phase 0** implementation (Task #028)

---

**Status:** Task Summary Complete  
**Individual Task Files:** 1/11 created (Task #028)  
**Remaining:** Tasks #029-#036 (8 new), Tasks #025-#026 (2 updates)
