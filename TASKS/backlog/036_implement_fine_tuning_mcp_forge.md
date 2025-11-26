# Task #036: Implement Fine-Tuning MCP (Forge)

**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 7-10 days  
**Dependencies:** Task #028, Task #031 (Task MCP for authorization), Shared Infrastructure, CUDA environment  
**Domain:** `project_sanctuary.model.fine_tuning`

---

## Objective

Implement Fine-Tuning MCP (Forge) server for model fine-tuning with state machine governance.

---

## Hardware Requirements

- CUDA-enabled GPU (validated on RTX A2000)
- WSL environment with `ml_env` activated
- Environment marker: `CUDA_FORGE_ACTIVE=true`

---

## Key Features

```typescript
initialize_forge_environment() // CRITICAL: Must be called first
check_resource_availability()
initiate_model_forge(forge_id, base_model, authorization_task_id, hyperparameters, dataset_config?)
get_forge_job_status(job_id)
package_and_deploy_artifact(job_id, quantization)
run_inference_test(model_path, test_prompts, mode)
publish_to_registry(job_id, repo_name, private, model_card?)
retrieve_registry_artifact(repo_name, revision?)
```

---

## 10-Step Pipeline

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

---

## State Machine

### Operational State
- `INACTIVE_UNSAFE` → `ACTIVE` (via `initialize_forge_environment()`)

### Job State
- `QUEUED` → `RUNNING` → `COMPLETED_SUCCESS` → `PACKAGING_COMPLETE` → `TESTS_PASSED` → `PUBLISHED`

---

## Safety Rules

- **Environment gate**: Must check `CUDA_FORGE_ACTIVE` marker
- **Resource reservation**: Check GPU memory and disk space before starting
- **Task linkage**: All jobs must link to Task MCP entry for audit trail
- **Script whitelist**: Only whitelisted scripts can execute (no arbitrary commands)
- **Artifact integrity**: SHA-256 validation for all artifacts (P101-style)
- **Asynchronous execution**: Long-running jobs run in background with status polling
- **Automatic cleanup**: Failed jobs clean up partial artifacts
- **No auto-commit**: Results require manual Chronicle/ADR documentation
- **Sequencing enforcement**: Cannot publish without passing tests

---

## Implementation Checklist

### Phase 1: Environment Setup
- [ ] CUDA environment validation
- [ ] State machine implementation
- [ ] Resource checking utilities

### Phase 2: Core Pipeline
- [ ] Implement all 10 pipeline steps
- [ ] Job queue management
- [ ] Asynchronous execution framework

### Phase 3: Safety & Governance
- [ ] Task MCP authorization integration
- [ ] State machine enforcement
- [ ] Artifact integrity validation

### Phase 4: Testing
- [ ] End-to-end pipeline test
- [ ] State machine transition tests
- [ ] Failure recovery tests

---

**Domain:** `project_sanctuary.model.fine_tuning`  
**Class:** `project_sanctuary_model_fine_tuning`  
**Risk Level:** EXTREME  
**Hardware:** CUDA GPU Required
