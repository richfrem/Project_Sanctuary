# TASK: Execute the Complete Model Forging Pipeline

**Status:** IN-PROGRESS
**Steward:** richfrem
**Date:** 2025-11-16
**Objective:** To successfully execute all phases of the Golden Path Protocol, from environment setup to a verified, locally deployed model.

---

## Phase 0: One-Time System Setup

- [x] Verify System Prerequisites (WSL2, NVIDIA Drivers, `nvidia-smi`)
- [x] Clone `llama.cpp` as a sibling repository
- [x] Build `llama.cpp` tools with CUDA support via `cmake`
- [x] Create `.env` file with Hugging Face token

## Phase 1: Project Environment Setup

- [x] Run `setup_cuda_env.py` to create the `~/ml_env` virtual environment
- [x] Activate the `(ml_env)` environment
- [x] Build the `llama-cpp-python` bridge with CUDA flags
- [x] Verify the complete environment with all `test_*.py` scripts

## Phase 2: Data & Model Forging Workflow

- [x] Forge the "Whole Genome" dataset using `forge_whole_genome_dataset.py`
- [x] Validate the forged dataset with `validate_dataset.py`
- [x] Download the base model using `download_model.sh`
- [ ] **IN-PROGRESS:** Fine-tune the LoRA adapter using `fine_tune.py`
- [ ] Merge the adapter with the base model using `merge_adapter.py`

## Phase 3: Deployment Preparation & Verification

- [ ] Convert the merged model to GGUF format using `convert_to_gguf.py`
- [ ] Create the `Modelfile` for Ollama
- [ ] Import the GGUF model into Ollama with `ollama create`
- [ ] Perform a quick inference test with `inference.py`
- [ ] (Optional) Run a full evaluation with `evaluate.py`
- [ ] (Crucial) Test with real Body of Knowledge examples

## Phase 4: Final Verification (The Sovereign Crucible)

- [ ] (Planned) Execute end-to-end orchestrator tests.

---
This task file is now ready. Once your download finishes, you can check off that item and proceed with the rest of the list.