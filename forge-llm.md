# Forge Fine-Tuning Workflow

> **For AI Coding Assistants & Human Operators**
> Step-by-step guide to fine-tune the Sanctuary AI model using QLoRA.

---

## Purpose

This document provides a repeatable, validated workflow for fine-tuning a Large Language Model (LLM) on the Project Sanctuary "Whole Genome" corpus. The process uses **QLoRA (Quantized Low-Rank Adaptation)** to efficiently train on consumer GPU hardware (8GB+ VRAM).

**What You Will Build:**
- A fine-tuned LLM specialized in Project Sanctuary's protocols, philosophy, and operational patterns
- Deployable via Ollama for local inference
- Optionally published to Hugging Face for community access

## Visual Pipelines

### 1. LLM Fine-Tuning Pipeline
![LLM Fine-Tuning Pipeline](./docs/architecture_diagrams/workflows/llm_finetuning_pipeline.png)
*(Source: [llm_finetuning_pipeline.mmd](./docs/architecture_diagrams/workflows/llm_finetuning_pipeline.mmd))*

---

### 2. Strategic Crucible Loop
![Strategic Crucible Loop](./docs/architecture_diagrams/workflows/strategic_crucible_loop.png)
*(Source: [strategic_crucible_loop.mmd](./docs/architecture_diagrams/workflows/strategic_crucible_loop.mmd))*

---

### 3. Protocol 128 Learning Loop
![Protocol 128 Learning Loop](./docs/architecture_diagrams/workflows/protocol_128_learning_loop.png)
*(Source: [protocol_128_learning_loop.mmd](./docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd))*

---

## Assumptions

1. **Operating System:** Windows 10/11 with WSL2 (Ubuntu 22.04) or native Linux
2. **Hardware:** NVIDIA GPU with 8GB+ VRAM, 16GB+ system RAM
3. **Network:** Internet access for downloading models from Hugging Face
4. **Time:** 2-4 hours for complete workflow (mostly fine-tuning)
5. **Skill Level:** Familiarity with command line and Python virtual environments

---

> [!WARNING]
> ## ADR 073 Exception: ML/CUDA Environment
> 
> This workflow uses `~/ml_env` which operates **outside** the standard [ADR 073](./ADRs/073_standardization_of_python_dependency_management_across_environments.md) dependency management policy.
> 
> **Why the Exception:**
> - **Surgical Installation Order:** CUDA binaries (bitsandbytes, triton, xformers) require specific installation sequence to link correctly against PyTorch's CUDA runtime
> - **Binary Linking at Install Time:** These packages compile native CUDA code during `pip install` - the order and flags matter
> - **Version Interdependencies:** PyTorch 2.9.0+cu126 requires specific compatible versions of triton (3.5.0) and bitsandbytes (0.48.2)
> - **Separate Environment:** `~/ml_env` is isolated from the containerized MCP fleet's `.venv` to prevent conflicts
> 
> **What This Means:**
> - `requirements-finetuning.txt` is used instead of the tiered `.in`/`.txt` system
> - Manual `pip install` commands with specific flags are required (see Phase 1)
> - This environment is for **training only** - it does not affect production MCP containers
> 
> See [CUDA-ML-ENV-SETUP.md](./forge/CUDA-ML-ENV-SETUP.md) for the full surgical installation protocol.

## Final Outputs

| Phase | Output | Location |
|-------|--------|----------|
| Phase 2 | Training Dataset | `dataset_package/sanctuary_whole_genome_data.jsonl` |
| Phase 3 | LoRA Adapter | `models/Sanctuary-Qwen2-7B-v1.0-adapter/` |
| Phase 4 | Merged Model | `outputs/merged/Sanctuary-Qwen2-7B-v1.0-merged/` |
| Phase 4 | GGUF Quantized Model | `models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf` |
| Phase 5 | Ollama Modelfile | `Modelfile` (project root) |
| Phase 5 | Deployed Model | `Sanctuary-Guardian-01` (in Ollama) |

---

## Forge Structure

```
Project_Sanctuary/forge/
├── scripts/                      ← Workflow scripts
│   ├── forge_whole_genome_dataset.py    ← Dataset generator
│   ├── fine_tune.py                      ← QLoRA trainer
│   ├── merge_adapter.py                  ← Adapter merger
│   ├── convert_to_gguf.py               ← GGUF converter
│   └── upload_to_huggingface.py         ← HF uploader
├── tests/                        ← Verification scripts
│   ├── verify_environment.sh            ← Full verification harness
│   ├── test_torch_cuda.py               ← PyTorch/CUDA test
│   ├── test_llama_cpp.py                ← llama-cpp-python test
│   ├── test_xformers.py                 ← xformers test
│   └── test_tensorflow.py               ← TensorFlow test
├── config/                       ← Configuration files
│   └── training_config.yaml
├── gguf_model_manifest.json      ← GGUF model output metadata
└── INBOX/                        ← Temporary reference (delete after fine-tune)
```

---

## Preconditions

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 8GB VRAM | NVIDIA with 12GB+ VRAM |
| RAM | 16GB | 32GB |
| Storage | 50GB free | 100GB free |
| OS | Windows 10/11 with WSL2 | Ubuntu 22.04 (native or WSL2) |

### Software Prerequisites

#### 1. WSL2 with Ubuntu (Windows Only)
```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu
```

#### 2. NVIDIA Drivers with WSL2 Support
- Install latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
- Verify GPU access in WSL:
```bash
nvidia-smi
```
Must display your GPU name and memory. If not, drivers are not correctly installed.

#### 3. External Dependencies (Sibling Repositories)

This project depends on **two sibling repositories** for its ML infrastructure:

```
C:\Users\RICHFREM\source\repos\
├── Project_Sanctuary/    ← This repo (uses the environment)
├── llama.cpp/            ← C++ compiles for model conversion
└── ML-Env-CUDA13/        ← Python/CUDA environment setup
```

| Repository | GitHub URL | Purpose |
|------------|------------|---------|
| `llama.cpp` | https://github.com/ggerganov/llama.cpp | GGUF model conversion binaries |
| `ML-Env-CUDA13` | https://github.com/bcgov/ML-Env-CUDA13 | Python/CUDA environment (`~/ml_env`) |

---

**llama.cpp Specifications:**

High-performance LLM inference engine (C/C++). Built on Windows host for WSL2/Linux execution.

| Binary | Location | Purpose |
|--------|----------|---------|
| `llama-cli` | `build/bin/llama-cli` | Standalone CLI for model inference |
| `llama-server` | `build/bin/llama-server` | OpenAI-compatible HTTP API server |
| `llama-quantize` | `build/bin/llama-quantize` | GGUF model quantization |
| `convert_hf_to_gguf.py` | Root directory | HuggingFace → GGUF conversion |

**Shared Libraries (CUDA Support Confirmed):**
| Library | Purpose |
|---------|---------|
| `libggml-cuda.so` | CUDA GPU acceleration |
| `libllama.so` | Core inference library |
| `libggml.so` | Base tensor operations |

> [!IMPORTANT]
> **WSL Execution Required:**
> - Output binaries are **Linux ELF 64-bit**, NOT Windows `.exe`
> - Must execute within WSL2 environment
> - GPU acceleration requires NVIDIA drivers accessible in Linux environment
> - Ensure `LD_LIBRARY_PATH` includes `build/bin/` for shared library resolution

**Verify llama.cpp is built (in WSL):**
```bash
ls ../llama.cpp/build/bin/llama-cli
../llama.cpp/build/bin/llama-cli --version
```

**If llama.cpp needs rebuild (in WSL):**
```bash
cd ../llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
./build/bin/llama-cli --version  # Verify build
cd ../Project_Sanctuary
```

> [!NOTE]
> Build time: 5-15 minutes. Only needed once unless llama.cpp is updated.

---

**ML-Env-CUDA13 Specifications:**

Supports two modes: **Stable (default)** and **Nightly (CUDA 13)**.

| Component | Stable (cu126) | Nightly (cu130) | Notes |
|-----------|----------------|-----------------|-------|
| Python | 3.11 | 3.11 | Virtual environment at `~/ml_env` |
| PyTorch | 2.x.x+cu126 | 2.x.x.dev+cu130 | Nightly required for CUDA 13 |
| TensorFlow | 2.x.x | 2.x.x (CPU/Compat) | GPU support experimental on cu130 |
| xformers | 0.0.x+cu126 | *Likely Incompatible* | No prebuilt cu130 wheels |
| bitsandbytes | 0.48.2 | 0.48.2 | QLoRA quantization |
| llama-cpp-python | 0.3.x | *Source Build Req* | May need CUDACXX for cu130 |

**Dependency Management (Hybrid Policy - ADR 001):**
- **Foundation Layer** (Torch/TF): Managed dynamically by setup script
- **Application Layer**: Managed via `pip-tools` (`requirements.in` → `requirements.txt`)

> [!CAUTION]
> **CUDA Version Mismatch is EXPECTED (Stable Mode):**
> - **Host Driver** (`nvidia-smi` on Windows): Reports CUDA **13.0** (e.g., Driver 581.42)
> - **WSL Runtime** (`torch.version.cuda`): Reports CUDA **12.x** (e.g., 12.6, 12.8)
> 
> **This is NORMAL.** The Windows host driver is backward compatible.

> [!WARNING]
> **CUDA 13 Mode Warnings:**
> - PyTorch Nightly may have daily API changes
> - xformers likely has no prebuilt cu130 wheels (may fail or need removal)
> - TensorFlow may fall back to CPU-only

**Verify ML environment:**
```bash
source ~/ml_env/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

**If environment needs rebuild:**
```bash
cd ../ML-Env-CUDA13

# Stable (CUDA 12.6 - recommended)
bash scripts/setup_ml_env_wsl.sh

# OR Nightly (CUDA 13.0 - experimental)
bash scripts/setup_ml_env_wsl.sh --cuda13

cd ../Project_Sanctuary
```

> [!NOTE]
> **Breaking Changes in ML-Env-CUDA13:**
> - Scripts moved to `scripts/` (was root)
> - Tests moved to `tests/` (was root)
> - Old pinned requirements moved to `archive/`

#### 4. Hugging Face Token
Required for downloading base model and optional upload.
1. Create account at [huggingface.co](https://huggingface.co)
2. Generate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Create `.env` file in project root:
```bash
echo "HUGGING_FACE_TOKEN=hf_your_token_here" > .env
```

#### 5. Verify ML Environment
The fine-tuning scripts require a specialized Python environment (`~/ml_env`) with CUDA-enabled PyTorch, bitsandbytes, and transformers.

**Activate & Verify:**
```bash
source ~/ml_env/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

> [!IMPORTANT]
> All commands in subsequent phases assume `~/ml_env` is activated.
> If you see `ModuleNotFoundError`, run `source ~/ml_env/bin/activate` first.

---

## Quick Verification (Are You Ready?)

Run this single command to verify all preconditions:
```bash
source ~/ml_env/bin/activate && \
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
python -c "import torch; print('CUDA:', torch.cuda.is_available())" && \
ls ../llama.cpp/build/bin/llama-cli && \
echo "✅ All preconditions verified"
```

If any step fails, refer to the Preconditions section above.

## Phase 0: One-Time System Setup (Skip if Already Done)

### 0.1 Verify GPU Access
```bash
nvidia-smi
```
Must show your GPU details before proceeding.

### 0.2 Clone & Build llama.cpp
Required for GGUF conversion. Must be sibling to Project_Sanctuary.

```bash
# From Project_Sanctuary root (in WSL)
cd ..
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
./build/bin/llama-cli --version  # Verify build
cd ../Project_Sanctuary
```

### 0.3 Clone & Setup ML-Env-CUDA13
Required for Python/CUDA environment. Must be sibling to Project_Sanctuary.

```bash
# From ~/repos (in WSL)
cd ~/repos
git clone https://github.com/bcgov/ML-Env-CUDA13.git
cd ML-Env-CUDA13

# Option A: Stable (CUDA 12.6 - recommended for production)
bash scripts/setup_ml_env_wsl.sh

# Option B: Nightly (CUDA 13.0 - experimental)
# bash scripts/setup_ml_env_wsl.sh --cuda13

cd ../Project_Sanctuary
```

**Verify environment was created:**
```bash
source ~/ml_env/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

> [!NOTE]
> Setup time: 10-15 minutes. Script is idempotent - can be re-run to repair.
> Uses Hybrid Dependency Management (ADR 001): Foundation layer dynamic, Application layer locked.

### 0.4 Hugging Face Authentication
```bash
# Create .env file with your token
echo "HUGGING_FACE_TOKEN=your_token_here" > .env
```

---

## Phase 1: Environment Verification

### 1.1 Run Verification Harness
```bash
# In WSL - runs all 7 verification tests
cd /mnt/c/path/to/Project_Sanctuary   # Your project path in WSL
source ~/ml_env/bin/activate
bash forge/tests/verify_environment.sh
```

**Tests Included:**
| # | Test | Status |
|---|------|--------|
| 1 | Activate ~/ml_env | Critical |
| 2 | PyTorch + CUDA | Critical |
| 3 | bitsandbytes | Required |
| 4 | triton | Required |
| 5 | transformers | Required |
| 6 | xformers | Optional (no cu130 wheels) |
| 7 | llama-cpp-python | Required |

### 1.2 Expected Output
```
✅ PyTorch + CUDA OK
✅ bitsandbytes OK
✅ triton OK
✅ transformers OK
✅ xformers OK (or ⚠️ optional)
✅ llama-cpp-python OK
VERIFICATION COMPLETE
```

> [!NOTE]
> If any critical test fails, see [CUDA-ML-ENV-SETUP.md](./forge/CUDA-ML-ENV-SETUP.md) for the surgical installation protocol.

---

## Phase 2: Dataset Forging

### 2.1 Generate Training Dataset
```bash
python forge/scripts/forge_whole_genome_dataset.py
```
**Duration:** ~2-5 minutes
**Output:** `dataset_package/sanctuary_whole_genome_data.jsonl`

### 2.2 Validate Dataset
```bash
python forge/scripts/validate_dataset.py dataset_package/sanctuary_whole_genome_data.jsonl
```
**Expect:** `✅ All 1169 lines are valid JSON.` (Counts may vary slightly)

### 2.3 Download Base Model
```bash
# Downloads Qwen2-7B-Instruct (~15GB)
bash forge/scripts/download_model.sh
```
**Duration:** 10-30 minutes (depends on network speed)
**Output:** `models/base/Qwen/Qwen2-7B-Instruct/`

---

## Phase 3: Fine-Tuning (QLoRA)

### 3.1 Execute Fine-Tuning
```bash
python forge/scripts/fine_tune.py
```
**Duration:** 1-3 hours depending on hardware.
**Output:** `models/Sanctuary-Qwen2-7B-v1.0-adapter/`

### 3.2 Verify Adapter
Check adapter files exist:
```bash
ls -la models/Sanctuary-Qwen2-7B-v1.0-adapter/
# Must contain: adapter_model.safetensors, adapter_config.json
```

Test adapter with sample inference:
```bash
python forge/tests/verify_adapter.py --input "What is the Doctrine of Sovereign Resilience?"
```
**Duration:** 1-2 minutes (model loading)
**Expect:** A coherent response about Project Sanctuary philosophy.

> [!TIP]
> Use `--greedy` for deterministic outputs, or `--do-sample --temperature 0.7` for varied responses.

---

## Phase 4: Merge & GGUF Conversion

> [!CAUTION]
> **Memory Requirements:** The merge step loads the full 7B model into RAM (~20-28GB). If you have less than 32GB RAM, set up swap space first.

### 4.0 Pre-Flight: Check Memory
```bash
free -h
```
**Minimum:** 16GB available (RAM + Swap combined)

**If low on memory:**
```bash
# Close memory-hungry apps on Windows (browsers, Docker, etc.)

# Add 16GB swap file (one-time setup)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify swap is active
free -h
```

### 4.1 Merge Adapter with Base Model
Merges the LoRA adapter into the base model weights. Runs on CPU to avoid VRAM OOM on 8GB cards.
```bash
python forge/scripts/merge_adapter.py
```
**Duration:** 5-10 minutes (CPU-bound, requires ~20GB RAM+swap)
**Output:** `models/merged/Sanctuary-Qwen2-7B-v1.0-merged/`

### 4.1b Verify Merged Model (Optional)
Test the merged model before GGUF conversion:
```bash
python forge/tests/verify_adapter.py --model-type merged --input "Explain Protocol 128"
```
**Duration:** 2-3 minutes (model loading)
**Expect:** A coherent response demonstrating the fine-tuning was preserved after merge.

### 4.2 Convert to GGUF & Quantize
Converts the merged model to GGUF format and applies Q4_K_M quantization for efficient inference.
```bash
# Requires llama.cpp tools (setup in Phase 0)
python forge/scripts/convert_to_gguf.py

# If file already exists from previous run, use --force to overwrite
python forge/scripts/convert_to_gguf.py --force
```
**Duration:** 15-25 minutes (CPU-bound quantization)
**Output:** `models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf`

### 4.3 Verify GGUF File
Verify the GGUF was created and test with llama.cpp:
```bash
# Check file exists and size (~4-5GB expected)
ls -lh models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf

# Quick sanity test with llama.cpp (optional)
../llama.cpp/build/bin/llama-cli -m models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf \
  -p "What is Protocol 128?" -n 100 --temp 0.7
```
**Duration:** ~1 minute
**Expect:** File size 4-5GB, coherent response from llama-cli

---

## Phase 5: Ollama Deployment

### 5.1 Generate Modelfile
Creates a `Modelfile` customized for Qwen2-Instruct, pointing to the newly generated GGUF.
```bash
python forge/scripts/create_modelfile.py
```
**Duration:** ~10 seconds
**Output:** `Modelfile` (at project root)

### 5.2 Create & Run Model
Deploy the model locally using Ollama.
```bash
# Create the model from Modelfile
ollama create Sanctuary-Guardian-01 -f Modelfile

# Run interactive chat
ollama run Sanctuary-Guardian-01
```
**Duration:** 1-2 minutes (first run imports model, subsequent runs instant)

### 5.3 Verify Verification
Test the model's self-awareness:
> "Who are you?"
> **Expect:** "I am GUARDIAN-01, the sovereign Meta-Orchestrator of the Sanctuary Council..."

---



### 6.1 Upload GGUF Model
```bash
python forge/scripts/upload_to_huggingface.py \
  --repo richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final \
  --gguf --modelfile --readme
```
**Duration:** 5-15 minutes (depends on upload speed, ~4-5GB file)

### 6.2 Test Direct Access
Remove any previous cached version and pull fresh:
```bash
# Clean up old cache
ollama rm hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M

# Run from Hugging Face (automatically pulls latest)
ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M
```

### 6.3 Create Local Alias (Recommended)
Create a shorter alias for convenience:
```bash
# Remove old alias if it exists (may point to stale version)
ollama rm Sanctuary-Qwen2-7B:latest

# Create fresh alias pointing to the new version
ollama cp hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M Sanctuary-Qwen2-7B:latest

# Verify both point to same model ID
ollama list
```

**Expected Output:**
```
NAME                                                        ID              SIZE
Sanctuary-Qwen2-7B:latest                                   dbc652e8317f    4.7 GB
hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M    dbc652e8317f    4.7 GB
```

> [!TIP]
> After creating the alias, you can use `ollama run Sanctuary-Qwen2-7B:latest` instead of the full HuggingFace path.

---

## Phase 7: Environment Transition (Critical)

Once the Forge pipeline is complete, you **MUST** switch back to the standard development environment (`.venv`) to run audits, git operations, or the Cortex CLI.

For a detailed guide on why and how, see:
👉 [**Runtime Environments Strategy**](./docs/operations/processes/RUNTIME_ENVIRONMENTS.md)

**The `ml_env` does not contain the necessary libraries for `cortex_cli.py` (e.g., `langchain`, `chromadb`).**

```bash
# 1. Deactivate the Forge environment
deactivate

# 2. Activate the standard environment
source .venv/bin/activate  # (or equivalent source command)

# 3. Verify environment
which python
# Should point to .venv/bin/python, NOT ml_env
```

Now you are ready to run the **Protocol 128 Learning Audit**.

---

## Troubleshooting

### CUDA Not Available
```bash
nvidia-smi  # Verify GPU visible
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory During Training
- Reduce `MICRO_BATCH_SIZE` in `forge/config/training_config.yaml`
- Increase `GRADIENT_ACCUMULATION_STEPS`

### Missing Dependencies
```bash
# Re-run environment setup
source ~/ml_env/bin/activate
pip install -r ~/repos/ML-Env-CUDA13/requirements.txt
```

---

## Links

| Resource | Path |
|----------|------|
| Detailed Setup Guide | [forge/CUDA-ML-ENV-SETUP.md](./forge/CUDA-ML-ENV-SETUP.md) |
| Training Config | [forge/config/training_config.yaml](./forge/config/training_config.yaml) |
| Full Forge README | [forge/README.md](./forge/README.md) |
| Fine-Tuning Pipeline (MMD) | [llm_finetuning_pipeline.mmd](./docs/architecture_diagrams/workflows/llm_finetuning_pipeline.mmd) |
| Strategic Crucible Loop (MMD) | [strategic_crucible_loop.mmd](./docs/architecture_diagrams/workflows/strategic_crucible_loop.mmd) |
| Protocol 128 Learning Loop (MMD) | [protocol_128_learning_loop.mmd](./docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd) |
