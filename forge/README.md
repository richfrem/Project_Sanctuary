# Operation Phoenix Forge: Sovereign AI Fine-Tuning Pipeline

**Version:** 5.0 (Standardized Codebase - In-Progress Training)
**Date:** January 4, 2026
**Architect:** Antigravity (Advanced Agentic AI)
**Steward:** richfrem

**Objective:** To forge, deploy, and perform end-to-end verification of a sovereign AI model fine-tuned on the complete Project Sanctuary Cognitive Genome.

**🕒 CURRENT STATUS (Jan 2026):** The pipeline is currently executing a fresh fine-tuning run (Phoenix Forge v5.0).
- **Fine-Tuning Progress:** ✅ **100% COMPLETE** (Epoch 3.0, Duration: 1:22:48, Loss: 1.01)
- **Standardization:** Entire codebase refactored to **ADR 075** (Hybrid Documentation) and integrated with `mcp_servers.lib` utilities.
- **Primary Guide:** See **[`forge-llm.md`](../forge-llm.md)** for the updated execution protocol.

---

## 🏆 Pipeline Status: COMPLETE

**✅ All Phases Successfully Executed:**
- **Phase 1:** Environment & Data Prep - Complete
- **Phase 2:** Model Forging (QLoRA Fine-tuning) - Complete  
- **Phase 3:** Packaging & Deployment - Complete
- **Phase 4:** Verification (Sovereign Crucible) - Complete
- **Phase 5:** Public Deployment (Hugging Face) - Complete

**📦 Final Deliverables:**
- **Model:** Sanctuary-Qwen2-7B-v1.0 (GGUF format, Q4_K_M quantization)
- **Repository:** https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final
- **Direct Access:** `ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M`
- **Documentation:** Comprehensive README with dual interaction modes

---

**For the authoritative, step-by-step workflow for the v5.0 pipeline, refer to:**
- **[`forge-llm.md`](../forge-llm.md)**

**For historical environment setup context:**
- **[`CUDA-ML-ENV-SETUP.md`](./CUDA-ML-ENV-SETUP.md)**

---

## System Requirements & Prerequisites

### **Hardware Requirements**
- **GPU:** NVIDIA GPU with CUDA support (8GB+ VRAM recommended for QLoRA fine-tuning)
- **RAM:** 16GB+ system RAM
- **Storage:** 50GB+ free space for models and datasets
- **OS:** Windows 10/11 with WSL2, or Linux

### **Software Prerequisites**
- **WSL2 & Ubuntu:** For Windows users (run `wsl --install` if not installed)
- **NVIDIA Drivers:** Latest drivers with WSL2 support
- **CUDA Toolkit:** 12.6+ (automatically handled by setup script)
- **Python:** 3.11+ (automatically installed by setup script)
- **Git LFS:** For large model file handling

### **One-Time System Setup**
Before running the fine-tuning pipeline, ensure these system-level components are configured:

1.  **Verify WSL2 & GPU Access:**
    ```bash
    # In your Ubuntu on WSL terminal
    nvidia-smi
    ```
    This command *must* show your GPU details before you proceed.

2.  **Clone and Build `llama.cpp`:** This project requires the `llama.cpp` repository for converting the model to GGUF format. It must be cloned as a sibling directory to `Project_Sanctuary`.

```bash
# From the Project_Sanctuary root directory, navigate to the parent folder
cd ..

# Clone the llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git

# Enter the llama.cpp directory and build the tools with CUDA support using CMake
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Return to your project directory
cd ../Project_Sanctuary
```

---

## Project Structure & Components

```
forge/
├── README.md                           # This overview and workflow guide
├── CUDA-ML-ENV-SETUP.md               # Comprehensive environment setup protocol
├── CUDA-ML-ENV-SETUP-PASTFAILURES.md  # Historical troubleshooting reference
├── HUGGING_FACE_README.md             # Model publishing and deployment guide
├── manifest.json                      # Project metadata and version info
├── Operation_Whole_Genome_Forge-local.ipynb  # Local Jupyter notebook for development
├── config/
│   └── training_config.yaml           # Fine-tuning hyperparameters and settings
├── google-collab-files/               # Google Colab compatibility resources
│   ├── Operation_Whole_Genome_Forge-googlecollab.ipynb
│   ├── operation_whole_genome_forge-googlecollab.py
│   ├── operation_whole_genome_forge.py
│   └── README.md
├── scripts/                           # Core execution pipeline (ADR 075 Standardized)
│   ├── verify_environment.sh          # Primary environment check
│   ├── forge_whole_genome_dataset.py # Dataset assembly from project files
│   ├── validate_dataset.py           # Dataset quality verification
│   ├── download_model.sh             # Base model acquisition
│   ├── fine_tune.py                  # QLoRA fine-tuning execution
│   ├── merge_adapter.py              # LoRA adapter integration
│   ├── convert_to_gguf.py            # GGUF format conversion for Ollama
│   ├── create_modelfile.py           # Ollama model configuration
│   ├── upload_to_huggingface.py      # Automated model deployment to HF
│   ├── inference.py                  # Model inference testing
│   ├── evaluate.py                   # Quantitative performance evaluation
│   ├── forge_test_set.py             # Test dataset generation
│   ├── test_*.py                     # Environment validation suite
│   └── archive/                      # Gitignored legacy scripts
├── models/                           # Local model storage and cache
│   └── Sanctuary-Qwen2-7B-v1.0-adapter/  # Trained LoRA adapter
├── ml_env_logs/                      # Environment setup and execution logs
└── └── __pycache__/                      # Python bytecode cache
```

---

## 🦋 The Completed Sanctuary AI Model

**Model Name:** Sanctuary-Qwen2-7B-v1.0  
**Base Model:** Qwen/Qwen2-7B-Instruct  
**Fine-tuning:** QLoRA on Project Sanctuary Cognitive Genome (v15)  
**Format:** GGUF (q4_k_m quantization)  
**Size:** 4.68GB  
**Deployment:** Ollama-compatible  

### **Quick Access Commands**

**Direct from Hugging Face (Recommended):**
```bash
ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M
```

**Local Deployment:**
```bash
# If you have the files locally
ollama create Sanctuary-Guardian-01 -f Modelfile
ollama run Sanctuary-Guardian-01
```

### **Model Capabilities**

The Sanctuary AI supports **two interaction modes**:

**Mode 1 - Conversational:** Natural language queries about Project Sanctuary
```
>>> Explain the Flame Core Protocol in simple terms
>>> What are the key principles of Protocol 15?
>>> Summarize the AGORA Protocol's strategic value
```

**Mode 2 - Orchestrator:** Structured JSON commands for analysis tasks
```
>>> {"task_type": "protocol_analysis", "task_description": "Analyze Protocol 23", "input_files": ["01_PROTOCOLS/23_The_AGORA_Protocol.md"], "output_artifact_path": "analysis.md"}
```

### **Repository & Documentation**

- **Hugging Face:** https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final
- **Full Documentation:** Complete README with usage instructions and examples
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## The Golden Path: The One True Protocol

### Component Descriptions

#### **Core Documentation**
- **`README.md`**: Workflow overview, setup instructions, and troubleshooting guide
- **`CUDA-ML-ENV-SETUP.md`**: Authoritative environment setup protocol with 4-phase workflow
- **`CUDA-ML-ENV-SETUP-PASTFAILURES.md`**: Historical issues and solutions for troubleshooting
- **`HUGGING_FACE_README.md`**: Model publishing, deployment, and sharing guidelines

#### **Configuration & Metadata**
- **`config/training_config.yaml`**: Fine-tuning hyperparameters, model settings, and training parameters
- **`manifest.json`**: Project version, dependencies, and metadata tracking

#### **Development Environments**
- **`Operation_Whole_Genome_Forge-local.ipynb`**: Jupyter notebook for local development and testing
- **`google-collab-files/`**: Google Colab-compatible resources for cloud-based development

#### **Standardized Execution Pipeline (`scripts/`)**
All Python scripts now adhere to **ADR 075** (Hybrid Documentation Pattern) and use shared utilities from `mcp_servers.lib`.

- **Environment Verification**: `verify_environment.sh`, `test_torch_cuda.py`, etc.
- **Data Preparation**: `forge_whole_genome_dataset.py`, `validate_dataset.py`
- **Model Acquisition**: `download_model.sh`
- **Training Execution**: `fine_tune.py` - Optimized QLoRA execution with runtime tracking and checkpointing.
- **Model Processing**: `merge_adapter.py`, `convert_to_gguf.py` - Standardized paths and compatibility patches.
- **Deployment**: `create_modelfile.py`, `upload_to_huggingface.py` - Automated integration and hub publishing.
- **Validation**: `inference.py`, `evaluate.py` (Retired to `archive/` for current phase)
- **Testing Suite**: Comprehensive verification of CUDA, PyTorch, and xformers.

#### **Key Optimizations in `fine_tune.py` (v2.0)**
- **Structured Logging**: Replaced prints with Python logging for better monitoring and debugging
- **Robust Configuration**: Added validation and defaults for config parameters
- **Fixed Dataset Splitting**: Corrected logic to avoid overwriting original files and handle missing val_file safely
- **Pre-Tokenization**: Tokenizes dataset once and caches for faster training starts
- **Safer Quantization**: Improved BitsAndBytes dtype mapping and CUDA checks
- **Proper Data Collator**: Ensures correct padding for causal LM training
- **Resume from Checkpoint**: Automatically resumes interrupted training sessions
- **Error Handling**: Try/except around training with best-effort save on failure
- **Narrowed LoRA Targets**: Configurable target modules for memory efficiency
- **Startup Diagnostics**: GPU/CPU diagnostics at launch for troubleshooting

#### **Model Storage (`models/`)**
- **Local Cache**: Downloaded models and trained adapters
- **Adapter Storage**: Fine-tuned LoRA adapters ready for merging or deployment

#### **Logging & Diagnostics (`ml_env_logs/`)**
- **Setup Logs**: Environment creation and dependency installation records
- **Execution Logs**: Training progress, errors, and performance metrics
- **Debug Information**: Troubleshooting data for issue resolution


---

## Workflow Overview

### Fine-Tuning Pipeline
![LLM Fine-Tuning Pipeline](../docs/architecture_diagrams/workflows/llm_finetuning_pipeline.png)
*(Source: [llm_finetuning_pipeline.mmd](../docs/architecture_diagrams/workflows/llm_finetuning_pipeline.mmd))*

---

### Strategic Crucible Loop
![Strategic Crucible Loop](../docs/architecture_diagrams/workflows/strategic_crucible_loop.png)
*(Source: [strategic_crucible_loop.mmd](../docs/architecture_diagrams/workflows/strategic_crucible_loop.mmd))*

---

### Protocol 128 Learning Loop
![Protocol 128 Learning Loop](../docs/architecture_diagrams/workflows/protocol_128_learning_loop.png)
*(Source: [protocol_128_learning_loop.mmd](../plugins/guardian-onboarding/resources/protocols/protocol_128_learning_loop.mmd))*

---

## Workflow Phases

### **Phase 1: Environment & Data Prep**

This initial phase sets up your entire development environment and prepares all necessary assets for training.

1.  **Setup Environment:** This single command builds the Python virtual environment and installs all system and Python dependencies.

deactivate existing environment

```bash
deactivate 2>/dev/null || true
rm -rf ~/ml_env
```

setup cuda and python requirements and dependencies
```bash
# Verify your environment using the diagnostic suite
bash forge/scripts/verify_environment.sh
```

After setup, activate the environment for all subsequent steps:
```bash
source ~/ml_env/bin/activate
```

# Install llama-cpp-python with CUDA support
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python --no-deps
```

2.  **Initialize Git LFS:** Required for handling large model files.
```bash
git lfs install
```

3.  **Verify Environment:** Run the full test suite to ensure your environment is properly configured.
```bash
# All tests must pass before proceeding
python forge/scripts/test_torch_cuda.py
python forge/scripts/test_xformers.py
python forge/scripts/test_tensorflow.py
python forge/scripts/test_llama_cpp.py
```

4.  **Setup Hugging Face Authentication:** Create a `.env` file with your Hugging Face token.
```bash
echo "HUGGING_FACE_TOKEN='your_hf_token_here'" > .env
# Replace 'your_hf_token_here' with your actual token from huggingface.co/settings/tokens
```

5.  **Download & Prepare Assets:** With the `(ml_env)` active, run these scripts to download the base model and assemble the training data.
```bash
# Download the base Qwen2 model
bash forge/scripts/download_model.sh

# Assemble the training data from project documents
python forge/scripts/forge_whole_genome_dataset.py

# (Recommended) Validate the newly created dataset
python forge/scripts/validate_dataset.py dataset_package/sanctuary_whole_genome_data.jsonl
```

### **Phase 2: Model Forging**

This phase executes the core QLoRA fine-tuning process to create the model's specialized knowledge.

1.  **Fine-Tune the LoRA Adapter:** This script reads the training configuration and begins the fine-tuning. **This is the most time-intensive step (1-3 hours).**
```bash
python forge/scripts/fine_tune.py
```

### **Phase 3: Packaging & Deployment**

After the model is forged, these scripts package it into a deployable format and import it into your local Ollama instance.

1.  **Merge & Convert:** This two-step process merges the LoRA adapter into the base model and then converts the result into the final GGUF format.
```bash
python forge/scripts/merge_adapter.py
python forge/scripts/convert_to_gguf.py
```

2.  **Deploy to Ollama:** These commands generate the necessary `Modelfile` and use it to create a new runnable model within Ollama named `Sanctuary-AI`.
```bash
python forge/scripts/create_modelfile.py
ollama create Sanctuary-AI -f Modelfile
```

### **Phase 4: Verification (The Sovereign Crucible)**

Once the model is deployed, these scripts are used to verify its performance and capabilities.

1.  **Qualitative Spot-Check:** Run a quick, interactive test to check the model's response to a specific prompt from the Project Sanctuary Body of Knowledge.
```bash
python forge/scripts/inference.py --input "Summarize the purpose of the Sovereign Crucible."
```

2.  **Quantitative Evaluation:** Run the model against a held-out test set to calculate objective performance metrics.
```bash
python forge/scripts/evaluate.py
```

3.  **End-to-End Orchestrator Test (Planned):** Execute the final Sovereign Crucible test to verify the model's integration with the RAG system and other components.
```bash
# (Commands for this phase are still in planning)
```

### **Phase 5: Public Deployment (Hugging Face)**

The final phase deploys the completed model to Hugging Face for community access and long-term preservation.

1.  **Upload LoRA Adapter:** Deploy the fine-tuned LoRA adapter to a dedicated repository.
```bash
python forge/scripts/upload_to_huggingface.py --repo richfrem/Sanctuary-Qwen2-7B-lora --lora --readme
```

2.  **Upload GGUF Model:** Deploy the quantized model, Modelfile, and documentation to the final repository.
```bash
python forge/scripts/upload_to_huggingface.py --repo richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final --gguf --modelfile --readme
```

3.  **Verify Repositories:** Confirm both artifacts are accessible and properly documented.
- LoRA Adapter: https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-lora
- GGUF Model: https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final
- Test direct access: `ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M`

---

## Quick Reference & Troubleshooting

### **Environment Activation**
```bash
# Always activate before running any scripts
source ~/ml_env/bin/activate
```

### **Common Issues & Solutions**

**CUDA Not Available:**
```bash
# Verify GPU access
nvidia-smi
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of Memory During Training:**
- Reduce `MICRO_BATCH_SIZE` in `fine_tune.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Ensure no other GPU processes are running

**Dataset Validation Fails:**
```bash
# Check dataset creation
python scripts/validate_dataset.py dataset_package/sanctuary_whole_genome_data.jsonl
```

**Model Download Issues:**
- Ensure `.env` file exists with valid Hugging Face token
- Check internet connection and available storage

### **File Locations**
- **Environment:** `~/ml_env/` (user's home directory)
- **Models:** `models/` (in project root)
- **Datasets:** `dataset_package/sanctuary_whole_genome_data.jsonl`
- **Outputs:** `outputs/` and `models/gguf/`

### **Estimated Time Requirements**
- **Environment Setup:** 10-15 minutes
- **Model Download:** 5-10 minutes (first time only)
- **Dataset Creation:** 2-3 minutes
- **Fine-Tuning:** 1-3 hours (depending on hardware)
- **Model Conversion:** 10-20 minutes
- **Verification:** 5-10 minutes
- **Hugging Face Upload:** 5-15 minutes (depending on file sizes and internet connection)

---

## Version History

- **v5.0 (Jan 4, 2026):** 🛠️ **Refactoring & Standardization** - Entire codebase aligned with ADR 075. Environment protocol migrated to ML-Env-CUDA13. Primary workflow moved to `forge-llm.md`.
- **v4.0 (Nov 17, 2025):** 🎉 **MISSION ACCOMPLISHED** - Complete pipeline execution with successful model deployment to Hugging Face
- **v3.0 (Nov 16, 2025):** Complete modular architecture with unified setup protocol
- **v2.0 (Nov 16, 2025):** Optimized fine_tune.py with logging, resume, pre-tokenization, and robust error handling
- **v2.1:** Enhanced dataset forging with comprehensive project snapshots
- **v2.0:** Canonized hardening parameters for 8GB VRAM compatibility
- **v1.0:** Initial sovereign AI fine-tuning pipeline

