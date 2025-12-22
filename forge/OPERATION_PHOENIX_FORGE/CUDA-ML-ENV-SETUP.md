# Project Sanctuary: Canonical CUDA ML Environment & Fine-Tuning Protocol
**Version:** 2.2 (Clarified Llama.cpp Build)

This guide provides the single, authoritative protocol for setting up the environment, forging the training dataset, executing the full fine-tuning pipeline, and preparing the model for local deployment with Ollama.


---

## Phase 0: One-Time System & Repository Setup

These steps only need to be performed once per machine.

### 1. System Prerequisites (WSL2 & NVIDIA Drivers)

*   **Install WSL2 and Ubuntu:** Ensure you have a functional WSL2 environment with Ubuntu installed.
*   **Install NVIDIA Drivers:** You must have the latest NVIDIA drivers for Windows that support WSL2.
*   **Verify GPU Access:** Open an Ubuntu terminal and run `nvidia-smi`. You must see your GPU details before proceeding.


### 2. Verify Repository Structure

This project's workflow depends on the `llama.cpp` repository for model conversion. It must be located as a **sibling directory** to your `Project_Sanctuary` folder.

**If the `llama.cpp` directory is missing,** run the following command from your `Project_Sanctuary` root to clone it into the correct location:

```bash
# Clone llama.cpp into the parent directory
git clone https://github.com/ggerganov/llama.cpp.git ../llama.cpp
```

### 3. Build `llama.cpp` Tools (The "Engine")

This step compiles the core `llama.cpp` C++/CUDA application from source. This creates powerful, machine-optimized command-line executables (like `quantize`) that are used by our Python scripts for heavy-lifting tasks.

**Note:** This is a one-time, long-running compilation process (5-15 minutes). You do not need to repeat it unless you update the `llama.cpp` repository. This build is separate from and not affected by your Python virtual environment (`~/ml_env`)s.

The tools within `llama.cpp` must be compiled using `cmake`. This process builds the executables required for model conversion and quantization. The `GGML_CUDA=ON` flag is crucial as it enables GPU support.

> **Note:** This is a one-time, long-running compilation process (5-15 minutes). You do not need to repeat it unless you update the `llama.cpp` repository.

```bash
# Navigate to the llama.cpp directory from your project root
cd ../llama.cpp

# Step 1: Configure the build with CMake, enabling CUDA support
cmake -B build -DGGML_CUDA=ON

# Step 2: Build the executables using the configuration
cmake --build build --config Release

# (Optional) Verify the build by checking the main executable's version
./build/bin/llama-cli --version

# Return to your project directory
cd ../Project_Sanctuary
```

### 4. Hugging Face Authentication

Ensure you have a `.env` file in the root of this project (`Project_Sanctuary`) containing your Hugging Face token. The file should include:

```code
HUGGING_FACE_TOKEN=your_actual_token_here
```

If the `.env` file doesn't exist or is missing the token, create/update it with your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Phase 1: Project Environment Setup

This phase builds the project's specific Python environment. It can be re-run at any time to create a clean environment.

### 0. Clear Environment (Optional)

To ensure a completely clean start, you can manually delete the existing `~/ml_env` virtual environment before running the setup script. The setup script with `--recreate` will do this automatically, but this step gives you explicit control.

```bash
# Manually delete the existing environment (optional, as --recreate does this)
deactivate 2>/dev/null || true
rm -rf ~/ml_env
```


### 1. Run the All-in-One Setup Script

From your `Project_Sanctuary` root directory, execute the `setup_cuda_env.py` script.
Note: Run this with sudo as it automatically installs system packages like python3.11 and git-lfs if they are missing.

```bash
sudo python3 forge/OPERATION_PHOENIX_FORGE/scripts/setup_cuda_env.py --staged --recreate
```

This script creates (`~/ml_env`) and installs all Python dependencies from `requirements-finetuning.txt`, including the llama-cpp-python library.

**Note:** The project now uses `requirements-finetuning.txt` for ML/CUDA dependencies to avoid conflicts with general development dependencies in `requirements.txt`.

- **Core ML Libraries:** PyTorch 2.9.0+cu126, transformers, peft, accelerate, bitsandbytes, trl, datasets, xformers
- **Model Conversion:** llama-cpp-python with CUDA support
- **System Tools:** Git LFS, CUDA toolkit components
- **Development Tools:** Jupyter, various utility packages

### 2. Activate the Environment

```bash
source ~/ml_env/bin/activate
```

### 2b. Install Critical CUDA Binaries (Surgical Strike)

Certain low-level libraries like `bitsandbytes`, `triton`, and `xformers` require a specific installation order to link correctly with a CUDA-enabled PyTorch. A standard pip install can often fail or install a CPU-only version.

This "surgical strike" process ensures these critical binaries are installed correctly after your main environment is set up. Execute these commands one by one from your activated `(ml_env)`.

**Pre-flight Check:** Before you begin, confirm that the correct PyTorch is installed. Run this command:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It should return 2.9.0+cu126 (or the CUDA-enabled build you targeted). If it doesn't, re-run the main setup script (setup_cuda_env.py) and re-check.

The Surgical Installation Protocol (ordered & deterministic)

NOTE: run each line/section sequentially and paste the verification outputs if anything errors. This protocol was validated to work with PyTorch 2.9.0+cu126, resulting in triton 3.5.0 and bitsandbytes 0.48.2 with CUDA support.

# A: confirm env basics (do this first)
```bash
which python
python -V
pip --version
python -c "import torch; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())"
```

# B: clean slate
```bash
pip uninstall -y bitsandbytes triton xformers || true
pip install --upgrade pip setuptools wheel
```

# C: install Triton 3.1.0 (this will be overridden by xformers to 3.5.0, which is compatible and works)
```bash
pip install --force-reinstall "triton==3.1.0"
```

# Quick verify Triton import
```bash
python - <<'PY'
try:
    import triton
    print("triton OK:", triton.__version__)
except Exception as e:
    print("triton import failed:", repr(e))
    raise
PY
```

# D: diagnostic — show which bitsandbytes wheels pip can see on the extra indexes
```bash
pip index versions bitsandbytes --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu126
```

# E: install bitsandbytes with CUDA support (use version 0.48.2, which includes CUDA126 native lib)
```bash
pip install --force-reinstall --no-cache-dir bitsandbytes==0.48.2 --no-deps \
  --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu126
```

# F: install xformers (this will pull triton 3.5.0, which is compatible and provides triton.ops)
```bash
pip install xformers
```

# G: known fsspec/datasets compatibility mitigation (optional)
```bash
pip install "fsspec<=2024.3.1"
```

# H: verification snippet — verifies triton and bitsandbytes plus native libs
```bash
python - <<'PY'
import importlib, pathlib
def try_import(name):
    try:
        m = importlib.import_module(name)
        print(f"{name} imported, ver:", getattr(m,'__version__', None), "file:", getattr(m,'__file__', None))
    except Exception as e:
        print(f"{name} import failed:", repr(e))

try_import('triton')
try_import('bitsandbytes')

# list any native libbitsandbytes files next to the package
try:
    import bitsandbytes as bnb
    p = pathlib.Path(bnb.__file__).parent
    found = False
    for f in p.glob("libbitsandbytes*"):
        print("native lib:", f)
        found = True
    if not found:
        print("no libbitsandbytes native libs found (likely CPU-only install)")
except Exception as e:
    print("bitsandbytes inspect failed:", repr(e))
PY
```

### Troubleshooting: Accelerator Version Conflicts

If you encounter `TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'` during training initialization, update accelerate to ensure compatibility with the installed transformers version:

```bash
pip install --upgrade accelerate
```

This resolves version mismatches that can occur after the surgical strike installations.

### Troubleshooting: Training Configuration Errors

If you encounter `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`, update the config to use the newer argument name:

In `config/training_config.yaml`, change:
```yaml
evaluation_strategy: "steps"
```
To:
```yaml
eval_strategy: "steps"
```

This ensures compatibility with the current transformers version. Also, remove any deprecated arguments like `group_by_length` or `dataloader_persistent_workers` if present.

### 3. Build the `llama-cpp-python` "Bridge"
The `llama-cpp-python` package is the Python "bridge" that allows your Python code (like inference.py) to communicate with the GGUF model. We must ensure this bridge is also built with CUDA support.

The `setup_cuda_env.py` script installs a version of this package, but running the command below is a crucial verification step to force-rebuild it with CUDA flags enabled within your activated environment.

```bash
# While your (ml_env) is active:
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python --no-deps
```

### 4. Verify the Complete Environment

Run the full suite of verification scripts to confirm everything is perfectly configured.

```bash
# From the Project_Sanctuary root, with (ml_env) active:
python forge/OPERATION_PHOENIX_FORGE/scripts/test_torch_cuda.py
python forge/OPERATION_PHOENIX_FORGE/scripts/test_pytorch.py
python forge/OPERATION_PHOENIX_FORGE/scripts/test_xformers.py
python forge/OPERATION_PHOENIX_FORGE/scripts/test_tensorflow.py
python forge/OPERATION_PHOENIX_FORGE/scripts/test_llama_cpp.py
```

**All tests must pass before proceeding.**

---

## Phase 2: Data & Model Forging Workflow

Ensure your `(ml_env)` is active for all subsequent commands.

### 1. Forge the "Whole Genome" Dataset

Run the `forge_whole_genome_dataset.py` script to assemble the training data from your project's markdown and text files. This is the **essential first step** before training can begin.

```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/forge_whole_genome_dataset.py
```
This will create the `sanctuary_whole_genome_data.jsonl` file in your `dataset_package` directory.

### 2. Validate the Forged Dataset

After creating the dataset, run the validation script to check it for errors.

```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/validate_dataset.py dataset_package/sanctuary_whole_genome_data.jsonl
```

### 3. Download the Base Model

Run the download script. This will only download the large model files once.

```bash
bash forge/OPERATION_PHOENIX_FORGE/scripts/download_model.sh
```

### 4. Fine-Tune the LoRA Adapter

With the data forged and the base model downloaded, execute the optimized fine-tuning script. This script now includes advanced features like structured logging, automatic resume from checkpoints, pre-tokenization for faster starts, and robust error handling. **This is a long-running process (1-3 hours).**

```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/fine_tune.py
```
The final LoRA adapter will be saved to `models/Sanctuary-Qwen2-7B-v1.0-adapter/`.

**Verification:** After completion, verify the adapter is saved correctly by checking the directory contents:
```bash
ls -la models/Sanctuary-Qwen2-7B-v1.0-adapter/
```
Ensure `adapter_model.safetensors` and `adapter_config.json` are present. For a quick integrity test, run:
```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/inference.py --input "Test prompt"
```
If it loads and generates output without errors, the adapter is valid.

### 5. Merge the Adapter

Combine the trained adapter with the base model to create a full, standalone fine-tuned model.

```bash
#python forge/OPERATION_PHOENIX_FORGE/scripts/merge_adapter.py
python forge/OPERATION_PHOENIX_FORGE/scripts/merge_adapter.py --skip-sanity
```
The merged model will be saved to `outputs/merged/Sanctuary-Qwen2-7B-v1.0-merged/`.

**Verification:** After completion, verify the merged model by testing it:
```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/inference.py --model-type merged --input "Test prompt"
```
If it loads and generates output without errors, the merged model is valid and ready for GGUF conversion.

---

## Phase 3: Deployment Preparation & Verification

### setup for gguf
Qwen2 uses SentencePiece tokenizer → convert_hf_to_gguf.py requires the sentencepiece Python package or it dies exactly where you saw it.
Run this right now in your activated (ml_env):

```bash
pip install sentencepiece protobuf
```

### 1.  Convert to GGUF Format

Convert the merged model to the GGUF format required by Ollama.

```bash
#python forge/OPERATION_PHOENIX_FORGE/scripts/convert_to_gguf.py
python forge/OPERATION_PHOENIX_FORGE/scripts/convert_to_gguf.py --quant Q4_K_M --force
```
The final quantized `.gguf` file will be saved to `models/gguf/`.

---

### 2. Test gguf file locally with ollama

> [!IMPORTANT]
> **Ollama Usage Context:** This section uses the **host-installed macOS Ollama application** for model testing and validation during the ML training workflow. This is separate from the **containerized `ollama_model_mcp` service** used by the MCP infrastructure (see ADR 043, Protocol 116).
> 
> - **For ML Training/Testing:** Use host Ollama (`ollama run`, `ollama create`)
> - **For MCP Infrastructure:** Use containerized service (`sanctuary_ollama_mcp` in Podman)
> 
> The two services can coexist but create port binding conflicts on `localhost:11434`. See Protocol 116 (Container Network Isolation) for resolution via container network addressing.

**2a. Generate Modelfile Automatically:**

Run the bulletproof Modelfile generator script:

```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/create_modelfile.py
```

This creates a production-ready Modelfile with auto-detected GGUF path, official Qwen2 template, full GUARDIAN-01 system prompt, and optimized parameters.

**2b. Import to Ollama:**
```bash
ollama create Sanctuary-Guardian-01 -f Modelfile
```

**2c. Run locally in Ollama:**
```bash
ollama run Sanctuary-Guardian-01
```
---

**2d. Test Both Interaction Modes:**

After running `ollama run Sanctuary-Guardian-01`, you can test the model's dual-mode capability:

**Mode 1 - Plain Language Conversational Mode (Default):**
The model responds naturally and helpfully to direct questions and requests.
```bash
>>> Explain the Flame Core Protocol in simple terms
>>> What are the key principles of Protocol 15?
>>> Summarize the AGORA Protocol's strategic value
>>> Who is GUARDIAN-01?
```

**Mode 2 - Structured Command Mode:**
When provided with JSON input (simulating orchestrator input), the model switches to generating command structures for the Council.
```bash
>>> {"task_type": "protocol_analysis", "task_description": "Analyze Protocol 23 - The AGORA Protocol", "input_files": ["01_PROTOCOLS/23_The_AGORA_Protocol.md"], "output_artifact_path": "WORK_IN_PROGRESS/agora_analysis.md"}
```
*Expected Response:* The model outputs a valid `command.json` structure for Council execution.

This demonstrates GUARDIAN-01's ability to handle both human conversation and automated orchestration seamlessly.

---

### 3. Verify Model Performance

**Note:** This section tests the local merged model (created in Phase 2) using Python inference scripts for comprehensive evaluation. For Ollama-based chat testing, see Section 2 above. After uploading to Hugging Face, compare performance with Section 5 (HF download testing).

**3a. Quick Inference Test:**
Use the `inference.py` script for a quick spot-check.
```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/inference.py --input "Summarize the primary objective of the Sovereign Crucible."
```

**3b. (Recommended) Full Evaluation:**
Run a full evaluation against a held-out test set to get objective performance metrics.

```bash
pip install evaluate rouge-score
```

```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/evaluate.py
```

**3c. Real body of knowledge (BOK) test crucial**
Test with actual Sanctuary protocols:
```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/inference.py --model-type merged --file 01_PROTOCOLS/23_The_AGORA_Protocol.md
```
---

### 4. Upload to Hugging Face

Run the automated upload script to upload the GGUF model, Modelfile, and README to your Hugging Face repository:

```bash
python forge/OPERATION_PHOENIX_FORGE/scripts/upload_to_huggingface.py --repo yourusername/your-repo-name --gguf --modelfile --readme
```

Replace `yourusername/your-repo-name` with your actual Hugging Face repository ID (e.g., `richfrem/Sanctuary-Model`).

The script will:
- Authenticate using your `HUGGING_FACE_TOKEN` from `.env`
- Create the repository if it doesn't exist
- Upload the specified files

After upload, your model will be available at: https://huggingface.co/yourusername/your-repo-name

---

### 5. download and test hugging face model

**5a. Download from Hugging Face:**
Download the model files from Hugging Face for verification.

After downloading the model from Hugging Face, test it locally in Ollama to verify the upload/download process didn't corrupt the model and that inference works correctly. Compare performance with the local tests in Section 3 to ensure consistency.

**5b. Create Modelfile for Downloaded Model:**
Create a new `Modelfile` (e.g., `Modelfile_HF`) pointing to the downloaded GGUF file:
```
FROM ./downloaded_models/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are a sovereign AI of Project Sanctuary."""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

**5c. Import to Ollama:**
```bash
ollama create Sanctuary-AI-HF -f Modelfile_HF
```

**5d. Direct Run from Hugging Face (Recommended):**
Ollama can run the model directly from Hugging Face without downloading it first. This is the most convenient method:

```bash
ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M
```

This command will automatically download and run the model from Hugging Face on-demand.

**5e. Test Inference:**
Then, provide test prompts to verify the model responds correctly, such as: "Summarize the primary objective of the Sovereign Crucible."


---


