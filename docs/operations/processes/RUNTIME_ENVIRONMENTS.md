# Runtime Environments Strategy

Project Sanctuary uses a **Dual Environment Strategy** to separate heavy machine learning dependencies from standard application development tools. This ensures stability, avoids dependency conflicts (hell), and optimizes resource usage.

## The Two Worlds

| Feature | **System A: `.venv`** | **System B: `ml_env`** |
| :--- | :--- | :--- |
| **Name** | **Standard Development** | **The Forge** |
| **Purpose** | Daily coding, RAG, Gateway, Audits, Git Ops | Fine-Tuning, Merging, Quantization (ML Ops) |
| **Key Libraries** | `fastapi`, `langchain`, `chromadb`, `mcp` | `torch` (CUDA), `transformers`, `unsloth`, `bitsandbytes` |
| **Hardware** | CPU / Standard | GPU (CUDA 12.x) |
| **Management** | [[073_standardization_of_python_dependency_management_across_environments|**P073**]] / [[dependency_management_policy|**Rules**]] | **External** (Governed by `ML-Env-CUDA13`) |
| **Size** | ~200MB - 500MB | ~5GB - 10GB |

> **Governance Note:** The **`ml_env`** is a shared resource governed by the external [**`ML-Env-CUDA13`**](https://github.com/richfrem/ML-Env-CUDA13) project. Project Sanctuary *consumes* it for Forge operations to ensure optimized CUDA alignment across all AI initiatives.

---

## When to Use Which?

### Use `.venv` (Standard)
This will be your primary environment for daily development, running the Gateway Fleet, and performing audits.
*   Running `setup/` scripts.
*   Running `mcp_servers/` code (Gateway, Cortex, Council).
*   Running **Audits** (`scripts/cortex_cli.py`).
*   Committing code (`scripts/run_genome_tests.sh`).
*   Running the Gateway Fleet.

### Use `ml_env` (The Forge)
Youll only need this environment for fine-tuning, merging, and quantization.
*   **Phase 2:** Running `forge/scripts/fine_tune.py`.
*   **Phase 4:** Running `forge/scripts/merge_adapter.py`.
*   **Phase 5:** Running `forge/scripts/convert_to_gguf.py`.
*   **Phase 6:** Testing locally with `llama-cpp-python` (if installed there).

---

## Cross-Platform Caveats (WSL vs Windows)

If you are developing on Windows with WSL, be aware:
*   **Windows** uses `.venv\Scripts\activate` (executables).
*   **Linux (WSL)** uses `.venv/bin/activate` (ELF binaries).

These are **NOT** compatible. If you created `.venv` on Windows (via VS Code or PowerShell), it will not work in WSL.

* ### 1. The "Platform Reset" (Recommended)
> **Note for macOS Users:** These instructions (Linux) also apply to macOS. `make bootstrap` works natively on both systems.

If you see `source: .venv/bin/activate: No such file or directory` (because it was created on Windows), you must reset it for Linux/macOS:

```bash
# Delete the incompatible Windows environment.  if on windows and have a .venv folder already
# Note don't confuse .venv with ml_env
rm -rf .venv

# Bootstrap a fresh Linux-compatible standard environment
# NOTE: This installs the "Universal Fleet Environment" (Core + All 8 Clusters).
# Use 40-60 minutes to install (WSL/HDD):
#   - mcp_servers/requirements-core.txt (fastapi, mcp)
#   - mcp_servers/gateway/clusters/*/requirements.txt (cortex/torch, git, filesystem, etc.)
make bootstrap

# Activate
source .venv/bin/activate
```

---

## How to Switch

You **MUST** deactivate one environment before activating the other to prevent "pollution" of `PYTHONPATH` or library paths.

### 1. From `.venv` to `ml_env` (Standard -> Forge)
```bash
# 1. Stop current shell
deactivate

# 2. Activate Forge
source ~/ml_env/bin/activate

# 3. Verify (GPU Check)
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. From `ml_env` to `.venv` (Forge -> Standard)
```bash
# 1. Stop Forge
deactivate

# 2. Activate Standard
source .venv/bin/activate

# 3. Verify (Path Check)
which python
# Should be: .../Project_Sanctuary/.venv/bin/python
```
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'langchain'"
*   **Cause:** You are running a Standard tool (like `cortex_cli.py`) inside `ml_env`.
*   **Fix:** `deactivate` and switch to `.venv`.

### "ModuleNotFoundError: No module named 'unsloth'"
*   **Cause:** You are trying to fine-tune inside `.venv`.
*   **Fix:** `deactivate` and switch to `ml_env`.

### "OSError: libcuda.so not found"
*   **Cause:** You are in `.venv` trying to access GPU functions not supported, or `ml_env` is broken.
*   **Fix:** Ensure you are in `ml_env`.

---

## Deep Dive: The `ml_env` Architecture
*Source: ML-Env-CUDA13 Handover (ADR 001 Hybrid Dependency Management)*

The `ml_env` is not a standard venv. It uses a **Hybrid Layered Approach** to verify hardware compatibility before installing libraries.

### Layer 1: The Foundation (Dynamic)
*   **What:** Hardware-specific binaries (`torch`, `tensorflow`, `xformers`).
*   **Management:** Installed via `setup_ml_env_wsl.sh --cuda13` (Direct pip, no verification).
*   **Feature:** Aggressively purges conflicting `nvidia-*-cu12` packages to prevent linker errors (`ncclDevCommDestroy`).

### Layer 2: The Application (Strict)
*   **What:** Project libraries (`peft`, `trl`, `gguf`, `chromadb`).
*   **Management:** `requirements.in` compiled against the *installed* Layer 1 to ensure compatibility.

### How to Rebuild (If Broken)
If `ml_env` becomes unstable (e.g., linker errors), do not just `pip install`. Use the Governor script:
```bash
# In ML-Env-CUDA13 repo:
bash scripts/setup_ml_env_wsl.sh --cuda13
```
