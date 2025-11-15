# CUDA ML Environment Setup Instructions
 [ML-Env-CUDA13 GitHub Repository](https://github.com/bcgov/ML-Env-CUDA13)

This guide will help you set up a CUDA-enabled machine learning environment for Project_Sanctuary using WSL2 and ML-Env-CUDA13.

## Prerequisites
- **ML-Env-CUDA13** cloned at the same level as this project.  
- **WSL2 (Ubuntu)** with NVIDIA GPU drivers
- **Python 3.10+** (managed by ML-Env-CUDA13)
- Install ML-Env-CUDA13 dependencies before running fine-tuning scripts


## One-Time Setup Steps

### 1. Install WSL2 and Ubuntu
- Install WSL2 and Ubuntu from the Microsoft Store.
- Set WSL2 as the default version:
   ```powershell
   wsl --set-default-version 2
   ```
- Launch Ubuntu and set up your username/password.

### 2. Install NVIDIA CUDA Drivers
- Download and install the latest NVIDIA GPU drivers for Windows.
- Install the CUDA toolkit for WSL2 (follow official NVIDIA instructions).
- Verify installation in WSL2:
   ```bash
   nvidia-smi
   ```

### 3. Clone ML-Env-CUDA13
- In your WSL2 Ubuntu terminal, navigate to the parent directory of your project:
   ```bash
   cd /mnt/c/Users/<YourUsername>/source/repos
   ```
- Clone the ML-Env-CUDA13 repository:
   ```bash
   git clone https://github.com/bcgov/ML-Env-CUDA13.git
   ```

### 4. Run ML Environment Setup
- Run the setup script from your project directory.

  Use the command appropriate to where you run it from:

    ```bash
    # If you run this from the Project_Sanctuary repository root
    bash ../ML-Env-CUDA13/setup_ml_env_wsl.sh

    # If you run this from inside this forge subfolder
    # (i.e. `forge/OPERATION_PHOENIX_FORGE`), use the alternative relative path:
    # bash ../../../ML-Env-CUDA13/setup_ml_env_wsl.sh
    ```

- This will create and configure a Python virtual environment at `~/ml_env`.

Run the Python helper (optional)
 - A convenience script `scripts/setup_cuda_env.py` automates the staged flow (venv creation, PyTorch -> TF -> core gate -> rest, logs, and a small activation helper).
 - Run it from the project root in WSL (recommended):
   ```bash
   # recommended staged flow (creates/uses ~/ml_env)
   python3.11 scripts/setup_cuda_env.py --staged

   # recreate venv and run staged flow
   python3.11 scripts/setup_cuda_env.py --staged --recreate

   # quick one-step install (risky/long):
   python3.11 scripts/setup_cuda_env.py --quick

   # regenerate test files only (no installs):
   python3.11 scripts/setup_cuda_env.py --regen-tests-only
   ```
 - After the script finishes it writes an activation helper `activate_ml_env.sh` in the repo root; run:
 - After the script finishes it writes an activation helper at `scripts/activate_ml_env.sh`; run:
  ```bash
  source scripts/activate_ml_env.sh
  # or
  source ~/ml_env/bin/activate
  ```
 - The script captures logs in `ml_env_logs/` and writes a `pinned-requirements-<timestamp>.txt` after a successful core gate.

### 5. Install Fine-Tuning Dependencies (recommended: use the helper script)

The repo includes a Python helper `scripts/setup_cuda_env.py` that automates the staged install flow (create/use venv, upgrade pip/tools, install CUDA PyTorch, install TensorFlow, run the core-gate tests, snapshot `pip freeze`, then install remaining requirements and run diagnostics). Using the helper avoids manual copy/paste and is the recommended approach.

Recommended: run the staged helper from the project root in WSL:
```bash
# staged (recommended): installs PyTorch -> TensorFlow -> runs core gate -> installs rest
python3.11 scripts/setup_cuda_env.py --staged

# recreate venv then staged install
python3.11 scripts/setup_cuda_env.py --staged --recreate

# quick one-step install (risky/long):
python3.11 scripts/setup_cuda_env.py --quick

# regen tests only (no installs):
python3.11 scripts/setup_cuda_env.py --regen-tests-only
```

After the script finishes it writes an activation helper `activate_ml_env.sh` in the repo root. Activate the venv in your current shell with:
```bash
# source the helper from the `scripts/` directory
source scripts/activate_ml_env.sh
# or directly
source ~/ml_env/bin/activate
```

The helper script performs the staged flow for you; you do not need to run the manual commands below unless you want to reproduce or debug a single step. The script will:

- create/use the venv at `~/ml_env`
- upgrade `pip`, `wheel`, and `setuptools` inside the venv
- install CUDA PyTorch wheels (it now installs `torch`, `torchvision`, and `torchaudio` up-front if they are pinned in `requirements.txt`)
- install TensorFlow (or a pinned TF if you pass `--pin-tensorflow`)
- run the core verification test (`test_torch_cuda.py`) and write `ml_env_logs/*` + `.exit` files
- snapshot `pip freeze` to `pinned-requirements-<timestamp>.txt` on success and install the remaining `requirements.txt` packages

If you still want to run the steps manually for debugging, they are shown below for reference:
```bash
# (manual reference) update installer tools
pip install --upgrade pip wheel setuptools

# (manual reference) 1) Install CUDA PyTorch wheels explicitly (match your requirements)
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126

# (manual reference) 2) Install TensorFlow (pin if you have a validated version)
pip install --upgrade tensorflow

# (manual reference) 3) Run core verification test (core gate)
mkdir -p ml_env_logs
# From the Project_Sanctuary repo root:
python ../../../ML-Env-CUDA13/test_torch_cuda.py > ml_env_logs/test_torch_cuda.log 2>&1 || true
cat ml_env_logs/test_torch_cuda.log
# If you are executing from this forge subfolder instead, use the alternative relative path:
# python ../../../ML-Env-CUDA13/test_torch_cuda.py > ml_env_logs/test_torch_cuda.log 2>&1 || true

# (manual reference) 4) If core verification passed, install the remainder of the requirements
pip install -r requirements.txt
```

Run the optional diagnostic tests (recommended):
```bash
# From Project_Sanctuary root:
# From Project_Sanctuary root:
python ../../../ML-Env-CUDA13/test_pytorch.py > ml_env_logs/test_pytorch.log 2>&1 || true
python ../../../ML-Env-CUDA13/test_tensorflow.py > ml_env_logs/test_tensorflow.log 2>&1 || true
python ../../../ML-Env-CUDA13/test_xformers.py > ml_env_logs/test_xformers.log 2>&1 || true
python ../../../ML-Env-CUDA13/test_llama_cpp.py > ml_env_logs/test_llama_cpp.log 2>&1 || true

# Or, run these from this forge subfolder (`forge/OPERATION_PHOENIX_FORGE`):
# python ../../../ML-Env-CUDA13/test_pytorch.py > ml_env_logs/test_pytorch.log 2>&1 || true
# python ../../../ML-Env-CUDA13/test_tensorflow.py > ml_env_logs/test_tensorflow.log 2>&1 || true
# python ../../../ML-Env-CUDA13/test_xformers.py > ml_env_logs/test_xformers.log 2>&1 || true
# python ../../../ML-Env-CUDA13/test_llama_cpp.py > ml_env_logs/test_llama_cpp.log 2>&1 || true

# Or, if running from this forge subfolder (`forge/OPERATION_PHOENIX_FORGE`), use:
# python ../../../ML-Env-CUDA13/test_pytorch.py > ml_env_logs/test_pytorch.log 2>&1 || true
# python ../../../ML-Env-CUDA13/test_tensorflow.py > ml_env_logs/test_tensorflow.log 2>&1 || true
# python ../../../ML-Env-CUDA13/test_xformers.py > ml_env_logs/test_xformers.log 2>&1 || true
# python ../../../ML-Env-CUDA13/test_llama_cpp.py > ml_env_logs/test_llama_cpp.log 2>&1 || true

# If core gate passed and you want a reproducible snapshot locally:
pip freeze > pinned-requirements-$(date +%Y%m%d%H%M).txt
```

Notes and recommendations
- `requirements.txt` currently contains CUDA-specific pins (e.g. `torch==2.8.0+cu126`) and an extra-index-url for the cu126 PyTorch wheel index. This is fine for WSL CUDA installs but will break CPU-only hosts.
- Recommended file layout for clarity and portability:
  - `requirements.txt` — portable, cross-platform dependencies (no CUDA-suffixed pins)
  - `requirements-wsl.txt` — CUDA-specific pinned wheels (torch+cu126, torchvision+cu126, torchaudio+cu126, specific TF if desired)
  - `requirements-gpu-postinstall.txt` — optional heavy/experimental packages installed after core gate (xformers, bitsandbytes, llama-cpp-python, etc.)
- Keep `pinned-requirements-<ts>.txt` as local artifacts generated after a successful core gate. Do not overwrite the repo-level `requirements.txt` with a pinned, machine-specific snapshot unless you intend to require that exact GPU environment for all contributors.

If you want, I can split the current `requirements.txt` into a portable `requirements.txt` and a `requirements-wsl.txt` (and create `requirements-gpu-postinstall.txt`) and add brief instructions in this document showing which file to use in WSL. I will not perform any git operations — I will only create the files locally in the workspace for you to review.

- Notes and troubleshooting:
  - If you already ran the ML-Env-CUDA13 setup and prerequisites (as in this project), the two-step install above should pick up the correct CUDA-enabled wheels (example: `torch-2.8.0+cu126`).
  - Heavy / CUDA-sensitive packages (may require special wheels or build tools): `bitsandbytes`, `xformers`, `llama-cpp-python`, `sentencepiece`. If `pip` attempts to build these from source and fails, install their prebuilt wheels where available or install them after PyTorch is installed.
  - Common small conflict: TensorFlow may require a different `tensorboard` minor version. If you see a conflict like `tensorflow 2.20.0 requires tensorboard~=2.20.0, but you have tensorboard 2.19.0`, reconcile by running:
    ```bash
    pip install 'tensorboard~=2.20.0'
    ```
  - If a package (e.g., `xformers`) has no wheel for your Python/CUDA combo, building from source can be slow and require system build tools (`gcc`, `cmake`, etc.). Prefer prebuilt wheels or conda/mamba installs for those packages.

- Example: after running `python ../../../ML-Env-CUDA13/test_pytorch.py` you should see output similar to:
  ```text
  PyTorch: 2.8.0+cu126
  GPU Detected: True
  GPU 0: NVIDIA RTX ...
  CUDA build: 12.6
  ```
  This confirms the environment is CUDA-ready for PyTorch.

## Activating Your Environment (After Restarting Sessions)

```bash
source ~/ml_env/bin/activate
```

## Testing Your Environment

```bash
# Run the full set of verification tests (writes logs to ml_env_logs/ and .exit files)
mkdir -p ml_env_logs

# Core gate (must pass before taking pinned snapshot)
# Run these from the Project_Sanctuary repo root:
python ../../../ML-Env-CUDA13/test_torch_cuda.py > ml_env_logs/test_torch_cuda.log 2>&1 || true; echo $? > ml_env_logs/test_torch_cuda.exit

# Additional diagnostics (non-fatal)
python ../../../ML-Env-CUDA13/test_pytorch.py    > ml_env_logs/test_pytorch.log    2>&1 || true; echo $? > ml_env_logs/test_pytorch.exit
python ../../../ML-Env-CUDA13/test_tensorflow.py > ml_env_logs/test_tensorflow.log 2>&1 || true; echo $? > ml_env_logs/test_tensorflow.exit
python ../../../ML-Env-CUDA13/test_xformers.py   > ml_env_logs/test_xformers.log   2>&1 || true; echo $? > ml_env_logs/test_xformers.exit
python ../../../ML-Env-CUDA13/test_llama_cpp.py  > ml_env_logs/test_llama_cpp.log  2>&1 || true; echo $? > ml_env_logs/test_llama_cpp.exit

# If you are running from the forge subfolder instead, use the alternative relative paths (three levels up):
# python ../../../ML-Env-CUDA13/test_torch_cuda.py > ml_env_logs/test_torch_cuda.log 2>&1 || true; echo $? > ml_env_logs/test_torch_cuda.exit
# python ../../../ML-Env-CUDA13/test_pytorch.py    > ml_env_logs/test_pytorch.log    2>&1 || true; echo $? > ml_env_logs/test_pytorch.exit
# python ../../../ML-Env-CUDA13/test_tensorflow.py > ml_env_logs/test_tensorflow.log 2>&1 || true; echo $? > ml_env_logs/test_tensorflow.exit
# python ../../../ML-Env-CUDA13/test_xformers.py   > ml_env_logs/test_xformers.log   2>&1 || true; echo $? > ml_env_logs/test_xformers.exit
# python ../../../ML-Env-CUDA13/test_llama_cpp.py  > ml_env_logs/test_llama_cpp.log  2>&1 || true; echo $? > ml_env_logs/test_llama_cpp.exit

# Quick checks
echo 'Core gate exit:' $(cat ml_env_logs/test_torch_cuda.exit || echo 'no-exit-file')
tail -n 200 ml_env_logs/test_torch_cuda.log || true

# If core gate passed (exit 0) you can create a reproducible snapshot:
pip freeze > pinned-requirements-$(date +%Y%m%d%H%M).txt
```

---
**Note:**
- Make sure ML-Env-CUDA13 is cloned at the same directory level as Project_Sanctuary.
- Run all commands in your Ubuntu WSL2 terminal, not PowerShell.
