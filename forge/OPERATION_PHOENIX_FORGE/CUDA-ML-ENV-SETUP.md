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

### 4a. Run ML Environment Setup (If not already done)
Run the setup script from your project directory. If not already done. 
Run from project root.  Assumes this project and  ML-Env-CUDA13 are in the same parent
directory. 

**Purge Old Environments (If Necessary)**

From your base WSL shell (no `(ml_env)` in the prompt), run:
```bash
deactivate 2>/dev/null || true
rm -rf ~/ml_env
```

```bash
# run this from the Project_Sanctuary repository root
bash ../ML-Env-CUDA13/setup_ml_env_wsl.sh
```

This will create and configure a Python virtual environment at `~/ml_env`.

### 4b. IF ML ENV already setup globally, Activate the venv in your current shell with:
```bash
deactivate
source ~/ml_env/bin/activate
```

### 4c. Run test scripts to verify environment working

#### Run the optional diagnostic tests (recommended):
```bash
# From Project_Sanctuary root:
python ../ML-Env-CUDA13/test_pytorch.py > forge/OPERATION_PHOENIX_FORGE/ml_env_logs/test_pytorch.log 2>&1 || true
python ../ML-Env-CUDA13/test_tensorflow.py > forge/OPERATION_PHOENIX_FORGE/ml_env_logs/test_tensorflow.log 2>&1 || true
python ../ML-Env-CUDA13/test_xformers.py > forge/OPERATION_PHOENIX_FORGE/ml_env_logs/test_xformers.log 2>&1 || true
python ../ML-Env-CUDA13/test_llama_cpp.py > forge/OPERATION_PHOENIX_FORGE/ml_env_logs/test_llama_cpp.log 2>&1 || true
python ../ML-Env-CUDA13/test_torch_cuda.py > ml_env_logs/test_torch_cuda.log 2>&1 || true

# If core gate passed and you want a reproducible snapshot locally:
pip freeze > pinned-requirements-$(date +%Y%m%d%H%M).txt
```

### 5. Install Fine-Tuning Libraries

**✅ PREDONDITIONS:  CUDA is already verified working** (all test scripts passed above), so **SKIP this entire section** and proceed directly to training. verify with 4c. Run test scripts to verify environment working

#### 5A.  setup the cuda fine tuning additional dependencies
```bash
deactivate
python3 forge/OPERATION_PHOENIX_FORGE/scripts/setup_cuda_env.py --staged
source ~/ml_env/bin/activate
```

#### 5B.  verify cuda
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

#### 5C. run tests again
```bash
# From Project_Sanctuary root:
python ../ML-Env-CUDA13/test_pytorch.py
python ../ML-Env-CUDA13/test_tensorflow.py
python ../ML-Env-CUDA13/test_xformers.py
python ../ML-Env-CUDA13/test_llama_cpp.py
python ../ML-Env-CUDA13/test_torch_cuda.py
```

This script automates venv setup, PyTorch/TensorFlow installation, testing, and dependency installation. Logs are written to `forge/OPERATION_PHOENIX_FORGE/ml_env_logs/`.
 - After the script finishes it writes an activation helper `activate_ml_env.sh` in the repo root; run:
 - After the script finishes it writes an activation helper at `scripts/activate_ml_env.sh`; run:
  ```bash
  source scripts/activate_ml_env.sh
  # or
  source ~/ml_env/bin/activate
  ```
 - The script captures logs in `ml_env_logs/` and writes a `pinned-requirements-<timestamp>.txt` after a successful core gate.

**Alternative: Forge-Specific Setup Script**
 - For Operation Phoenix Forge specifically, you can also use the forge-local setup script:
   ```bash
   # From the forge directory
   cd forge/OPERATION_PHOENIX_FORGE
   python scripts/setup_cuda_env.py --staged
   ```
 - This will create logs in `forge/OPERATION_PHOENIX_FORGE/ml_env_logs/` instead of the project root.

## Notes and recommendations
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

**Note:**
- Make sure ML-Env-CUDA13 is cloned at the same directory level as Project_Sanctuary.
- Run all commands in your Ubuntu WSL2 terminal, not PowerShell.
