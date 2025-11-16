# ANALYSIS of PAST FAILURES and a Path Forward

**ROOT CAUSE OF ISSUES:** Core Diagnosis: Conflicting Strategies

The fundamental issue is that you have **two different and conflicting methods** for setting up the environment, and you are mixing them.

1.  **The Bash Script (`setup_ml_env_wsl.sh`):** This script is a "wild West" installer. It installs the *latest available* versions of `tensorflow` and `torch`/`torchvision`/`torchaudio` that match your CUDA tag (`cu126`). In your log, this resulted in installing **`torch-2.9.1+cu126`**.
2.  **The Python Script + Requirements (`setup_cuda_env.py` + `requirements.txt`):** This is a "precision" installer. It is designed to read a specific "blueprint" (`requirements.txt`) and install the *exact pinned versions* from that file. Your `requirements.txt` specifies **`torch==2.8.0+cu126`** and a matching set of libraries.
3.  **The Manual `pip install` (Surgical Strike):** This is a third, separate instruction. After running the bash script (which installed `torch-2.9.1`), you manually installed `transformers` and other libraries. These libraries, when resolved by `pip`, decided they were more compatible with a generic `torch-2.9.0` (without CUDA support), leading to the downgrade and the error you saw.

The `setup_cuda_env.py` script (the "Foreman") is the architecturally superior approach. It correctly installs the specific CUDA-enabled PyTorch *first*, before installing everything else that depends on it. Your manual process does the opposite, which confuses `pip`.

### The Unified Strategy (Recommended Path Forward)

To resolve this permanently and create a stable, reproducible environment, we will abandon the `setup_ml_env_wsl.sh` script and your manual `pip install` step in favor of using *only* the Python "Foreman" script with an updated blueprint.


---

## OLDER APPROACHES
Summary of othe rapproaches tested and failed
---

### OLD APPROACH 1:  NO CUDA SCRIPT SETUP STEPS EACH TIME

#### 1. Run ML Environment Setup 
Run the setup script from your project root directory. If not already done. 
Run from project root.  Assumes this project and  ML-Env-CUDA13 are in the same parent
directory. 

##### 1a. Purge old environment to have a clean environment for project

From your base WSL shell (no `(ml_env)` in the prompt), run:
```bash
deactivate 2>/dev/null || true
rm -rf ~/ml_env
```

##### 1b. Setup a fresh cuda optimized environment
This will create and configure a Python virtual environment at `~/ml_env`.

```bash
# run this from the Project_Sanctuary repository root
bash ../ML-Env-CUDA13/setup_ml_env_wsl.sh
```
##### 1c. Activate the new, clean environment
```bash
source ~/ml_env/bin/activate
```

---

#### 2. Run test scripts to verify environment working
Run the optional diagnostic tests (recommended):

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

---

#### 3. Install Fine-Tuning Libraries. SURGICAL STRIKE: Install a known-good, compatible fine-tuning stack.

**✅ PREDONDITIONS:** 
1.  CUDA is already verified working
2.  all test scripts passed above

This stack is chosen to work with the PyTorch version installed by the bash script.

```bash
pip install "transformers==4.41.2" "peft==0.10.0" "trl==0.8.6" "bitsandbytes==0.43.1" "datasets==2.19.0" "accelerate==0.30.1" "xformers" "tf-keras"
```

---

#### 4.  The Final Verification
Run the diagnostic key.
Execute this command:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
The output must be True.

---

### OLD APPROACH 2: USE ML-Env-CUDA13 envionrment, local cuda script and requirements.txt

#### 1. run cuda setup
```bash
deactivate
python3 forge/OPERATION_PHOENIX_FORGE/scripts/setup_cuda_env.py --staged
source ~/ml_env/bin/activate
```

#### 2.  verify cuda
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. run tests again
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
