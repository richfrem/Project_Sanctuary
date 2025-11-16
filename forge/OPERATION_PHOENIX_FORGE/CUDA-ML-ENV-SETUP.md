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

---

## **Project Environment Setup (Unified Script)**

This process uses a single, purpose-built Python script (`setup_cuda_env.py v2.1`) that handles everything from system prerequisites to installing the specific Python libraries required for this project. This replaces the need for the `setup_ml_env_wsl.sh` script.

It will setup a custom cuda environment for NVDIA GPU as a unified script specifically
for fine-tuning in this project. 

---

### Step 0: Start with a Clean Slate

As you correctly did in your own steps, ensure no old environment exists.

```bash
deactivate 2>/dev/null || true
rm -rf ~/ml_env
```

### **Step 1: Configure the Blueprint (`requirements.txt`)**

This file acts as the master plan for your Python environment. Ensure the `requirements.txt` file in your project's root directory contains the exact contents below. This defines the precise, known-good versions of all libraries.

```text
# Use the PyTorch CUDA 12.6 wheel index. This is the known-good index.
--extra-index-url https://download.pytorch.org/whl/cu126

# --- CORE ML & FINE-TUNING STACK ---
# Pinned to PyTorch 2.9.0 to match xformers dependency.
torch==2.9.0+cu126
torchvision==0.24.0+cu126
torchaudio==2.9.0+cu126

# Pinned to a known-good, compatible set for the PyTorch version above.
transformers==4.41.2
peft==0.10.0
trl==0.8.6
bitsandbytes==0.43.1
datasets==2.19.0
accelerate==0.30.1
xformers

# Keras compatibility layer
tf-keras==2.20.1

# --- CORE RAG & ORCHESTRATOR STACK ---
langchain
chromadb
google-generativeai
ollama
gpt4all
```



### **Step 2: Run the All-in-One Setup Script**

From your project's root directory, run the setup script. This single command will configure your entire environment.

> **Note:** You must run this command with `sudo` because the script will automatically install necessary system packages (like `python3.11` and `python3.11-venv`) if they are missing.

```bash
sudo python3 forge/OPERATION_PHOENIX_FORGE/scripts/setup_cuda_env.py --staged --recreate
```

This command will perform the following actions:
1.  Ask for your `sudo` password.
2.  Install system prerequisites like Python 3.11 if they are not already present.
3.  Remove any old `~/ml_env` because of the `--recreate` flag.
4.  Create a new, clean Python 3.11 virtual environment at `~/ml_env`, owned by **your user**.
5.  Install the exact Python libraries specified in your `requirements.txt`.

### **Step 3: Activate and Verify**

Once the script completes successfully, activate the new environment.

```bash
source ~/ml_env/bin/activate
```
Your command prompt will now start with `(ml_env)`.

Finally, run the verification script to confirm that PyTorch can see your GPU:
```bash
(ml_env) $ python forge/OPERATION_PHOENIX_FORGE/scripts/test_torch_cuda.py
```

The expected output should show `torch.__version__ = 2.9.1+cu126` and `cuda_available = True`, confirming your setup is successful.

---

### Why This Streamlined Approach Works

*   **Single Command:** The new script handles system dependencies and Python packages in one step, reducing complexity.
*   **Single Source of Truth:** Your environment is built from one file: `requirements.txt`. There is no conflicting logic from other scripts or manual commands.
*   **Staged Installation:** The script installs the most critical package (PyTorch with CUDA) *first* and in isolation, preventing `pip` from making incorrect dependency decisions.
*   **Reproducibility:** Anyone can now perfectly recreate your environment on a similar machine with this single, powerful command.

By adopting this unified strategy, you will have a stable and powerful environment for your A2000 GPU.

---
