# Windows WSL2 & CUDA ML Environment Setup

> **Canonical learning from Project Sanctuary's sovereign fine-tuning runs on consumer NVIDIA hardware.**

## 1. System Prerequisites

1. **WSL2 & Ubuntu**: Ensure Windows 11 with WSL2 (`Ubuntu 22.04` or `24.04`) is installed.
2. **NVIDIA Driver Pass-Through**: Install the latest Windows NVIDIA Game Ready or Studio driver. **Never** install Linux display drivers inside WSL2.
3. **GPU Verification**: Inside WSL2 terminal:
   ```bash
   nvidia-smi
   ```
   Must display NVIDIA GPU, driver version, and CUDA version (e.g., CUDA 12.6+).

## 2. llama.cpp Build (The Engine)

The GGUF quantization toolchain requires `llama.cpp` as a sibling repository (`../llama.cpp`):
```bash
# Clone sibling repo
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp

# Compile with CUDA acceleration
cd ../llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ../Project_Sanctuary
```

## 3. Surgical Python CUDA Installation Protocol

Standard `pip install` often pulls CPU-only wheels or causes binary ABI mismatches. Follow this deterministic installation order:

```bash
# 1. Create dedicated isolated environment
python3.11 -m venv ~/ml_env
source ~/ml_env/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Install PyTorch with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. Verify PyTorch CUDA binding
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"

# 4. Install Surgical CUDA Binaries in order
pip install triton==3.5.0
pip install bitsandbytes==0.48.2
pip install xformers --index-url https://download.pytorch.org/whl/cu126

# 5. Install Hugging Face & Training Libraries
pip install transformers peft accelerate datasets trl
pip install gguf sentencepiece
```

## 4. RAM & Swap Space Requirement (LoRA Merge Step)

Merging the LoRA adapter into the base model weights runs on CPU to prevent VRAM Out-Of-Memory (OOM) on 8GB-12GB consumer cards. This requires ~20-24GB system RAM:

```bash
# If system RAM is less than 32GB, create 16GB swap file in WSL2:
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```
