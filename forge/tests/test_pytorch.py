#============================================
# forge/tests/test_pytorch.py
# Purpose: Diagnostic script to verify PyTorch installation, CUDA availability, and system environment.
# Role: Environment Verification / Diagnostic Layer
# Used by: Phase 1 of the Forge Pipeline
#============================================

import sys
import json
import subprocess
from typing import Dict, Any, Optional

import torch

#============================================
# Function: run_cmd
# Purpose: Executes a shell command and returns the output or error message.
# Args:
#   cmd (list): The command and its arguments.
# Returns: (str) Standard output or error string.
#============================================
def run_cmd(cmd: list) -> str:
    """
    Executes a shell command and returns the output or error message.

    Args:
        cmd: List of command arguments.

    Returns:
        The command output as a string.
    """
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        return out.strip()
    except Exception as e:
        return f"Error running {' '.join(cmd)}: {e}"


#============================================
# Function: main
# Purpose: Main entry point for diagnostic tests.
# Args: None
# Returns: None
#============================================
def main() -> None:
    """
    Main diagnostic check for PyTorch and CUDA.
    """
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available: bool = torch.cuda.is_available()
    print(f"GPU Detected: {cuda_available}")
    
    gpu_name: str = "None"
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = repr(torch.cuda.current_device())
    print(f"GPU 0: {gpu_name}")

    # Build info
    cuda_build: Optional[str] = None
    try:
        cuda_build = getattr(torch.version, 'cuda', None) or torch.version.cuda
    except Exception:
        cuda_build = None
        
    cudnn_build: Optional[int] = None
    try:
        cudnn_build = torch.backends.cudnn.version()
    except Exception:
        cudnn_build = None

    build_info: Dict[str, Any] = {
        'torch_version': torch.__version__,
        'cuda_build': cuda_build,
        'cudnn_build': cudnn_build,
    }

    print("\nPyTorch Build Info:")
    print(json.dumps(build_info, indent=2))

    print('\nSystem NVIDIA / CUDA Info (nvidia-smi, nvcc):')
    print(run_cmd(['nvidia-smi']))
    
    nvcc_out: str = run_cmd(['nvcc', '--version'])
    if 'Error running' in nvcc_out:
        print('nvcc not on PATH or not installed in WSL')
    else:
        print(nvcc_out)


if __name__ == "__main__":
    main()
