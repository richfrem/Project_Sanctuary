#============================================
# forge/tests/test_torch_cuda.py
# Purpose: Minimal smoke test to confirm PyTorch can access the GPU.
# Role: Diagnostic / Smoke Test
# Used by: Phase 1 of the Forge Pipeline
#============================================

import sys
from typing import Optional

import torch

#============================================
# Function: main
# Purpose: Performs a quick check on PyTorch CUDA connectivity.
# Args: None
# Returns: None
# Raises: SystemExit with code 2 if CUDA is unavailable.
#============================================
def main() -> None:
    """
    Main smoke test for PyTorch CUDA availability.
    """
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available: bool = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        try:
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("CUDA Device Name: Unknown")
            
        try:
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        except Exception:
            print("cuDNN Version: Unknown")
    else:
        print("🛑 FATAL ERROR: CUDA is NOT available to PyTorch.")
        sys.exit(2)


if __name__ == "__main__":
    main()
