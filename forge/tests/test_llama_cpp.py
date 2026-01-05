#============================================
# forge/tests/test_llama_cpp.py
# Purpose: Diagnostic script to verify llama-cpp-python installation and GPU offload support.
# Role: Environment Verification / Diagnostic Layer
# Used by: Phase 1 of the Forge Pipeline
#============================================

import sys
from typing import Optional

#============================================
# Function: main
# Purpose: Validates llama-cpp-python import and CUDA support.
# Args: None
# Returns: None
# Raises: RuntimeError if CUDA support is not detected.
#============================================
def main() -> None:
    """
    Diagnostic check for llama-cpp-python.
    """
    try:
        import llama_cpp
        print("✅ llama_cpp import successful.")

        # Verify CUDA support in the bridge
        cuda_supported: bool = llama_cpp.llama_supports_gpu_offload()
        print(f"llama-cpp-python CUDA support: {cuda_supported}")
        
        if not cuda_supported:
            raise RuntimeError("llama-cpp-python was not built with CUDA support. Offloading will fail.")
        else:
            print("🚀 GPU offloading is supported.")

    except ImportError:
        print("❌ Error: llama-cpp-python not found. Install it with CUDA arguments.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ llama-cpp-python test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
