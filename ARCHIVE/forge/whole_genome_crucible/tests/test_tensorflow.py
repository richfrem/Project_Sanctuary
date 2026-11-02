#============================================
# forge/tests/test_tensorflow.py
# Purpose: Diagnostic script to verify TensorFlow installation and GPU visibility.
# Role: Environment Verification / Diagnostic Layer
# Used by: Phase 1 of the Forge Pipeline
#============================================

import sys
import json
import subprocess
from typing import List, Dict, Any, Optional

import tensorflow as tf

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
    Main diagnostic check for TensorFlow and CUDA.
    """
    print(f"TensorFlow Version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Detected: {len(gpus) > 0}")
    
    for i, gpu in enumerate(gpus):
        try:
            name = gpu.name
        except Exception:
            name = repr(gpu)
        print(f"GPU {i}: {name}")

    try:
        build = tf.sysconfig.get_build_info()
        cuda_build = build.get('cuda_version') or build.get('cuda_version_text') or None
        cudnn_build = build.get('cudnn_version') or None
        
        print("\nTensorFlow Build Info:")
        print(json.dumps({
            'tf_version': tf.__version__,
            'cuda_build': cuda_build,
            'cudnn_build': cudnn_build,
        }, indent=2))
    except Exception as e:
        print(f"Could not retrieve TensorFlow build info: {e}")

    print('\nSystem NVIDIA / CUDA Info (nvidia-smi, nvcc):')
    print(run_cmd(['nvidia-smi']))
    
    nvcc_out: str = run_cmd(['nvcc', '--version'])
    if 'Error running' in nvcc_out:
        print('nvcc not on PATH or not installed in WSL')
    else:
        print(nvcc_out)


if __name__ == "__main__":
    main()
