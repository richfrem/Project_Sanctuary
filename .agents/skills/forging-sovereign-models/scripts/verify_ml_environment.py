#!/usr/bin/env python3
"""
verify_ml_environment.py — Pre-flight readiness check for fine-tuning.
"""

import sys
import json
import shutil
import platform
from pathlib import Path


def check_environment():
    status = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": False,
        "device_name": "None",
        "device_count": 0,
        "vram_gb": 0.0,
        "pytorch_version": "Not installed",
        "llama_cpp_sibling": False,
        "ready_for_training": False
    }

    try:
        import torch
        status["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            status["cuda_available"] = True
            status["device_count"] = torch.cuda.device_count()
            status["device_name"] = torch.cuda.get_device_name(0)
            status["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if status["vram_gb"] >= 7.5:
                status["ready_for_training"] = True
    except ImportError:
        pass

    sibling_llama = Path(__file__).resolve().parents[5] / "llama.cpp"
    status["llama_cpp_sibling"] = sibling_llama.exists()

    return status


def main():
    report = check_environment()
    print(json.dumps(report, indent=2))
    if not report["cuda_available"]:
        print("\n⚠️ CUDA is not available in the current environment.")
        print("Note: QLoRA training requires NVIDIA GPU with CUDA. See references/windows-wsl-cuda-setup.md.")
    else:
        print(f"\n✅ GPU Ready: {report['device_name']} ({report['vram_gb']} GB VRAM)")


if __name__ == "__main__":
    main()
