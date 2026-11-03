#!/usr/bin/env python3
"""verify_ml_environment.py - Pre-Flight Hardware & Runtime Readiness Auditor for ML Workloads.
Purpose: Audits hardware environment to differentiate CUDA training rigs from Apple Silicon inference.
Layer: sanctuary-core / forging-sovereign-models (Layer 3 Machine Diagnostics).
Architectural Grounding: Fine-tuning requires NVIDIA CUDA / WSL2 (>= 8GB VRAM); Apple Silicon M1
is designated strictly for native local inference (Apple MLX), not resource-starved local fine-tuning.
Key Inputs: Local platform and PyTorch runtime telemetry.
Key Outputs: JSON machine profile, training readiness flag, and architecture recommendation.
Usage:
    python3 verify_ml_environment.py
Key Functions: check_environment(), main().
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
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "cuda_available": False,
        "device_name": "None",
        "device_count": 0,
        "vram_gb": 0.0,
        "pytorch_version": "Not installed",
        "llama_cpp_sibling": False,
        "ready_for_training": False,
        "recommended_role": "inference" if platform.system() == "Darwin" else "unknown"
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
                status["recommended_role"] = "training_and_inference"
    except ImportError:
        pass

    sibling_llama = Path(__file__).resolve().parents[5] / "llama.cpp"
    status["llama_cpp_sibling"] = sibling_llama.exists()

    return status


def main():
    report = check_environment()
    print(json.dumps(report, indent=2))
    if not report["cuda_available"]:
        if report["is_apple_silicon"]:
            print("\n🍎 Apple Silicon (M-series) detected:")
            print("  - Local Inference: FULLY SUPPORTED natively via Apple MLX (mlx-lm) on unified memory.")
            print("  - Model Fine-Tuning: NOT RECOMMENDED locally on 16GB M1 (untenable compute latency & swap pressure).")
            print("  - Architecture Rule: Fine-tuning is reserved for dedicated NVIDIA CUDA / WSL2 rigs (see references/windows-wsl-cuda-setup.md).")
        else:
            print("\n⚠️ CUDA is not available in the current environment.")
            print("Note: QLoRA training requires NVIDIA GPU with CUDA. See references/windows-wsl-cuda-setup.md.")
    else:
        print(f"\n✅ GPU Ready: {report['device_name']} ({report['vram_gb']} GB VRAM)")


if __name__ == "__main__":
    main()
