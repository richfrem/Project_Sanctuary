import json
import subprocess
import torch

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        return out.strip()
    except Exception as e:
        return f"Error running {' '.join(cmd)}: {e}"

print(f"PyTorch: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"GPU Detected: {cuda_available}")
if cuda_available:
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = repr(torch.cuda.current_device())
else:
    gpu_name = 'None'
print(f"GPU 0: {gpu_name}")

# Build info
cuda_build = None
try:
    cuda_build = getattr(torch.version, 'cuda', None) or torch.version.cuda
except Exception:
    cuda_build = None
try:
    cudnn_build = torch.backends.cudnn.version()
except Exception:
    cudnn_build = None

build_info = {
    'torch_version': torch.__version__,
    'cuda_build': cuda_build,
    'cudnn_build': cudnn_build,
}

print("\nPyTorch build info:")
print(json.dumps(build_info, indent=2))

print('\nSystem NVIDIA / CUDA info (nvidia-smi, nvcc)')
print(run_cmd(['nvidia-smi']))
nvcc_out = run_cmd(['nvcc', '--version'])
if 'Error running' in nvcc_out:
    print('nvcc not on PATH or not installed in WSL')
else:
    print(nvcc_out)
