import json
import subprocess
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
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
    print("TensorFlow build info:")
    print(json.dumps({
        'tf_version': tf.__version__,
        'cuda_build': cuda_build,
        'cudnn_build': cudnn_build,
    }, indent=2))
except Exception as e:
    print("Could not retrieve TensorFlow build info:", e)

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        return out.strip()
    except Exception as e:
        return f"Error running {' '.join(cmd)}: {e}"

print('\nSystem NVIDIA / CUDA info (nvidia-smi, nvcc)')
print(run_cmd(['nvidia-smi']))
nvcc_out = run_cmd(['nvcc','--version'])
if 'Error running' in nvcc_out:
    print('nvcc not on PATH or not installed in WSL')
else:
    print(nvcc_out)
