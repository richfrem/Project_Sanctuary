import torch
import sys
print('torch.__version__ =', torch.__version__)
print('cuda_available =', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count =', torch.cuda.device_count())
    try:
        print('cuda_device_name =', torch.cuda.get_device_name(0))
    except Exception:
        print('cuda_device_name = unknown')
    try:
        print('cudnn_version =', torch.backends.cudnn.version())
    except Exception:
        print('cudnn_version = unknown')
else:
    sys.exit(2)
