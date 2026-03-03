from os import environ

__env_name = 'GPU_LANG'

if __env_name in environ and environ.get(__env_name).lower() == "cuda":
    from .cuda.cuda_device import cuda_device_count as gpu_device_count
else:
    from .hip.hip_device import hip_device_count as gpu_device_count
