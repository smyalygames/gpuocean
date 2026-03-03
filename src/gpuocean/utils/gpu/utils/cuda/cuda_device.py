import pycuda.driver as cuda


def cuda_device_count() -> int:
    """
    Gets the total amount of available GPUs that can be run CUDA on the system.
    :returns: Number of CUDA GPUs available.
    """
    return cuda.Device.count()
