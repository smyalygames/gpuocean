from hip import hip

from ...hip_utils import hip_check


def hip_device_count() -> int:
    """
    Gets the total amount of available GPUs that can be run HIP on the system.
    :returns: Number of HIP GPUs available.
    """
    return hip_check(hip.hipGetDeviceCount())
