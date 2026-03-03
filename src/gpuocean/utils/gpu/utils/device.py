from . import gpu_device_count


def get_device_count() -> int:
    """
    Gets the total amount of available GPUs on the system.
    :returns: Number of GPUs available.
    """
    return gpu_device_count()


def get_device_for_rank(rank: int) -> int:
    """
    Shares the devices between ranks.
    :param rank: MPI rank of the process.
    :returns: GPU device number.
    """
    return rank % get_device_count()
