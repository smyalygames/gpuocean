from os import environ
from typing import TYPE_CHECKING

__env_name = 'GPU_LANG'

if __env_name in environ and environ.get(__env_name).lower() == "cuda":
    from .arrays.cuda.cuda_array2d import CudaArray2D as Array2D
    from .arrays.cuda.cuda_array3d import CudaArray3D as Array3D
    from .kernels.cuda.cuda_context import CudaContext as KernelContext
    from .kernels.cuda.cuda_handler import CudaHandler as GPUHandler
    from .kernels.cuda.cuda_event import CudaEvent as Event
    from .kernels.cuda.cuda_stream import CudaStream as GPUStream
    from .utils.cuda.cuda_random_numbers import CudaRandomNumbers as XORWOWHandler

    if TYPE_CHECKING:
        from pycuda import driver
        stream_t = driver.Stream
        module_t = driver.Module
else:
    # from .arrays.hip.hip_array2d import HIPArray2D as Array2D
    from .arrays.cupy.cupy_array2d import CuPyArray2D as Array2D
    from .arrays.hip.hip_array3d import HIPArray3D as Array3D
    from .kernels.hip.hip_context import HIPContext as KernelContext
    from .kernels.hip.hip_handler import HIPHandler as GPUHandler
    from .kernels.hip.hip_event import HIPEvent as Event
    from .kernels.hip.hip_stream import HIPStream as GPUStream
    from .utils.hip.hip_random_numbers import HIPRandomNumbers as XORWOWHandler

    if TYPE_CHECKING:
        from hip import hip
        stream_t = hip.hipStream_t
        module_t = hip.ihipModule_t

# Objects that are not dependent on GPU language
from .arrays.arkawa import SWEDataArakawaA, SWEDataArakawaC
from .arrays.bc_arkawa import BoundaryConditionsArakawaA
