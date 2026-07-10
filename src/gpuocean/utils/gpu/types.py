from __future__ import annotations
from typing import TYPE_CHECKING, Union

# GPU Streams
if TYPE_CHECKING:
    from .kernels.cuda.cuda_stream import CudaStream
    from .kernels.cupy.cupy_stream import CuPyStream
    from .kernels.hip.hip_stream import HIPStream

GPUStream = Union[CudaStream, CuPyStream, HIPStream]

# Arrays
if TYPE_CHECKING:
    from .arrays.cuda.cuda_array2d import CudaArray2D
    from .arrays.cuda.cuda_array3d import CudaArray3D

    from .arrays.cupy.cupy_array2d import CuPyArray2D
    from .arrays.cupy.cupy_array3d import CuPyArray3D

    from .arrays.hip.hip_array2d import HIPArray2D
    from .arrays.hip.hip_array3d import HIPArray3D

Array2D = Union[CudaArray2D, CuPyArray2D, HIPArray2D]
"""
Union of the class :class:`~.arrays.array2d.BaseArray2d`.
"""
Array3D = Union[CudaArray3D, CuPyArray3D, HIPArray3D]