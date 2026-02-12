from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import cupy as cp

from ..array3d import BaseArray3D, direction_t
from ...kernels.cupy.cupy_stream import CuPyStream

if TYPE_CHECKING:
    from hip._util.types import Pointer
    from ..array3d import data_t


class CuPyArray3D(BaseArray3D):
    """
    Class that holds 3D CuPy data
    """

    def __init__(self, gpu_stream: CuPyStream, nx: int, ny: int, nc: int, data: data_t,
                 double_precision=False, integers=False):
        """
        Uploads initial data to the HIP device
        """

        super().__init__(gpu_stream, nx, ny, nc, data, double_precision, integers)
        shape_z, shape_y, shape_x = self.shape

        self.height = shape_y

        # Create the array on the device
        self.data: cp.ndarray = cp.asarray(self._host_data)

        # FIXME: This could be potentially dangerous as it could be deleting the entire array before the copy has been completed.
        self._host_data = None

    @property
    def pointer(self) -> Pointer:
        return self.data.data

    @property
    def pitch(self):
        """
        Pitch in the device memory.
        """
        return self.data.strides[1]

    def upload(self, gpu_stream: CuPyStream, data: data_t):
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')

        if np.ma.is_masked(data):
            self.mask = data.mask

        # Make sure that the input is of correct size:
        host_data = self._convert_to_precision(data)

        self._check(host_data.shape, host_data.itemsize)

        # Copy data from CPU to GPU
        self.data.set(data)

    def copy_buffer(self, gpu_stream: CuPyStream, buffer: CuPyArray3D) -> None:
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before copying buffer')

        if not buffer.holds_data:
            raise RuntimeError('The provided buffer is either not allocated, or has been freed before copying buffer')

        self._check(buffer.shape, buffer.bytes_per_float)

        # Okay, everything is fine - issue device-to-device-copy:
        self.data = buffer.data

    def download(self, gpu_stream: CuPyStream) -> np.ndarray:
        """
        Enables downloading data from GPU to Python
        Args:
            gpu_stream: The GPU stream to add the memory copy to.
        Returns:
            ``data`` with the data from the GPU memory.
            Note the data in `cpu_data` may be uninitialized if `asynch` was not set to `True`.
        """

        if not self.holds_data:
            raise RuntimeError('HIP buffer has been freed')

        # Convert cupy array to numpy array
        data = cp.asnumpy(self.data)

        return data

    def release(self) -> None:
        if self.holds_data:
            self.data = None
