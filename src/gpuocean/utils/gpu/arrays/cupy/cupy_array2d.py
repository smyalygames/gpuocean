from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import cupy as cp

from ..array2d import BaseArray2D, direction_t
from ...kernels.cupy.cupy_stream import CuPyStream

if TYPE_CHECKING:
    from hip._util.types import Pointer
    from ..array2d import data_t


class CuPyArray2D(BaseArray2D):
    """
    Class that holds 2D HIP data
    """

    def __init__(self, gpu_stream: CuPyStream, nx: int, ny: int, x_halo: int, y_halo: int, data: data_t,
                 asym_halo: list[int] = None, double_precision=False, integers=False, padded=True):
        """
        Uploads initial data to the HIP device
        """

        super().__init__(gpu_stream, nx, ny, x_halo, y_halo, data, asym_halo, double_precision, integers)
        shape_y, shape_x = self.shape

        self._pitch = shape_x * self.bytes_per_float
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
        return self._pitch

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

    def copy_buffer(self, gpu_stream: CuPyStream, buffer: CuPyArray2D) -> None:
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

    def download_boundary(self, gpu_stream: CuPyStream, direction: direction_t) -> data_t:
        if not self.holds_data:
            raise RuntimeError('CuPy buffer has been freed.')

        start, end = self._get_boundary_coordinates(direction)
        # shape = self._get_boundary_shape(direction)

        # FIXME add checks to make sure this does not overflow
        if direction == "north":
            start = (start[0] + self.halo_y, start[1])
            end = (end[0] + self.halo_y, end[1])
        if direction == "east":
            start = (start[0], start[1] - self.halo_x)
            end = (end[0], end[1] - self.halo_x)
        if direction == "south":
            start = (start[0] - self.halo_y, start[1])
            end = (end[0] - self.halo_y, end[1])
        if direction == "west":
            start = (start[0], start[1] + self.halo_x)
            end = (end[0], end[1] + self.halo_x)

        return self.data[start[0]:end[0], start[1]:end[1]].copy()

    def upload_boundary(self, gpu_stream: CuPyStream, data: CuPyArray2D, direction: direction_t) -> None:
        start, end = self._get_boundary_coordinates(direction)
        shape = self._get_boundary_shape(direction)

        # Check that the shape is correct
        if data.shape != shape:
            raise ValueError(f"The shape of the boundary data is not correct. Expected shape: {shape};"
                             f" Shape of passed data: {data.shape}.")

        self.data[start[0]:end[0], start[1]:end[1]] = data

    def release(self) -> None:
        if self.holds_data:
            self.data = None
