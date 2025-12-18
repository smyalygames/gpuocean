from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import numpy as np
from hip import hip
from hip._util.types import Pointer

from ...hip_utils import hip_check
from ..array2d import BaseArray2D
from .enums import Host, Device, Transfer
from ...kernels.hip.hip_stream import HIPStream

if TYPE_CHECKING:
    from ..array2d import data_t


class HIPArray2D(BaseArray2D):
    """
    Class that holds 2D HIP data
    """

    def __init__(self, gpu_stream: HIPStream, nx: int, ny: int, x_halo: int, y_halo: int, data: data_t,
                 asym_halo: list[int] = None, double_precision=False, integers=False, padded=True):
        """
        Uploads initial data to the HIP device
        """

        super().__init__(gpu_stream, nx, ny, x_halo, y_halo, data, asym_halo, double_precision, integers)
        shape_y, shape_x = self.shape

        self.width = shape_x * self.bytes_per_float
        self.height = shape_y

        self.num_bytes = self.width * self.height

        # Checks if a padded 2D array is wanted or not
        if padded:
            malloc: tuple[Pointer, int] = hip_check(hip.hipMallocPitch(self.width, self.height))
            self.data, self.pitch = malloc
        else:
            self.pitch = self.width
            self.data = hip_check(hip.hipMalloc(self.num_bytes))

        hip_check(hip.hipMemcpy2DAsync(dst=self.pointer, dpitch=self.pitch,
                                       src=self._host_data, spitch=self._host_data.strides[0],
                                       width=self.width, height=self.height,
                                       kind=hip.hipMemcpyKind.hipMemcpyHostToDevice, stream=gpu_stream.pointer))

        # FIXME: This could be potentially dangerous as it could be deleting the entire array before the copy has been completed.
        self._host_data = None

    @property
    def pointer(self) -> Pointer:
        return self.data

    def upload(self, gpu_stream: HIPStream, data: data_t):
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')

        if np.ma.is_masked(data):
            self.mask = data.mask

        # Make sure that the input is of correct size:
        host_data = self._convert_to_precision(data)

        self._check(host_data.shape, host_data.itemsize)

        # Parameters to copy to GPU memory
        src = Host(data)
        dst = Device(self.pointer, self.pitch, self.dtype)
        transfer = Transfer(src, dst, self.width, self.height)
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream.pointer))

    def copy_buffer(self, gpu_stream: HIPStream, buffer: HIPArray2D) -> None:
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before copying buffer')

        if not buffer.holds_data:
            raise RuntimeError('The provided buffer is either not allocated, or has been freed before copying buffer')

        self._check(buffer.shape, buffer.bytes_per_float)

        # Okay, everything is fine - issue device-to-device-copy:
        src = Device(buffer.pointer, buffer.pitch, buffer.dtype)
        dst = Device(self.pointer, self.pitch, self.dtype)
        transfer = Transfer(src, dst, self.width, self.height)
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream.pointer))

    def download(self, gpu_stream: HIPStream) -> np.ndarray:
        """
        Enables downloading data from GPU to Python
        Args:
            gpu_stream: The GPU stream to add the memory copy to.
        Returns:
            ``data`` with the data from the GPU memory.
            Note the data in `cpu_data` may be uninitialized if `asynch` was not set to `True`.
        """

        if not self.holds_data:
            raise RuntimeError('HIP buffer has been freed.')

        data = np.zeros(self.shape, dtype=self.dtype)

        # Parameters to copy from GPU memory
        src = Device(self.pointer, self.pitch, self.dtype)
        dst = Host(data)
        transfer = Transfer(src, dst, self.width, self.height)
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream.pointer))

        return data

    def upload_boundary(self, gpu_stream: HIPStream, data, direction) -> None:
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')

        if np.ma.is_masked(data):
            self.mask = data.mask

        # Make sure that the input is of correct size:
        host_data = self._convert_to_precision(data)

        start, _ = self._get_boundary_coordinates(direction)
        shape = self._get_boundary_shape(direction)

        # Check that the shape is correct
        if host_data.shape != shape:
            raise ValueError(f"The shape of the boundary data is not correct. Expected shape: {shape};"
                             f" Shape of passed data: {host_data.shape}.")

        # Parameters to copy to GPU memory
        src = Host(data)
        dst = Device(self.pointer, self.pitch, self.dtype, x=start[1], y=start[0])
        transfer = Transfer(src, dst, shape[1] * self.bytes_per_float, shape[0])
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream.pointer))

    def download_boundary(self, gpu_stream: HIPStream, direction) -> np.ndarray:
        if not self.holds_data:
            raise RuntimeError('HIP buffer has been freed.')

        start, _ = self._get_boundary_coordinates(direction)
        shape = self._get_boundary_shape(direction)

        data = np.zeros(shape, dtype=self.dtype)

        # Parameters to copy from GPU memory
        src = Device(self.pointer, self.pitch, self.dtype, x=start[1], y=start[0])
        dst = Host(data)
        transfer = Transfer(src, dst, shape[1] * self.bytes_per_float, shape[0])
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream.pointer))

        return data

    def release(self) -> None:
        if self.holds_data:
            hip_check(hip.hipFree(self.data))
            self.holds_data = False
