from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pycuda.gpuarray
import pycuda.driver as cuda

from ..array2d import BaseArray2D
from ...kernels.cuda.cuda_stream import CudaStream

if TYPE_CHECKING:
    from ..array2d import data_t


class CudaArray2D(BaseArray2D):
    """
    Class that holds 2D CUDA data
    """

    def __init__(self, gpu_stream: CudaStream, nx: int, ny: int, x_halo: int, y_halo: int, data: data_t,
                 asym_halo: list[int] = None, double_precision=False):
        """
        Uploads initial data to the CUDA device
        """

        super().__init__(gpu_stream, nx, ny, x_halo, y_halo, data, asym_halo, double_precision)

        self.data = pycuda.gpuarray.to_gpu_async(self.__host_data, stream=gpu_stream.pointer)
        self.pitch = np.int32(self.nx_halo * self.bytes_per_float)
        self.__host_data = None

    def __del__(self, *args):
        # self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        self.data.gpudata.free()
        self.data = None

    def upload(self, gpu_stream: CudaStream, data: data_t) -> None:
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')

        if np.ma.is_masked(data):
            self.mask = data.mask

        # Make sure that the input is of correct size:
        host_data = self.__convert_to_precision(data)

        self.__check(host_data.shape, host_data.itemsize)

        # Okay, everything is fine, now upload:
        self.data.set_async(host_data, stream=gpu_stream.pointer)

    def copy_buffer(self, gpu_stream: CudaStream, buffer: CudaArray2D) -> None:
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before copying buffer')

        if not buffer.holds_data:
            raise RuntimeError('The provided buffer is either not allocated, or has been freed before copying buffer')

        self.__check(buffer.shape[::-1], buffer.bytes_per_float)

        # Okay, everything is fine - issue device-to-device-copy:
        total_num_bytes = self.bytes_per_float * self.nx_halo * self.ny_halo
        cuda.memcpy_dtod_async(self.data.ptr, buffer.data.ptr, total_num_bytes, stream=gpu_stream.pointer)

    def download(self, gpu_stream: CudaStream) -> data_t:
        if not self.holds_data:
            raise RuntimeError('CUDA buffer has been freed')

        # Copy data from device to host
        host_data = self.data.get(stream=gpu_stream.pointer)

        if self.mask is not None:
            host_data = np.ma.array(host_data, mask=self.mask)

        return host_data

    def release(self) -> None:
        if self.holds_data:
            self.data.gpudata.free()
            self.holds_data = False
