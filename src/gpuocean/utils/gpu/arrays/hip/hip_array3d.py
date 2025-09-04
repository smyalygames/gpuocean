from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from hip import hip

from ...hip_utils import hip_check
from ..array3d import BaseArray3D
from .enums import Host, Device, Transfer

if TYPE_CHECKING:
    from ..array3d import data_t


class HIPArray3D(BaseArray3D):
    """
    Class that holds 2D HIP data
    """

    def __init__(self, gpu_stream: hip.ihipStream_t, nx: int, ny: int, nc: int, data: data_t,
                 double_precision=False, integers=False):
        """
        Uploads initial data to the HIP device
        """

        super().__init__(gpu_stream, nx, ny, nc, data, double_precision, integers)
        shape_x = self.shape[0]
        shape_y = self.shape[1]
        shape_z = self.shape[2]

        self.extent = hip.hipExtent(width=shape_x, height=shape_y, depth=shape_z)

        # Pointer for the memory
        self.data = hip.hipPitchedPtr()
        # Allocating memory
        hip.hipMalloc3D(self.data, self.extent)

        src_pos = hip.hipPos(x=0, y=0, z=0)

        copy_params = hip.hipMemcpy3DParms(srcArray=self.__host_data, srcPos=src_pos, srcPtr=)

        hip_check(hip.hipMemcpy3DAsync(params, gpu_stream))

    def __del__(self, *args):
        hip_check(hip.hipFree(self.data))

    def upload(self, gpu_stream: hip.ihipStream_t, data: data_t):
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')

        if np.ma.is_masked(data):
            self.mask = data.mask

        # Make sure that the input is of correct size:
        host_data = self.__convert_to_precision(data)

        self.__check(host_data.shape, host_data.itemsize)

        # Parameters to copy to GPU memory
        src = Host(data)
        dst = Device(self.data, self.pitch_d, self.dtype)
        transfer = Transfer(src, dst, self.width, self.height)
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream))

    def copy_buffer(self, gpu_stream, buffer: HIPArray3D) -> None:
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before copying buffer')

        if not buffer.holds_data:
            raise RuntimeError('The provided buffer is either not allocated, or has been freed before copying buffer')

        self.__check((buffer.ny_halo, buffer.nx_halo), buffer.bytes_per_float)

        # Okay, everything is fine - issue device-to-device-copy:
        src = Device(self.data, self.pitch_d, self.dtype)
        dst = Device(buffer.data, buffer.pitch_d, buffer.dtype)
        transfer = Transfer(src, dst, self.width, self.height)
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream))

    def download(self, gpu_stream: hip.ihipStream_t) -> np.ndarray:
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

        data = np.zeros(self.shape, dtype=self.dtype)

        # Parameters to copy to GPU memory
        src = Device(self.data, self.pitch_d, self.dtype)
        dst = Host(data)
        transfer = Transfer(src, dst, self.width, self.height)
        copy = transfer.get_transfer()

        hip_check(hip.hipMemcpyParam2DAsync(copy, gpu_stream))

        return data
