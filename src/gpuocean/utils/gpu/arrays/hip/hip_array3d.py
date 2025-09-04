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
        hip_check(hip.hipMalloc3D(self.data, self.extent))

        self.__pos = hip.hipPos(x=0, y=0, z=0)

        copy_params = hip.hipMemcpy3DParms(srcPos=self.__pos, srcPtr=self.__host_data,
                                           dstPos=self.__pos, dstPtr=self.data,
                                           kind=hip.hipMemcpyKind.hipMemcpyHostToDevice)

        hip_check(hip.hipMemcpy3DAsync(copy_params, gpu_stream))

        # FIXME: This may be dangerous, as the array may be deleted when the memory copy to the GPU occurs.
        #  This should be tested.
        self.__host_data = None

    def upload(self, gpu_stream: hip.ihipStream_t, data: data_t):
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')

        if np.ma.is_masked(data):
            self.mask = data.mask

        # Make sure that the input is of correct size:
        host_data = self.__convert_to_precision(data)

        self.__check(host_data.shape, host_data.itemsize)

        # Parameters to copy to GPU memory
        copy_params = hip.hipMemcpy3DParms(srcPos=self.__pos, srcPtr=data,
                                           dstPos=self.__pos, dstPtr=self.data,
                                           kind=hip.hipMemcpyKind.hipMemcpyHostToDevice)

        hip_check(hip.hipMemcpy3DAsync(copy_params, gpu_stream))

    def copy_buffer(self, gpu_stream, buffer: HIPArray3D) -> None:
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before copying buffer')

        if not buffer.holds_data:
            raise RuntimeError('The provided buffer is either not allocated, or has been freed before copying buffer')

        self.__check(buffer.shape[::-1], buffer.bytes_per_float)

        # Okay, everything is fine - issue device-to-device-copy:
        copy_params = hip.hipMemcpy3DParms(srcPos=self.__pos, srcPtr=buffer.data,
                                           dstPos=self.__pos, dstPtr=self.data,
                                           kind=hip.hipMemcpyKind.hipMemcpyDeviceToDevice)

        hip_check(hip.hipMemcpy3DAsync(copy_params, gpu_stream))

    def download(self, gpu_stream: hip.ihipStream_t) -> np.ndarray:
        """
        Enables downloading data from GPU to Python
        Args:
            gpu_stream: The GPU stream to add the memory copy to.
        Returns:
            A numpy array with the data from the GPU memory.
            Note the data in `cpu_data` may be uninitialized if `asynch` was not set to `True`.
        """

        if not self.holds_data:
            raise RuntimeError('HIP buffer has been freed')

        data_h = np.zeros(self.shape, dtype=self.dtype)

        # Parameters to copy from GPU memory
        copy_params = hip.hipMemcpy3DParms(srcPos=self.__pos, srcPtr=self.data,
                                           dstPos=self.__pos, dstPtr=data_h,
                                           kind=hip.hipMemcpyKind.hipMemcpyDeviceToHost)

        hip_check(hip.hipMemcpy3DAsync(copy_params, gpu_stream))

        return data_h

    def release(self):
        hip_check(hip.hipFree(self.data))

