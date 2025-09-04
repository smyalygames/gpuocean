import ctypes

import numpy as np
from hip import hip, hipblas

from ...hip_utils import hip_check
from ..handler import BaseGPUHandler
from gpuocean.utils.gpu import KernelContext


class HIPHandler(BaseGPUHandler):
    def __init__(self, context: KernelContext, module, function, arguments,
                 grid_size):
        super().__init__(context, module, function, arguments, grid_size)

        self.kernel = hip_check(hip.hipModuleGetFunction(module, bytes(function, "utf-8")))
        self.context = context

        self.dtype = np.float32
        self.cfl_data_h = np.empty(grid_size, dtype=self.dtype)

        self.num_bytes = self.cfl_data_h.size * self.cfl_data_h.itemsize
        self.cfl_data = hip_check(hip.hipMalloc(self.num_bytes)).configure(
            typestr=self.cfl_data_h.dtype.str, shape=grid_size
        )

    def __del__(self):
        hip_check(hip.hipFree(self.cfl_data))

    def prepared_call(self, grid_size: tuple[int, int], block_size: tuple[int, int, int], stream: hip.ihipStream_t,
                      args: list):
        grid = hip.dim3(*grid_size)
        block = hip.dim3(*block_size)

        for i in range(len(args)):
            val = args[i]
            if isinstance(val, np.int64):
                args[i] = ctypes.c_int64(val)
            elif isinstance(val, int) or isinstance(val, np.int32):
                args[i] = ctypes.c_int(val)
            elif isinstance(val, float) or isinstance(val, np.float32):
                args[i] = ctypes.c_float(val)

        args = tuple(args)

        hip_check(hip.hipModuleLaunchKernel(
            self.kernel,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=stream,
            kernelParams=None,
            extra=args
        ))

    def array_fill(self, data: float, stream: hip.ihipStream_t):
        self.cfl_data_h.fill(data)

        hip_check(
            hip.hipMemcpyAsync(self.cfl_data, self.cfl_data_h, self.num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice,
                               stream))

    def array_min(self, stream: hip.ihipStream_t) -> float:
        handle = hip_check(hipblas.hipblasCreate())

        value_h = np.empty(1, self.dtype)
        value_d = hip_check(hip.hipMalloc(value_h.itemsize))

        hip_check(hipblas.hipblasIsamin(handle, self.cfl_data.size, self.cfl_data, 1, value_d))
        hip_check(hipblas.hipblasDestroy(handle))

        hip_check(
            hip.hipMemcpy(value_h, self.cfl_data, self.cfl_data_h.itemsize, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

        return value_h[0]
