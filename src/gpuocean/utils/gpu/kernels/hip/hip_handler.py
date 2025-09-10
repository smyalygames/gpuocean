from typing import TYPE_CHECKING
import ctypes

import numpy as np
from hip import hip

from ...hip_utils import hip_check
from ..handler import BaseGPUHandler

if TYPE_CHECKING:
    from ... import KernelContext, GPUStream


class HIPHandler(BaseGPUHandler):
    def __init__(self, module: hip.ihipModule_t, function: str, arguments: str):
        super().__init__(module, function, arguments)

        self.kernel = hip_check(hip.hipModuleGetFunction(module, bytes(function, "utf-8")))

    def call(self, grid_size: tuple[int, int], block_size: tuple[int, int, int], stream: GPUStream,
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
