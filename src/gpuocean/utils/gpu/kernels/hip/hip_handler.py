from __future__ import annotations
from typing import TYPE_CHECKING
import ctypes

import numpy as np
from hip import hip

from ...hip_utils import hip_check
from ..handler import BaseGPUHandler
from .hip_stream import HIPStream


class HIPHandler(BaseGPUHandler):
    def __init__(self, module: hip.ihipModule_t, function: str, arguments: str):
        super().__init__(module, function, arguments)

        self.kernel = hip_check(hip.hipModuleGetFunction(module, bytes(function, "utf-8")))

    def async_call(self, grid_size: tuple[int, int], block_size: tuple[int, int, int], stream: HIPStream | None,
                   args: list):
        grid = hip.dim3(*grid_size)
        block = hip.dim3(*block_size)

        for i in range(len(args)):
            val = args[i]
            if isinstance(val, np.int64):
                args[i] = ctypes.c_int64(val)
            elif isinstance(val, (int, np.int32)):
                args[i] = ctypes.c_int32(val)
            elif isinstance(val, (float, np.float32, np.float64)):
                args[i] = ctypes.c_float(val)

        args = tuple(args)

        if isinstance(stream, HIPStream):
            hip_stream = stream.pointer
        else:
            hip_stream = None

        hip_check(hip.hipModuleLaunchKernel(
            self.kernel,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=hip_stream,
            kernelParams=None,
            extra=args
        ))

    def call(self, grid_size: tuple[int, int], block_size: tuple[int, int, int], args: list):
        self.async_call(grid_size, block_size, None, args)
