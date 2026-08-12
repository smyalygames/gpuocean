from __future__ import annotations
import ctypes
from collections.abc import Iterable

import cupy as cp
import numpy as np
from hip import hip
from hip._util import types

from ...hip_utils import hip_check
from ..handler import BaseGPUHandler
from .hip_stream import HIPStream


class HIPHandler(BaseGPUHandler):
    def __init__(self, module: hip.ihipModule_t, function: str, arguments: str):
        super().__init__(module, function, arguments)

        self.kernel = hip_check(hip.hipModuleGetFunction(module, bytes(function, "utf-8")))

    def async_call(self, grid_size: tuple[int, int], block_size: tuple[int, int, int], stream: HIPStream | None,
                   args: list, exchange=True, exchange_exclude: Iterable = None):
        grid = hip.dim3(*grid_size)
        block = hip.dim3(*block_size)

        pointers: list[types.Pointer | cp.cuda.MemoryPointer] = []

        if exchange_exclude is None:
            exchange_exclude = []

        for i in range(len(args)):
            val = args[i]
            if isinstance(val, int):
                args[i] = ctypes.c_int32(val)
            elif isinstance(val, float):
                args[i] = ctypes.c_float(val)
            elif isinstance(val, (np.ndarray, np.generic)):
                if getattr(val, 'size', 1) != 1:
                    raise RuntimeError(f"Called a GPU function with a CPU array of size {val.size}.")

                dtype = val.dtype if hasattr(val, 'dtype') else np.dtype(type(val))
                clean_val = val.item()

                match dtype:
                    case np.float32:
                        args[i] = ctypes.c_float(clean_val)
                    case np.float64:
                        args[i] = ctypes.c_double(clean_val)
                    case np.int32:
                        args[i] = ctypes.c_int32(clean_val)
                    case np.int64:
                        args[i] = ctypes.c_int64(clean_val)
                    case np.uint32:
                        args[i] = ctypes.c_uint32(clean_val)
                    case np.uint64:
                        args[i] = ctypes.c_uint64(clean_val)
                    case _:
                        args[i] = np.ctypeslib.as_ctypes(clean_val)

            elif isinstance(val, types.Pointer) and val not in exchange_exclude:
                pointers.append(val)
            elif isinstance(val, cp.cuda.MemoryPointer):
                args[i] = types.Pointer(val.ptr)
                if val not in exchange_exclude:
                    pointers.append(val)

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

        # Exchange arrays after starting the kernel
        if exchange:
            self.exchange(pointers)

    def call(self, grid_size: tuple[int, int], block_size: tuple[int, int, int], args: list,
             exchange=True, exchange_exclude: Iterable = None):
        self.async_call(grid_size, block_size, None, args, exchange, exchange_exclude)
