from typing import TYPE_CHECKING

import numpy as np
from pycuda import gpuarray

from ..handler import BaseGPUHandler

if TYPE_CHECKING:
    import pycuda.driver as cuda
    from ... import KernelContext


class CudaHandler(BaseGPUHandler):
    def __init__(self, module: cuda.Module, function: str, arguments: str):
        super().__init__(module, function, arguments)

        self.arguments = arguments

        self.kernel = module.get_function(function)
        self.kernel.prepare(arguments)

    def call(self, grid_size, block_size, stream, args: list):
        # if len(args) != len(self.arguments):
        #     raise ValueError("The parameters do not match the defined arguments.")

        self.kernel.prepared_async_call(grid_size, block_size, stream, *args)
