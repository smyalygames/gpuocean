from typing import TYPE_CHECKING

import numpy as np
from pycuda import gpuarray

from ..handler import BaseGPUHandler

if TYPE_CHECKING:
    from pycuda import driver
    from ... import KernelContext


class CudaHandler(BaseGPUHandler):
    def __init__(self, context: KernelContext, module, function, arguments,
                 grid_size):
        super().__init__(context, module, function, arguments, grid_size)

        self.arguments = arguments

        self.kernel = module.get_function(function)
        self.kernel.prepare(arguments)

        self.cfl_data = gpuarray.GPUArray(grid_size, dtype=np.float32)

    def prepared_call(self, grid_size, block_size, stream, args: list):
        # if len(args) != len(self.arguments):
        #     raise ValueError("The parameters do not match the defined arguments.")

        self.kernel.prepared_async_call(grid_size, block_size, stream, *args)

    def array_fill(self, stream: driver.Stream, data):
        self.cfl_data.fill(data, stream=stream)

    def array_min(self, stream: driver.Stream) -> np.float32 | np.float64:
        return gpuarray.min(self.cfl_data, stream=stream).get()
