from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import GPUStream


class BaseGPUHandler(object):
    """
    A handler to make GPU calls.
    """

    def __init__(self, module, function: str, arguments: str):
        """
        Create a new GPU handler.

        Args:
            module: The module created from KernelContext for a function.
            function: Name of the function to use in the kernel.
            arguments: A string of the argument types to parse to the kernel.
        """

    def async_call(self, grid_size, block_size: tuple[int, int, int], stream: GPUStream, args: list):
        """
        Makes an asynchronous call to the kernel on the GPU with the function that was used to initialize this object.

        Args:
            grid_size: The size of the grid to do the computation of.
            block_size: The block size, as a tuple.
            stream: The GPU data stream.
            args: Parameters to be passed into the GPU kernel.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def call(self, grid_size, block_size: tuple[int, int, int], args: list):
        """
        Makes a call to the kernel on the GPU with the function that was used to initialize this object.

        Args:
            grid_size: The size of the grid to do the computation of.
            block_size: The block size, as a tuple.
            args: Parameters to be passed into the GPU kernel.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")
