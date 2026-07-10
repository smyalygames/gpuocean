from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Iterable
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import cupy as cp
    from hip._util import types
    from .. import GPUStream, Array2D


class BaseGPUHandler(ABC):
    """
    A handler to make GPU calls.
    """

    mpi_exchange_func: Callable[[list[types.Pointer | cp.cuda.MemoryPointer]], None] = None

    @abstractmethod
    def __init__(self, module, function: str, arguments: str):
        """
        Create a new GPU handler.

        Args:
            module: The module created from KernelContext for a function.
            function: Name of the function to use in the kernel.
            arguments: A string of the argument types to parse to the kernel.
        """
        self.exchange_arrays: list[Array2D] | None = None

    @abstractmethod
    def async_call(self, grid_size, block_size: tuple[int, int, int], stream: GPUStream, args: list,
                   exchange=True, exchange_exclude: Iterable = None):
        """
        Makes an asynchronous call to the kernel on the GPU with the function that was used to initialize this object.

        Args:
            grid_size: The size of the grid to do the computation of.
            block_size: The block size, as a tuple.
            stream: The GPU data stream.
            args: Parameters to be passed into the GPU kernel.
            exchange: Allow exchanging the arrays if a communication protocol is present
            exchange_exclude: Exclude certain arrays from being exchanged
        """

    @abstractmethod
    def call(self, grid_size, block_size: tuple[int, int, int], args: list,
             exchange=True, exchange_exclude: Iterable = None):
        """
        Makes a call to the kernel on the GPU with the function that was used to initialize this object.

        Args:
            grid_size: The size of the grid to do the computation of.
            block_size: The block size, as a tuple.
            args: Parameters to be passed into the GPU kernel.
            exchange: Allow exchanging the arrays if a communication protocol is present
            exchange_exclude: Exclude certain arrays from being exchanged
        """

    def exchange(self, pointers: list[types.Pointer | cp.cuda.MemoryPointer]) -> None:
        """
        Completes an exchange between nodes. Should be used before running the kernel.
        """

        if self.mpi_exchange_func is None:
            return

        self.mpi_exchange_func(pointers)
