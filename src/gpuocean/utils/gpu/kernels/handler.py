from .. import KernelContext


class BaseGPUHandler(object):
    """
    A handler to make GPU calls.
    """

    def __init__(self, context: KernelContext, module, function: str, arguments: str,
                 grid_size: tuple[int, int]):
        """
        Create a new GPU handler.

        Args:
            context: The KernelContext that is used to make the calls to the kernel.
            module: The module created from KernelContext for a function.
            function: Name of the function to use in the kernel.
            arguments: A string of the argument types to parse to the kernel.
            grid_size: The size of the array for the data of the simulation.
        """

    def prepared_call(self, grid_size, block_size: tuple[int, int, int], stream, args: list):
        """
        Makes a call to the kernel on the GPU with the function that was used to initialize this object.

        Args:
            grid_size: The size of the grid to do the computation of.
            block_size: The block size, as a tuple.
            stream: The GPU data stream.
            args: Parameters to be passed into the GPU kernel.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def array_fill(self, data: float, stream):
        """
        Fills the entire array with the same data that was parsed as the parameter.

        Args:
            data: The data to fill the array with.
            stream: The GPU data stream.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def array_min(self, stream) -> float:
        """
        Gets the minimum value in the array stored in the handler.

        Args:
            stream: The GPU data stream.

        Returns:
            The minimum value in the stored array as a float.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")
