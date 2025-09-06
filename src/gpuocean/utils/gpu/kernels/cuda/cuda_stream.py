import pycuda.driver as cuda

from ..stream import Stream


class HIPStream(Stream):
    """
    An object to handle CUDA Streams
    """

    def __init__(self):
        self.__stream= cuda.Stream()

    def synchronize(self):
        """
        Synchronize the CUDA Stream
        """
        self.__stream.synchronize()

    def destroy(self):
        """
        Destroy the CUDA Stream.
        """
        pass
