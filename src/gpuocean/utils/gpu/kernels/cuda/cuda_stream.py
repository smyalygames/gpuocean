import pycuda.driver as cuda

from ..stream import Stream


class CudaStream(Stream):
    """
    An object to handle CUDA Streams
    """

    def __init__(self):
        self._stream = cuda.Stream()

    def synchronize(self):
        """
        Synchronize the CUDA Stream
        """
        self._stream.synchronize()

    def destroy(self):
        """
        Destroy the CUDA Stream.
        """
        pass
