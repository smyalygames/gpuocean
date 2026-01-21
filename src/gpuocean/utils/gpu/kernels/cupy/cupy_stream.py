import cupy as cp

from ..stream import Stream


class CuPyStream(Stream):
    """
    An object to handle CuPy Streams
    """

    def __init__(self):
        self.__stream: cp.cuda.Stream = cp.cuda.Stream()
        self.__stream.use()

    @property
    def _stream(self):
        return self.__stream.ptr

    def synchronize(self):
        """
        Synchronize the CuPy Stream
        """
        self.__stream.synchronize()

    def destroy(self):
        """
        Destroy the CuPy Stream.
        """
        cp.cuda.Stream.null.use()
