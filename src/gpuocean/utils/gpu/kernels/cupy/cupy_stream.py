import cupy as cp

from ..hip.hip_stream import HIPStream


class CuPyStream(HIPStream):
    """
    An object to handle CuPy Streams
    """

    def __init__(self):
        super().__init__()
        self._cupy_stream: cp.cuda.ExternalStream = cp.cuda.ExternalStream(int(self._stream))
        self._cupy_stream.use()

    def destroy(self):
        """
        Destroy the CuPy Stream.
        """
        cp.cuda.Stream.null.use()
        super().destroy()
