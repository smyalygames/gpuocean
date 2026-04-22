import cupy as cp

from ..hip.hip_stream import HIPStream


class CuPyStream(HIPStream):
    """
    An object to handle CuPy Streams
    """

    def __init__(self, default_stream=True):
        super().__init__(False)
        self._cupy_stream: cp.cuda.ExternalStream = cp.cuda.ExternalStream(int(self._stream))

        if default_stream:
            self.make_default()

    def destroy(self):
        """
        Destroy the CuPy Stream.
        """
        cp.cuda.Stream.null.use()
        super().destroy()

    def make_default(self):
        self._cupy_stream.use()
