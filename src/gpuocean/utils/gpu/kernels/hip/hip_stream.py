from hip import hip

from ..stream import Stream
from ...hip_utils import hip_check


class HIPStream(Stream):
    """
    An object to handle HIP Streams
    """

    def __init__(self):
        self._stream: hip.ihipStream_t = hip_check(hip.hipStreamCreate())

    def synchronize(self):
        """
        Synchronize the HIP Stream
        """
        hip_check(hip.hipStreamSynchronize(self._stream))

    def destroy(self):
        """
        Destroy the HIP Stream.
        """
        hip_check(hip.hipStreamDestroy(self._stream))
