from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import stream_t


class Stream:
    """
    A base class to handle streams for each GPU platform.
    """

    def __del__(self):
        self.destroy()

    @property
    def pointer(self) -> stream_t:
        """
        Gets the object/pointer of the stream at the library level.
        Returns:
            HIP stream pointer or pycuda Stream.
        """
        return self._stream

    def synchronize(self):
        """
        Synchronize the GPU Stream
        """
        raise NotImplementedError("Needs to be implemented in a separate subclass.")

    def destroy(self):
        """
        Destroy the GPU Stream.
        """
        raise NotImplementedError("Needs to be implemented in a separate subclass.")
