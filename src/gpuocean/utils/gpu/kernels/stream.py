from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .. import stream_t


class Stream(ABC):
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

    @abstractmethod
    def synchronize(self) -> None:
        """
        Synchronize the GPU Stream
        """

    @abstractmethod
    def destroy(self) -> None:
        """
        Destroy the GPU Stream.
        """
