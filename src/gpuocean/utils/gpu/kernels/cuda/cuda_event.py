from typing import TYPE_CHECKING

import pycuda.driver as cuda

from ..event import BaseEvent


class CudaEvent(BaseEvent):
    """
    A GPU Event handler.
    """

    def __init__(self):
        """
        Creates a GPU Event.
        """
        super().__init__()
        self.event = cuda.Event()

    def record(self, stream: cuda.Stream):
        """
        Insert a recording point into the ``stream``.

        Args:
            stream: The stream to insert the recording point into.
        """
        self.event.record(stream)

    def synchronize(self):
        """
        Wait for the event to complete.
        """
        self.event.synchronize()

    def time_since(self, start: cuda.Event) -> float:
        """
        Return the elapsed time from the ``start`` event and this class.

        Args:
            start: The Event to measure time from.

        Returns:
            Time since the ``start`` event and the end time of this class.
        """
        return self.event.time_since(start)
