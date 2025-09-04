from hip import hip
from hip.hip import ihipStream_t, ihipEvent_t

from ..event import BaseEvent
from ...hip_utils import hip_check


class HIPEvent(BaseEvent):
    """
    A GPU Event handler.
    """

    def __init__(self):
        """
        Creates a GPU Event.
        """
        super().__init__()
        self.event = hip_check(hip.hipEventCreate())

    def __del__(self):
        hip_check(hip.hipEventDestroy(self.event))

    def record(self, stream: ihipStream_t | object):
        """
        Insert a recording point into the ``stream``.

        Args:
            stream: The stream to insert the recording point into.
        """
        hip_check(hip.hipEventRecord(self.event, stream))

    def synchronize(self):
        """
        Wait for the event to complete.
        """
        hip_check(hip.hipEventSynchronize(self.event))

    def time_since(self, start: ihipEvent_t | object):
        """
        Return the elapsed time from the ``start`` event and this class.

        Args:
            start: The Event to measure time from. Can also use the HIPEvent class instead of obj.event.

        Returns:
            Time since the ``start`` event and the end time of this class.
        """
        if isinstance(start, HIPEvent):
            start = start.event

        return hip_check(hip.hipEventElapsedTime(start, self.event))
