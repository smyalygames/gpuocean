from abc import ABC, abstractmethod

class BaseEvent(ABC):
    """
    A GPU Event handler.
    """

    @abstractmethod
    def __init__(self):
        """
        Creates a GPU Event.
        """

    @abstractmethod
    def record(self, stream):
        """
        Insert a recording point into the ``stream``.

        Args:
            stream: The stream to insert the recording point into.
        """

    @abstractmethod
    def synchronize(self):
        """
        Wait for the event to complete.
        """

    @abstractmethod
    def time_since(self, start) -> float:
        """
        Return the elapsed time from the ``start`` event and this class.

        Args:
            start: The Event to measure time from.

        Returns:
            Time since the ``start`` event and the end time of this class.
        """
