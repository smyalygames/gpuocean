class BaseEvent(object):
    """
    A GPU Event handler.
    """

    def __init__(self):
        """
        Creates a GPU Event.
        """

    def record(self, stream):
        """
        Insert a recording point into the ``stream``.

        Args:
            stream: The stream to insert the recording point into.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def synchronize(self):
        """
        Wait for the event to complete.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def time_since(self, start) -> float:
        """
        Return the elapsed time from the ``start`` event and this class.

        Args:
            start: The Event to measure time from.

        Returns:
            Time since the ``start`` event and the end time of this class.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")
