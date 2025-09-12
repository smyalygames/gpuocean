import logging
import time


class Timer(object):
    """
    Class which keeps track of time spent for a section of code
    """

    def __init__(self, tag: str, log_level=logging.DEBUG):
        self.tag = tag
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # milliseconds
        self.logger.log(self.log_level, f"{self.tag}: {self.msecs} ms")
