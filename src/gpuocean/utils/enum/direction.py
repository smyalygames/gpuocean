from __future__ import annotations
from enum import IntEnum


class Direction(IntEnum):
    """
    Gives a direction for the MPI grid an assigned value.
    Used for tagging.
    """
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def opposite(self) -> Direction:
        """
        Gets the opposite direction of the current enum.
        :returns: Opposite direction.
        """
        if self.value < 2:
            value = self.value + 2
        else:
            value = self.value - 2

        return self.__class__(value)
