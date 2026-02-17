from dataclasses import dataclass, field

from .direction import DirectionsValue

@dataclass
class GhostCells(DirectionsValue):
    """
    Holds the number of ghost cells in each direction.
    Calculates the number of ghost cells vertically (`x`) and horizontally (`y`).
    """
    north: int
    east: int
    south: int
    west: int
    total_x: int = field(init=False)
    total_y: int = field(init=False)

    def __post_init__(self):
        self.total_x = self.east + self.west
        self.total_y = self.north + self.south

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        """
        Creates a tuple with the number of ghost cells in each direction.
        :returns: Tuple of directions: `(north, east, south, west)`.
        """
        return self.north, self.east, self.south, self.west

    @property
    def x(self) -> int:
        """
        Gets the number of ghost cells of one side in the horizontal direction.
        :returns: Number of ghost cells.
        :raises ValueError: When number of ghost cells horizontally are asymmetrical.
        """
        if self.east != self.west:
            raise ValueError("East and West ghost cells are asymmetrical, "
                             "cannot compute generic number of ghost cells. "
                             f"Number of ghost cells east: {self.north}, west: {self.south}.")
        return self.east

    @property
    def y(self) -> int:
        """
        Gets the number of ghost cells of one side in the vertical direction.
        :returns: Number of ghost cells.
        :raises ValueError: When number of ghost cells vertically are asymmetrical.
        """
        if self.north != self.south:
            raise ValueError("North and South ghost cells are asymmetrical, "
                             "cannot compute generic number of ghost cells. "
                             f"Number of ghost cells north: {self.north}, south: {self.south}.")
        return self.north
