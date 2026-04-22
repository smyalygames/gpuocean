from typing import Literal, ClassVar
from dataclasses import dataclass, field
import math

from gpuocean.utils.enum import Direction



@dataclass
class Coordinate:
    """
    Represents coordinate on a decomposed domain grid.
    Indexing starts at 0 for coordinates.
    """
    x: int
    y: int
    nodes_x: ClassVar[int]
    nodes_y: ClassVar[int]
    rank: int = field(init=False)

    def __post_init__(self):
        self.rank = (self.y * self.nodes_x) + self.x


class Grid:
    """
    Creates a grid that is a decomposition of the domain.
    """

    global_nx: int
    global_ny: int
    # Decomposition of all nodes in dimensions
    nodes_x: int
    nodes_y: int
    # Position of local domain on global domain
    x_pos: int
    y_pos: int
    # Global domain coordinates
    x0: int
    x1: int
    y0: int
    y1: int

    def __init__(self, nx: int, ny: int, total_nodes: int, rank: int):
        """
        Creates a grid for the given domain.
        :param nx: Size of the domain in the x-axis.
        :param ny: Size of the domain in the y-axis.
        :param total_nodes: Total number of compute nodes.
        :param rank: Rank of this current process.
        """

        Grid.global_nx = nx
        Grid.global_ny = ny
        self.total_nodes = total_nodes
        self.rank = rank

        # Subdomain grid size
        Grid.nodes_x, Grid.nodes_y = self._decompose_domain()

        Coordinate.nodes_x = self.nodes_x
        Coordinate.nodes_y = self.nodes_y

        # Current coordinates of this grid.
        self.location = self._calculate_coordinate()
        Grid.x_pos = self.location.x
        Grid.y_pos = self.location.y

        # Calculate the size of the local subdomain
        self.local_nx, self.local_ny = self._calculate_subdomain_size(self.location.x, self.location.y)

        # Provide the coordinates of the domain relative to the global domain
        # TODO replace nx with global nx, or have something to determine the type of domain decomposition
        global_coordinates = self._calculate_global_coordinates()
        Grid.x0, Grid.x1 = global_coordinates[0]
        Grid.y0, Grid.y1 = global_coordinates[1]

        # Get the coordinates of all the neighbors
        self.north = self.get_neighbor(Direction.NORTH)
        self.east = self.get_neighbor(Direction.EAST)
        self.south = self.get_neighbor(Direction.SOUTH)
        self.west = self.get_neighbor(Direction.WEST)

    def _decompose_domain(self) -> tuple[int, int]:
        """
        Decomposes the domain to a somewhat efficient order,
        based on the total number of nodes there are.
        :returns: How many subdomains in the x- and y-axis respectively.
        """
        # Check that the total number of nodes is positive
        if self.total_nodes < 1:
            raise ValueError("There cannot be be zero or a negative number of total nodes.")

        # Check that there are more than one node
        if self.total_nodes == 1:
            return (1, 1)

        # Get all the pairs of factors for total number of nodes
        factors: list[tuple[int, int]] = []
        for n in range(1, self.total_nodes + 1):
            if self.total_nodes % n == 0:
                factors.append((n, self.total_nodes // n))

        # Figure out which factor has the smallest exchange perimeter
        best: tuple[int, int] = (0, 0)
        best_perimeter: int | float = math.inf

        for pair in factors:
            perimeter = (self.global_nx * (pair[0] - 1)) + (self.global_ny * (pair[1] - 1))
            if perimeter < best_perimeter:
                best = pair
                best_perimeter = perimeter

        return best

    def _calculate_coordinate(self) -> Coordinate:
        """
        Calculate the coordinate for the rank of this process.
        :returns: This rank's coordinates, with indexing starting at 0.
        """
        y = self.rank // self.nodes_x
        x = self.rank % self.nodes_x

        # Checks that the coordinates are sane.
        if y > self.nodes_y - 1:
            raise RuntimeError("Processed y-coordinate in grid is out of bounds.")
        if x > self.nodes_x - 1:
            raise RuntimeError("Processed x-coordinate in grid is out of bounds.")

        coordinates = Coordinate(x, y)

        return coordinates

    def _calculate_global_coordinates(self):
        """
        Calculates the edge coordinates of this subdomain in relation to the global domain.
        """
        start_x = 0
        start_y = 0
        # Calculate x-axis start position
        for x in range(self.location.x):
            start_x += self._calculate_subdomain_size(x, self.location.y)[0]
        # Calculate y-axis start position
        for y in range(self.location.y):
            start_y += self._calculate_subdomain_size(self.location.x, y)[1]

        end_x = start_x + self.local_nx
        end_y = start_y + self.local_ny

        return (start_x, end_x), (start_y, end_y)

    def _calculate_subdomain_size(self, x_pos: int, y_pos: int) -> tuple[int, int]:
        """
        Calculates the size of the subdomain,
        taking into account divisions with remainders from the original domain.
        :param x_pos: Position of the node in relation to the other nodes globally in the x-axis.
        :param y_pos: Position of the node in relation to the other nodes globally in the y-axis.
        :returns: Size of the subdomain in the x- and y-axis respectively.
        """
        x_remainder =  x_pos % self.nodes_x
        y_remainder = y_pos % self.nodes_y

        # Calculate the size of the subdomain
        nx = self.global_nx / self.nodes_x
        ny = self.global_ny / self.nodes_y

        # Account for decimals
        if x_remainder == 0:
            nx = math.ceil(nx)
        else:
            nx = math.floor(nx)

        if y_remainder == 0:
            ny = math.ceil(ny)
        else:
            ny = math.floor(ny)

        return nx, ny

    def get_neighbor(self, direction: Direction) -> Coordinate | None:
        """
        Gets the coordinate of the neighboring process.
        This function would be useful for data exchanges.
        :param direction: Direction for the neighboring cell.
        :returns: Coordinate of the next cell over. None if there does not exist a neighbor in that direction.
        """
        match direction:
            case Direction.NORTH:
                new_y = self.location.y + 1
                # Check if the new y location goes out of bounds of the grid
                if new_y >= self.nodes_y:
                    return None

                return Coordinate(self.location.x, new_y)
            case Direction.SOUTH:
                # Check if the current rank is in the southernly most point
                if self.location.y == 0:
                    return None

                new_y = self.location.y - 1
                return Coordinate(self.location.x, new_y)
            case Direction.EAST:
                new_x = self.location.x + 1
                # Check if the new x location goes out of bounds of the grid
                if new_x >= self.nodes_x:
                    return None

                return Coordinate(new_x, self.location.y)
            case Direction.WEST:
                # Checks if the current location is in the westerly most position already
                if self.location.x == 0:
                    return None

                new_x = self.location.x - 1
                return Coordinate(new_x, self.location.y)
            case _:
                raise ValueError("Did not correctly specify the direction of the neighbouring coordinate.")
