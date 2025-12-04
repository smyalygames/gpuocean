from typing import Literal
from dataclasses import dataclass
import math


@dataclass
class Coordinate:
    """
    Represents coordinate on a decomposed domain grid.
    Indexing starts at 0 for coordinates.
    """
    x: int
    y: int


class Grid:
    """
    Creates a grid that is a decomposition of the domain.
    """

    def __init__(self, nx: int, ny: int, total_nodes: int, rank: int):
        """
        Creates a grid for the given domain.
        :param nx: Size of the domain in the x-axis.
        :param ny: Size of the domain in the y-axis.
        :param total_nodes: Total number of compute nodes.
        :param rank: Rank of this current process.
        """

        self.domain_nx = nx
        self.domain_ny = ny
        self.total_nodes = total_nodes
        self.rank = rank

        # Subdomain grid size
        self.x, self.y = self._decompose_domain()

        # Current coordinates of this grid.
        self.location = self._calculate_coordinate()

        # Calculate the size of the local subdomain
        self.nx, self.ny = self._calculate_subdomain_size()

        # Get the coordinates of all the neighbors
        self.north = self.get_neighbor("north")
        self.east = self.get_neighbor("east")
        self.south = self.get_neighbor("south")
        self.west = self.get_neighbor("west")

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
        best_perimeter: int = int('inf')

        for pair in factors:
            perimeter = (self.domain_nx * (pair[0] - 1)) + (self.domain_ny * (pair[1] - 1))
            if perimeter < best_perimeter:
                best = pair
                best_perimeter = perimeter

        return best

    def _calculate_coordinate(self) -> Coordinate:
        """
        Calculate the coordinate for the rank of this process.
        :returns: This rank's coordinates, with indexing starting at 0.
        """
        y = self.rank // self.x
        x = self.rank % self.x

        # Checks that the coordinates are sane.
        if y > self.y - 1:
            raise RuntimeError("Processed y-coordinate in grid is out of bounds.")
        if x > self.x - 1:
            raise RuntimeError("Processed x-coordinate in grid is out of bounds.")

        coordinates = Coordinate(x, y)

        return coordinates

    def _calculate_subdomain_size(self) -> tuple[int, int]:
        """
        Calculates the size of the subdomain,
        taking into account divisions with remainders from the original domain.
        :returns: Size of the subdomain in the x- and y-axis respectively.
        """
        x_remainder = (self.domain_nx - self.location.x) % self.x
        y_remainder = (self.domain_ny - self.location.y) % self.y

        # Calculate the size of the subdomain
        nx = self.domain_nx / self.x
        ny = self.domain_ny / self.y

        # Account for decimals
        if x_remainder == 0:
            nx = math.floor(nx)
        else:
            nx = math.ceil(nx)

        if y_remainder == 0:
            ny = math.floor(ny)
        else:
            ny = math.ceil(ny)

        return nx, ny

    def get_neighbor(self, direction: Literal["north", "east", "south", "west"]) -> Coordinate | None:
        """
        Gets the coordinate of the neighboring process.
        This function would be useful for data exchanges.
        :param direction: Direction for the neighboring cell.
        :returns: Coordinate of the next cell over. None if there does not exist a neighbor in that direction.
        """
        match str(direction):
            case "north":
                new_y = self.location.y + 1
                # Check if the new y location goes out of bounds of the grid
                if new_y >= self.y:
                    return None

                return Coordinate(self.location.x, new_y)
            case "south":
                # Check if the current rank is in the southernly most point
                if self.location.y == 0:
                    return None

                new_y = self.location.y - 1
                return Coordinate(self.location.x, new_y)
            case "east":
                new_x = self.location.x + 1
                # Check if the new x location goes out of bounds of the grid
                if new_x >= self.x:
                    return None

                return Coordinate(new_x, self.location.y)
            case "west":
                # Checks if the current location is in the westerly most position already
                if self.location.x == 0:
                    return None

                new_x = self.location.x - 1
                return Coordinate(new_x, self.location.y)
            case _:
                raise ValueError("Did not correctly specify the direction of the neighbouring coordinate.")
