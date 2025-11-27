from math import sqrt


class Distribute:
    """
    Distributes work and does domain decomposition.
    """
    def __init__(self, nx: int, ny: int, rank: int, nodes: int):
        """
        Handles what workload to give each process in MPI.
        :param nx: Full grid size in x-axis.
        :param ny: Full grid size in y-axis.
        :param rank: Rank of the current node.
        :param nodes: Total number of MPI nodes.
        """
        self.total_nx = nx
        self.total_ny = ny
        self.rank = rank
        self.nodes = nodes

        self.nx, self.ny = self._decompose()

    def _decompose(self) -> tuple[int, int]:
        """
        Handles domain decomposition.
        """
        if self.nodes == 1:
            return self.total_nx, self.total_ny

        if self.nodes != 4:
            raise NotImplementedError("Decomposition for nodes other than 4 or 1 is not implemented yet.")

        # FIXME this most likely is total junk code, should be written up with much better logic...
        scale = self.nodes / 2 # 4
        self.total_x_nodes = scale # 2

        return round(self.total_nx / scale), round(self.total_ny / scale)

    @property
    def position(self) -> tuple[int, int]:
        return int(self.rank % self.total_x_nodes), int(self.rank // self.total_x_nodes)
