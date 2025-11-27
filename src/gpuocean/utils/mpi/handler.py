from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from gpuocean.utils.gpu import Array2D

if TYPE_CHECKING:
    from gpuocean.utils.gpu import GPUStream


class MPIHandler:
    def __init__(self, gpu_stream: GPUStream, nx: int, ny: int):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        init_east_west_data = np.zeros(shape=(ny, 2), dtype=np.float32)
        init_north_south_data = np.zeros(shape=(2, nx), dtype=np.float32)
        init_east_west = (gpu_stream, 2, ny, 0, 0, init_east_west_data)
        init_north_south = (gpu_stream, nx, 2, 0, 0, init_north_south_data)

        self.east = Array2D(*init_east_west)
        self.west = Array2D(*init_east_west)
        self.north = Array2D(*init_north_south)
        self.south = Array2D(*init_north_south)
