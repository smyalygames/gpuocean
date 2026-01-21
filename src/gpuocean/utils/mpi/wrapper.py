from collections.abc import Iterable
from typing import TYPE_CHECKING
from enum import IntEnum, auto
import logging

import numpy as np

from gpuocean.SWEsimulators.Simulator import Simulator
from gpuocean.utils.gpu import Array2D

from .grid import Grid

if TYPE_CHECKING:
    from mpi4py.MPI import Request


class MPIWrapper:
    """
    An MPI wrapper for the SWE simulator schemes.
    """

    def __init__(self, sim: Simulator):
        self.logger = logging.getLogger(__name__)
        self.comm = sim.comm
        total_nodes = self.comm.size
        rank = self.comm.rank

        self.logger.info(f"Rank: {rank}, Total Ranks: {total_nodes}.")

        # FIXME change this so that it does not use the pre-existing simulator,
        #   and initializes a simulator.
        self.sim = sim
        self.grid = Grid(sim.nx, sim.ny, total_nodes, rank)

        self.step_number = 0

        # Create grid for domain decomposition

        self.exists = {
            Direction.NORTH: self.grid.north is not None,
            Direction.EAST: self.grid.east is not None,
            Direction.SOUTH: self.grid.south is not None,
            Direction.WEST: self.grid.west is not None
        }

        self._exchange()

    def __getattr__(self, item):
        return getattr(self.sim, item)

    def _exchange(self):
        """
        Completes the MPI exchange for all the arrays.
        """
        for array in self._get_domains():
            # Prepare the data
            comm_send: list[Request] = []
            comm_recv: list[Request] = []

            # Shift by 2 bits
            tag_pad = 4 * self.step_number

            if self.exists[Direction.NORTH]:
                send_north = array.download_boundary(self.sim.gpu_stream, "north")
                recv_north = np.empty_like(send_north)

                exchange_rank = self.grid.north.rank
                send_tag = tag_pad + Direction.NORTH
                recv_tag = tag_pad + Direction.SOUTH
                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (north), "
                                  f"shape: {send_north.shape}, send tag: {send_tag}, receive tag: {recv_tag}")

                comm_send.append(self.comm.Isend(send_north, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(recv_north, source=exchange_rank, tag=recv_tag))
            if self.exists[Direction.EAST]:
                send_east = array.download_boundary(self.sim.gpu_stream, "east")
                recv_east = np.empty_like(send_east)

                exchange_rank = self.grid.east.rank
                send_tag = tag_pad + Direction.EAST
                recv_tag = tag_pad + Direction.WEST
                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (east), "
                                  f"shape: {send_east.shape}, send tag: {send_tag}, receive tag: {recv_tag}")

                comm_send.append(self.comm.Isend(send_east, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(recv_east, source=exchange_rank, tag=recv_tag))
            if self.exists[Direction.SOUTH]:
                send_south = array.download_boundary(self.sim.gpu_stream, "south")
                recv_south = np.empty_like(send_south)

                exchange_rank = self.grid.south.rank
                send_tag = tag_pad + Direction.SOUTH
                recv_tag = tag_pad + Direction.NORTH
                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (south), "
                                  f"shape: {send_south.shape}, send tag: {send_tag}, receive tag: {recv_tag}")

                comm_send.append(self.comm.Isend(send_south, dest=exchange_rank, tag=tag_pad + Direction.SOUTH))
                comm_recv.append(self.comm.Irecv(recv_south, source=exchange_rank, tag=recv_tag))
            if self.exists[Direction.WEST]:
                send_west = array.download_boundary(self.sim.gpu_stream, "west")
                recv_west = np.empty_like(send_west)

                exchange_rank = self.grid.west.rank
                send_tag = tag_pad + Direction.WEST
                recv_tag = tag_pad + Direction.EAST
                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (west), "
                                  f"shape: {send_west.shape}, send tag: {send_tag} receive tag: {recv_tag}")
                
                comm_send.append(self.comm.Isend(send_west, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(recv_west, source=exchange_rank, tag=recv_tag))

            # Do MPI exchange

            # Wait for transfer to complete
            for comm in comm_recv:
                comm.wait()

            self.logger.debug(f"Rank {self.comm.rank} received all data for transfer {self.step_number}")

            # Upload to the array
            if self.exists[Direction.NORTH]:
                array.upload_boundary(self.sim.gpu_stream, recv_north, "north")
            if self.exists[Direction.EAST]:
                array.upload_boundary(self.sim.gpu_stream, recv_east, "east")
            if self.exists[Direction.SOUTH]:
                array.upload_boundary(self.sim.gpu_stream, recv_south, "south")
            if self.exists[Direction.WEST]:
                array.upload_boundary(self.sim.gpu_stream, recv_west, "west")

            # Wait for transfers to complete
            for comm in comm_send:
                comm.wait()

            self.logger.debug(f"Rank {self.comm.rank} sent all data for transfer {self.step_number}")
            self.step_number += 1

    def _get_domains(self) -> Iterable[Array2D]:
        """
        Gets the domains to exchange data through MPI.
        """
        return self.sim.gpu_data.arrays

    def step(self, t_end=0.0):
        self.sim.step(t_end)
        self.step_number += 1
        self._exchange()

    def cleanUp(self):
        self.sim.cleanUp()


class Direction(IntEnum):
    """
    Gives a direction for the MPI grid an assigned value.
    Used for tagging.
    """
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
