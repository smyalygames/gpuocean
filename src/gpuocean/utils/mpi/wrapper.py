from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal
from enum import IntEnum
from dataclasses import dataclass
import logging

import numpy as np
import cupy as cp

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

        self.exchange_arrays: dict[Array2D, ArrayExchange] = {}
        self._prepare_exchanges()

        self._exchange()

    def __getattr__(self, item):
        return getattr(self.sim, item)

    def _prepare_exchanges(self):
        """
        Creates buffers for exchanging partial data between arrays.
        """
        arrays = self._get_domains()
        for array in arrays:
            exchanges = ArrayExchange()

            if self.exists[Direction.NORTH]:
                buffer = array.download_boundary(self.sim.gpu_stream, "north")
                exchanges.north = Exchange(cp.zeros_like(buffer), cp.zeros_like(buffer))
            if self.exists[Direction.EAST]:
                buffer = array.download_boundary(self.sim.gpu_stream, "east")
                exchanges.east = Exchange(cp.zeros_like(buffer), cp.zeros_like(buffer))
            if self.exists[Direction.SOUTH]:
                buffer = array.download_boundary(self.sim.gpu_stream, "south")
                exchanges.south = Exchange(cp.zeros_like(buffer), cp.zeros_like(buffer))
            if self.exists[Direction.WEST]:
                buffer = array.download_boundary(self.sim.gpu_stream, "west")
                exchanges.west = Exchange(cp.zeros_like(buffer), cp.zeros_like(buffer))

            self.exchange_arrays[array] = exchanges

        # Remove unused array
        buffer = None

    def _exchange(self):
        """
        Completes the MPI exchange for all the arrays.
        """
        self.sim.gpu_stream.synchronize()

        for array in self._get_domains():
            # Prepare the data
            comm_send: list[Request] = []
            comm_recv: list[Request] = []

            # Shift by 2 bits
            tag_pad = 4 * self.step_number

            # Copy boundary data to buffer
            self.exchange_arrays[array].prepare_send(array, self.sim.gpu_stream)

            if self.exists[Direction.NORTH]:
                exchange_rank = self.grid.north.rank
                send_tag = tag_pad + Direction.NORTH
                recv_tag = tag_pad + Direction.SOUTH
                exchange = self.exchange_arrays[array].north

                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (north), "
                                  f"shape: {exchange.send.shape}, send tag: {send_tag}, receive tag: {recv_tag}")

                comm_send.append(self.comm.Isend(exchange.send, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(exchange.recv, source=exchange_rank, tag=recv_tag))
            if self.exists[Direction.EAST]:
                exchange_rank = self.grid.east.rank
                send_tag = tag_pad + Direction.EAST
                recv_tag = tag_pad + Direction.WEST
                exchange = self.exchange_arrays[array].east

                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (east), "
                                  f"shape: {exchange.send.shape}, send tag: {send_tag}, receive tag: {recv_tag}")

                comm_send.append(self.comm.Isend(exchange.send, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(exchange.recv, source=exchange_rank, tag=recv_tag))
            if self.exists[Direction.SOUTH]:
                exchange_rank = self.grid.south.rank
                send_tag = tag_pad + Direction.SOUTH
                recv_tag = tag_pad + Direction.NORTH
                exchange = self.exchange_arrays[array].south

                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (south), "
                                  f"shape: {exchange.send.shape}, send tag: {send_tag}, receive tag: {recv_tag}")

                comm_send.append(self.comm.Isend(exchange.send, dest=exchange_rank, tag=tag_pad + Direction.SOUTH))
                comm_recv.append(self.comm.Irecv(exchange.recv, source=exchange_rank, tag=recv_tag))
            if self.exists[Direction.WEST]:
                exchange_rank = self.grid.west.rank
                send_tag = tag_pad + Direction.WEST
                recv_tag = tag_pad + Direction.EAST
                exchange = self.exchange_arrays[array].west

                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} (west), "
                                  f"shape: {exchange.send.shape}, send tag: {send_tag} receive tag: {recv_tag}")

                comm_send.append(self.comm.Isend(exchange.send, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(exchange.recv, source=exchange_rank, tag=recv_tag))

            # Do MPI exchange

            # Wait for transfer to complete
            for comm in comm_recv:
                comm.wait()

            self.logger.debug(f"Rank {self.comm.rank} received all data for transfer {self.step_number}")

            self.exchange_arrays[array].upload_received(array, self.sim.gpu_stream)

            # Wait for transfers to complete
            for comm in comm_send:
                comm.wait()

            self.logger.debug(f"Rank {self.comm.rank} sent all data for transfer {self.step_number}")
            self.step_number += 1

    def _get_domains(self) -> list[Array2D]:
        """
        Gets the domains to exchange data through MPI.
        """
        return self.sim.arrays

    def step(self, t_end=0.0):
        self.sim.step(t_end)
        self.step_number += 1
        self._exchange()

    def cleanUp(self):
        self.sim.cleanUp()


@dataclass
class Exchange:
    send: cp.ndarray
    recv: cp.ndarray


@dataclass
class ArrayExchange:
    north: Exchange | None = None
    east: Exchange | None = None
    south: Exchange | None = None
    west: Exchange | None = None

    def prepare_send(self, array: Array2D, gpu_stream):
        """
        Updates the send arrays in each direction if there is one defined.
        """
        if self.north is not None:
            self.north.send = array.download_boundary(gpu_stream, "north")
        if self.east is not None:
            self.east.send = array.download_boundary(gpu_stream, "east")
        if self.south is not None:
            self.south.send = array.download_boundary(gpu_stream, "south")
        if self.west is not None:
            self.west.send = array.download_boundary(gpu_stream, "west")

        gpu_stream.synchronize()

    def upload_received(self, array: Array2D, gpu_stream):
        """
        Updates the array with all the received boundary data.
        """
        if self.north is not None:
            array.upload_boundary(gpu_stream, self.north.recv, "north")
        if self.east is not None:
            array.upload_boundary(gpu_stream, self.east.recv, "east")
        if self.south is not None:
            array.upload_boundary(gpu_stream, self.south.recv, "south")
        if self.west is not None:
            array.upload_boundary(gpu_stream, self.west.recv, "west")


class Direction(IntEnum):
    """
    Gives a direction for the MPI grid an assigned value.
    Used for tagging.
    """
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
