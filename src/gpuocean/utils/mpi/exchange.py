from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import logging
from dataclasses import dataclass

from mpi4py import MPI
import numpy as np

from gpuocean.utils.enum import Direction
from gpuocean.utils.gpu import CuPyArray2D

if TYPE_CHECKING:
    from mpi4py.MPI import Request
    from gpuocean.utils.gpu import Array2D, Array3D, GPUStream
    from .grid import Grid


class MPIExchange:
    """
    Handles exchanging data through MPI.
    """

    def __init__(self, gpu_stream: GPUStream, grid: Grid, arrays: Iterable[Array2D], comm=MPI.COMM_WORLD):
        """
        Handler for exchanging data through CuPy using MPI.
        :param gpu_stream: GPU Stream for synchronizing arrays.
        :param grid: Object that handles domain decomposition.
        :param arrays: GPU arrays to handle exchanges for.
        :param comm: MPI communicator for handling exchanges.
        """
        self.logger = logging.getLogger(__name__)

        self.comm = comm
        self.gpu_stream = gpu_stream
        self.grid = grid

        self.step_number = 0

        self.exists: list[Direction] = []

        if self.grid.north is not None:
            self.exists.append(Direction.NORTH)
        if self.grid.east is not None:
            self.exists.append(Direction.EAST)
        if self.grid.south is not None:
            self.exists.append(Direction.SOUTH)
        if self.grid.west is not None:
            self.exists.append(Direction.WEST)

        self.exchange_arrays: dict[Array2D, ArrayExchange] = {}
        self._prepare_exchanges(arrays)
        self.total_arrays = len(self.exchange_arrays)

    def _prepare_exchanges(self, arrays: Iterable[Array2D]):
        """
        Creates buffers for exchanging partial data between arrays.
        """
        for index, array in enumerate(arrays):
            exchanges = ArrayExchange(index, array)

            if Direction.NORTH in self.exists:
                buffer = array.download_boundary(self.gpu_stream, "north")
                exchanges.north = Exchange(CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                       0, 0, buffer),
                                           CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                       0, 0, buffer))
            if Direction.EAST in self.exists:
                buffer = array.download_boundary(self.gpu_stream, "east")
                exchanges.east = Exchange(CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                      0, 0, buffer),
                                          CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                      0, 0, buffer))
            if Direction.SOUTH in self.exists:
                buffer = array.download_boundary(self.gpu_stream, "south")
                exchanges.south = Exchange(CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                       0, 0, buffer),
                                           CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                       0, 0, buffer))
            if Direction.WEST in self.exists:
                buffer = array.download_boundary(self.gpu_stream, "west")
                exchanges.west = Exchange(CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                      0, 0, buffer),
                                          CuPyArray2D(self.gpu_stream, buffer.shape[1], buffer.shape[0],
                                                      0, 0, buffer))

            self.exchange_arrays[array] = exchanges

        # Remove unused array
        buffer = None

    def exchange(self):
        """
        Completes the MPI exchange for all the arrays.
        """

        # Gather all the data to exchange
        for array, exchange in self.exchange_arrays.items():
            exchange.prepare_send(self.gpu_stream)

        self.gpu_stream.synchronize()

        comm_send: list[Request] = []
        comm_recv: list[Request] = []

        # Send MPI data.
        for array, exchange in self.exchange_arrays.items():

            for direction in self.exists:
                exchange_rank = self.grid.get_neighbor(direction).rank
                send_tag, recv_tag = exchange.get_tags(self.step_number, self.total_arrays, direction)
                array_exchange = self.exchange_arrays[array].get_direction(direction)

                self.logger.debug(f"Sending from {self.comm.rank} to {exchange_rank} ({direction.value}), "
                                  f"shape: {array_exchange.send.shape}, send tag: {send_tag}, receive tag: {recv_tag}.")
                comm_send.append(self.comm.Isend(array_exchange.send.data, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(array_exchange.recv.data, source=exchange_rank, tag=recv_tag))

        # Wait to receive all arrays
        for comm in comm_recv:
            comm.wait()

        self.logger.debug(f"Rank {self.comm.rank} received all data for transfer {self.step_number}")

        for exchange in self.exchange_arrays.values():
            exchange.upload_received(self.gpu_stream)

        # Wait for transfers to complete
        for comm in comm_send:
            comm.wait()

        self.logger.debug(f"Rank {self.comm.rank} sent all data for transfer {self.step_number}")
        self.step_number += 1


@dataclass
class Exchange:
    send: CuPyArray2D
    recv: CuPyArray2D


@dataclass
class ArrayExchange:
    index: int
    array: Array2D | Array3D
    north: Exchange | None = None
    east: Exchange | None = None
    south: Exchange | None = None
    west: Exchange | None = None

    def prepare_send(self, gpu_stream):
        """
        Updates the send arrays in each direction if there is one defined.
        """
        if self.north is not None:
            self.array.download_boundary(gpu_stream, "south", data=self.north.send)
        if self.east is not None:
             self.array.download_boundary(gpu_stream, "east", data=self.east.send)
        if self.south is not None:
             self.array.download_boundary(gpu_stream, "north", data=self.south.send)
        if self.west is not None:
            self.array.download_boundary(gpu_stream, "west", data=self.west.send)

        # gpu_stream.synchronize()

    def upload_received(self, gpu_stream):
        """
        Updates the array with all the received boundary data.
        """
        if self.north is not None:
            self.array.upload_boundary(gpu_stream, self.north.recv, "south")
        if self.east is not None:
            self.array.upload_boundary(gpu_stream, self.east.recv, "east")
        if self.south is not None:
            self.array.upload_boundary(gpu_stream, self.south.recv, "north")
        if self.west is not None:
            self.array.upload_boundary(gpu_stream, self.west.recv, "west")

    def get_direction(self, direction: Direction) -> Exchange | None:
        """
        Gets a direction's Exchange.
        :param direction: Direction of which variable to get from.
        """
        match direction:
            case Direction.NORTH:
                return self.north
            case Direction.EAST:
                return self.east
            case Direction.SOUTH:
                return self.south
            case Direction.WEST:
                return self.west
            case _:
                raise RuntimeError(f"{direction} is an invalid direction to get a variable from.")

    def get_tags(self, step: int, total_arrays: int, direction: Direction) -> tuple[int, int]:
        """
        Gives a unique tag for sending in the MPI exchange.
        :param step: How many full exchanges have already occurred, plus one.
            Should be `self.step_number` in `MPIExchange`.
        :param total_arrays: Total number of arrays that will be exchanged globally,
            Not to be confused with this specific array.
        :param direction: Direction of where the array lies.
        :returns: Tuple of send tag and recv tag.
        """
        direction_bits = max(Direction).bit_length()
        total_arrays_bits = total_arrays.bit_length()

        # Using bit shift to allow them to be unique for each step, array, and direction combination.
        tag = step << total_arrays_bits + direction_bits
        tag += self.index << direction_bits

        return tag + direction, tag + direction.opposite
