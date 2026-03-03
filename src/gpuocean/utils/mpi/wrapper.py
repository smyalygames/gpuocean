from __future__ import annotations
from typing import TYPE_CHECKING
from enum import IntEnum
from dataclasses import dataclass
import logging

import numpy as np
import cupy as cp
from mpi4py import MPI

from gpuocean.utils.gpu import Array2D
from gpuocean.utils.Common import BoundaryConditions, BoundaryType
from gpuocean.SWEsimulators import SimulatorType
from gpuocean.utils.dataclass import GhostCells

from .grid import Grid

if TYPE_CHECKING:
    from mpi4py.MPI import Request
    from gpuocean.utils.types import AnySimulator


class MPIWrapper:
    """
    An MPI wrapper for the SWE simulator schemes.
    """

    def __init__(self, simulator_type: SimulatorType, global_nx: int, global_ny: int,
                 ghost_cells: tuple[int, int, int, int],
                 comm=MPI.COMM_WORLD, boundary_conditions=BoundaryConditions(), *args, **kwargs):
        """
        Creates a wrapper for the simulator, and the simulator chosen.
        :param simulator_type: Simulator type to create from the given arguments.
        :param global_nx: Size of the global domain in the x-axis.
        :param global_ny: Size of the global domain in the y-axis.
        :param ghost_cells: A tuple consisting of the number of ghost cells in the directions of (north, east, south, west).
        :param comm: MPI interface.
        :param args: Positional arguments for the specified simulator.
        :param kwargs: Keyword arguments for the specified simulator.
        """
        self.logger = logging.getLogger(__name__)
        self.comm = comm
        total_nodes = self.comm.size
        rank = self.comm.rank

        self.logger.info(f"Rank: {rank}, Total Ranks: {total_nodes}.")

        self.global_nx = global_nx
        self.global_ny = global_ny
        self.grid = Grid(global_nx, global_ny, total_nodes, rank)
        self.logger.debug(f"Decomposed domain is: ({self.grid.local_nx}, {self.grid.local_ny}) "
                          f"from global domain size ({self.global_nx}, {self.global_ny}).")

        # Create boundary conditions
        boundary_conditions_args = {
            'north': boundary_conditions.north,
            'east': boundary_conditions.east,
            'south': boundary_conditions.south,
            'west': boundary_conditions.west,
            'sponge_cells': boundary_conditions.spongeCells
        }

        if not boundary_conditions.isPeriodicNorthSouth():
            if self.grid.north is not None:
                boundary_conditions_args['north'] = BoundaryType.DIRICHLET
            if self.grid.south is not None:
                boundary_conditions_args['south'] = BoundaryType.DIRICHLET
        if not boundary_conditions.isPeriodicEastWest():
            if self.grid.east is not None:
                boundary_conditions_args['east'] = BoundaryType.DIRICHLET
            if self.grid.west is not None:
                boundary_conditions_args['west'] = BoundaryType.DIRICHLET

        boundary_conditions = BoundaryConditions(**boundary_conditions_args)

        # Decompose the information
        kwargs.update({'nx': self.grid.local_nx, 'ny': self.grid.local_ny, 'comm': self.comm,
                       'boundary_conditions': boundary_conditions})

        ghost_cells = GhostCells(*ghost_cells)

        original_shape = (global_ny + ghost_cells.total_y, global_nx + ghost_cells.total_x)

        splice_x0 = self.grid.x0
        splice_x1 = self.grid.x1 + ghost_cells.total_x
        splice_y0 = self.grid.y0
        splice_y1 = self.grid.y1 + ghost_cells.total_y

        self.logger.debug(f"Splicing arrays with [{splice_y0}:{splice_y1}, {splice_x0}:{splice_x1}].")

        # Go through arrays in args
        args = list(args)
        for i in range(len(args)):
            var = args[i]
            if isinstance(var, np.ndarray):
                if var.shape == original_shape:
                    args[i] = var[splice_y0:splice_y1, splice_x0:splice_x1]
                elif var.shape == (original_shape[0] + 1, original_shape[1] + 1):
                    args[i] = var[splice_y0:splice_y1 + 1, splice_x0:splice_x1 + 1]

        args = tuple(args)

        # Go through arrays in kwargs
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                if value.shape == original_shape:
                    kwargs[key] = value[splice_y0:splice_y1, splice_x0:splice_x1]
                elif value.shape == (original_shape[0] + 1, original_shape[1] + 1):
                    kwargs[key] = value[splice_y0:splice_y1 + 1, splice_x0:splice_x1 + 1]

        # Create the simulator
        self.sim: AnySimulator = simulator_type.value(*args, **kwargs)

        # Check if dt needs to calculated
        if kwargs['dt'] <= 0:
            self.update_dt()

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

    def step(self, t_end=0.0, update_dt=False):
        t_now = 0.0

        if t_end == 0:
            self._exchange()
            self.sim.step(t_end)

        while t_now < t_end:
            self._exchange()
            if update_dt:
                self.update_dt()

            t_now += self.sim.dt
            self.sim.step(self.sim.dt)

    def update_dt(self, courant_number: float = None):
        """
        Updates the time step self.dt by finding the maximum size of dt according to the
        CFL conditions, and scale it with the provided courant number (0.8 on default).
        """
        if not isinstance(self.sim, SimulatorType.CDKLM16.value):
            raise TypeError(f"Cannot update time step size (dt) with simulator type: {type(self.sim)}. "
                            f"Only CDKLM16 simulator is supported.")

        # Can probably remove the async call and just run self.sim.updateDt(), and allreduce self.sim.dt

        if courant_number is None:
            courant_number = self.sim.courant_number

        self.sim.per_block_max_dt_kernel.async_call(self.sim.global_size, self.sim.local_size, self.sim.gpu_stream,
                                                    [self.sim.nx, self.sim.ny,
                                                     self.sim.dx, self.sim.dy,
                                                     self.sim.g,
                                                     self.sim.gpu_data.h0.pointer, self.sim.gpu_data.h0.pitch,
                                                     self.sim.gpu_data.hu0.pointer, self.sim.gpu_data.hu0.pitch,
                                                     self.sim.gpu_data.hv0.pointer, self.sim.gpu_data.hv0.pitch,
                                                     self.sim.bathymetry.Bm.pointer, self.sim.bathymetry.Bm.pitch,
                                                     self.sim.bathymetry.mask_value,
                                                     self.sim.device_dt.pointer, self.sim.device_dt.pitch])

        self.sim.max_dt_reduction_kernel.async_call((1, 1),
                                                    (self.sim.num_threads_dt, 1, 1),
                                                    self.sim.gpu_stream,
                                                    [self.sim.num_blocks_dt,
                                                     self.sim.device_dt.pointer,
                                                     self.sim.max_dt_buffer.pointer])

        dt_host = self.sim.max_dt_buffer.download(self.gpu_stream)

        global_dt_max = np.zeros(1, dtype=np.float32)

        self.comm.Allreduce(dt_host, global_dt_max, op=MPI.MIN)

        if global_dt_max == 0:
            raise RuntimeError("New timestep (dt) is zero.")

        self.logger.debug(f"New dt is: {global_dt_max[0]}. Local dt was: {dt_host[0][0]}.")

        self.sim.dt = courant_number * float(global_dt_max)

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
            self.north.send = array.download_boundary(gpu_stream, "south")
        if self.east is not None:
            self.east.send = array.download_boundary(gpu_stream, "east")
        if self.south is not None:
            self.south.send = array.download_boundary(gpu_stream, "north")
        if self.west is not None:
            self.west.send = array.download_boundary(gpu_stream, "west")

        gpu_stream.synchronize()

    def upload_received(self, array: Array2D, gpu_stream):
        """
        Updates the array with all the received boundary data.
        """
        if self.north is not None:
            array.upload_boundary(gpu_stream, self.north.recv, "south")
        if self.east is not None:
            array.upload_boundary(gpu_stream, self.east.recv, "east")
        if self.south is not None:
            array.upload_boundary(gpu_stream, self.south.recv, "north")
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
