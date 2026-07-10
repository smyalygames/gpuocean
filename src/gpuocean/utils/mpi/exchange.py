from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, wait

from mpi4py import MPI
import cupy as cp
from cupy.cuda import nccl
from cupyx.distributed import NCCLBackend
from cupy.cuda.nvtx import RangePush, RangePop
from hip._util import types

from gpuocean.utils.enum import Direction

if TYPE_CHECKING:
    from mpi4py.MPI import Request, Prequest, Comm
    from gpuocean.utils.gpu import Array2D, Array3D, GPUStream
    from .grid import Grid
    from concurrent.futures import Future


class MPIExchange:
    """
    Handles exchanging data through MPI.
    """

    def __init__(self, gpu_stream: GPUStream, grid: Grid, arrays: Iterable[Array2D], comm=MPI.COMM_WORLD,
                 nccl_id: bytes = None, use_mpi=False, mpi_persistent=True, direct_exchange=False):
        """
        Handler for exchanging data through CuPy using MPI.
        :param gpu_stream: GPU Stream for synchronizing arrays.
        :param grid: Object that handles domain decomposition.
        :param arrays: GPU arrays to handle exchanges for.
        :param comm: MPI communicator for handling exchanges.
        :param: nccl_id: ID for `NcclCommunicator`.
        :param use_mpi: Use MPI instead of NCCL.
        :param mpi_persistent: Use persistent MPI connections
        :param direct_exchange: Set to `True` to directly exchange arrays
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Creating MPIExchange.")

        self.comm = comm
        self.sim_stream = gpu_stream
        # self.exchange_stream = GPUStream(default_stream=False)
        self.grid = grid
        self.mpi_persistent = use_mpi and mpi_persistent
        self.using_nccl = True
        self.nccl: NCCLBackend | None = None
        self.nccl_comm: nccl.NcclCommunicator | None = None
        self.mpi_executor: ThreadPoolExecutor | None = None

        if nccl_id is not None:
            rank = comm.rank
            world_size = comm.size
            self.nccl_comm = nccl.NcclCommunicator(world_size, nccl_id, rank)
        elif not use_mpi:
            self.nccl = NCCLBackend(comm.size, comm.rank)

        if use_mpi:
            self.mpi_executor = ThreadPoolExecutor()

        self.logger.info("CuPy is using device %d.", cp.cuda.runtime.getDevice())

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

        self.total_arrays = len(arrays)

        # Creates CuPy arrays for MPI exchanges.
        self.exchange_arrays: dict[Array2D, ArrayExchange | MPIArrayExchange] = {}
        for index, array in enumerate(arrays):
            if self.mpi_persistent:
                exchanges = MPIArrayExchange(index, array)
            else:
                exchanges = ArrayExchange(index, array)

            for direction in self.exists:
                download_direction = direction.name.lower()
                send_buf = array.download_boundary(self.sim_stream, download_direction, copy=False, ghost_cells=False)
                recv_buf = array.download_boundary(self.sim_stream, download_direction, copy=False, ghost_cells=True)

                if self.mpi_persistent:
                    exchange_rank = self.grid.get_neighbor(direction).rank
                    send_tag, recv_tag = exchanges.get_tags(self.step_number, self.total_arrays, direction)
                    exchanges[direction] = ExchangeMPI(send=send_buf, recv=recv_buf,
                                                       send_tag=send_tag, recv_tag=recv_tag,
                                                       comm=self.comm, rank=exchange_rank)
                else:
                    exchanges[direction] = Exchange(send_buf, recv_buf)

            self.exchange_arrays[array] = exchanges

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.nccl_comm is not None:
            self.nccl_comm.destroy()
        if self.nccl is not None:
            self.nccl.stop()
        if self.mpi_executor is not None:
            self.mpi_executor.shutdown()

    def _prepare_exchanges(self, gpu_stream: GPUStream, exchanges: dict[Array2D, ArrayExchange] = None):
        """
        Prepares an exchange for use in MPI or NCCL.
        """
        if exchanges is None:
            exchanges = self.exchange_arrays.items()

        for array, exchange in exchanges.items():
            exchange.prepare_send(gpu_stream)

    def exchange(self, arrays: list[Array2D] = None):
        """
        Completes the MPI exchange for all the arrays.
        """
        RangePush("run/exchange")

        if arrays is not None:
            exchange_arrays = {array: self.exchange_arrays[array] for array in arrays if array in self.exchange_arrays}
        else:
            exchange_arrays = self.exchange_arrays

        self._prepare_exchanges(self.sim_stream, exchange_arrays)
        # self.exchange_stream.synchronize()

        if self.nccl_comm is not None:
            self.nccl_comm_exchange(exchange_arrays)
        elif self.nccl is not None:
            self.nccl_exchange(exchange_arrays)
        elif self.mpi_persistent:
            self.sim_stream.synchronize()
            self.mpi_persistent_exchange(exchange_arrays)
        else:
            self.sim_stream.synchronize()
            self.mpi_exchange(exchange_arrays)

        RangePop()
        # self.exchange_stream.synchronize()

    def mpi_exchange(self, exchanges: dict[Array2D, ArrayExchange]):
        """
        Uses mpi4py to exchange data between arrays.
        """
        comm_send: list[Request] = []
        comm_recv: list[Request] = []

        # Send MPI data.
        for array, exchange in exchanges.items():

            for direction in self.exists:
                exchange_rank = self.grid.get_neighbor(direction).rank
                send_tag, recv_tag = exchange.get_tags(self.step_number, self.total_arrays, direction)
                array_exchange = self.exchange_arrays[array][direction]

                self.logger.debug("Sending from %d to %d (%s), "
                                  f"shape: (%d, %d), send tag: %d, receive tag: %d.",
                                  self.comm.rank, exchange_rank, direction.name, array_exchange.send.shape[0], array_exchange.send.shape[1], send_tag, recv_tag)
                comm_send.append(self.comm.Isend(array_exchange.send, dest=exchange_rank, tag=send_tag))
                comm_recv.append(self.comm.Irecv(array_exchange.recv, source=exchange_rank, tag=recv_tag))

        # Wait to receive all arrays
        for comm in comm_recv:
            comm.wait()

        self.logger.debug("Rank %d received all data for transfer %d", self.comm.rank, self.step_number)

        for exchange in exchanges.values():
            exchange.upload_received(self.gpu_stream)

        # Wait for transfers to complete
        for comm in comm_send:
            comm.wait()

        self.logger.debug("Rank %d sent all data for transfer %d", self.comm.rank, self.step_number)
        self.step_number += 1
    def mpi_persistent_exchange(self, exchanges: dict[Array2D, MPIArrayExchange]):
        """
        Uses persistent connections with mpi4py to exchange data between arrays.
        """
        arrays: list[MPIArrayExchange] = []
        directions: list[Direction] = []
        comm_send: list[Prequest] = []
        comm_recv: list[Prequest] = []

        RangePush("run/exchange/send")
        for array, array_exchange in exchanges.items():
            RangePush(f"run/exchange/send/{array_exchange.index}")
            if not isinstance(array_exchange, MPIArrayExchange):
                raise RuntimeError(f"Wrong type of ArrayExchange received."
                                   f" Expected `MPIArrayExchange`, got `{array_exchange.__class__.__name__}` instead.")
            for direction in self.exists:
                RangePush(f"run/exchange/send/{array_exchange.index}/{direction.name.lower()}")
                arrays.append(array_exchange)
                directions.append(direction)

                exchange = array_exchange[direction]
                if exchange is None:
                    raise RuntimeError(
                        f"Exchange was not prepared for the {direction.name} exchange, got `{exchange}` from `{array_exchange}`.")

                comm_send.append(exchange.mpi_send)
                comm_recv.append(exchange.mpi_recv)
                RangePop()
            RangePop()

        RangePush("run/exchange/send/prequest/recv")
        MPI.Prequest.Startall(comm_recv)
        RangePop()
        RangePush("run/exchange/send/prequest/send")
        send_futures: list[Future] = []
        for send in comm_send:
            send_futures.append(self.mpi_executor.submit(MPI.Prequest.Startall, comm_send))
        RangePop()

        for i in range(len(directions)):
            RangePush(f"run/exchange/send/exchange/wait/{i}")
            index = MPI.Request.Waitany(comm_recv)
            RangePop()

            # arrays[index].upload_direction(directions[index], self.exchange_stream)
            arrays[index].upload_direction(directions[index], self.sim_stream)

        wait(send_futures)

        MPI.Request.Waitall(comm_send)
        RangePop()

    def nccl_exchange(self, exchanges: dict[Array2D, ArrayExchange]):
        """
        Uses the CuPyx `NCCLBackend` to communicate NCCL/RCCL exchanges.
        """
        if self.nccl is None:
            return RuntimeError("NCCLBackend is not initialized.")

        # Send MPI data.
        for array, exchange in exchanges.items():
            for direction in self.exists:
                exchange_rank = self.grid.get_neighbor(direction).rank
                array_exchange = self.exchange_arrays[array].get_direction(direction)

                self.logger.debug("Sending from %d to %d (%s), shape: (%d, %d).",
                                  self.comm.rank, exchange_rank, direction.name,
                                  array_exchange.send.shape[0], array_exchange.send.shape[1])
                self.nccl.send_recv(array_exchange.send, array_exchange.recv, exchange_rank,
                                    self.gpu_stream._cupy_stream)

        self.logger.debug("Rank %d exchanged all data %d", self.comm.rank, self.step_number)
        self.step_number += 1

    def nccl_comm_exchange(self, exchanges: dict[Array2D, ArrayExchange]):
        """
        Uses the CuPy `NcclCommunicator` to communicate NCCL/RCCL exchanges.
        """
        self.logger.debug("Starting NCCL exchange")
        _NCCL_DTYPE_MAP = {
            cp.dtype('float32'): nccl.NCCL_FLOAT32,
            cp.dtype('float64'): nccl.NCCL_FLOAT64,
        }

        try:
            stream_ptr = self.gpu_stream._cupy_stream.ptr
        except AttributeError:
            # Fallback if _cupy_stream is not available
            stream_ptr = int(self.gpu_stream.pointer)

        nccl.groupStart()
        index = 0
        for array, exchange in exchanges.items():
            self.logger.debug("Starting NCCL exchange for %d", index)
            index += 1
            for direction in self.exists:
                self.logger.debug("Exchanging for %s", direction.name)
                exchange_rank = self.grid.get_neighbor(direction).rank
                array_exchange = exchanges[array].get_direction(direction)

                send_buf: cp.ndarray = array_exchange.send
                recv_buf: cp.ndarray = array_exchange.recv

                nccl_dtype = _NCCL_DTYPE_MAP[send_buf.dtype]

                self.logger.debug(
                    f"NCCL sending from {self.comm.rank} to {exchange_rank} ({direction.value}), "
                    f"shape: {send_buf.shape}."
                )

                self.nccl_comm.send(send_buf.data.ptr, send_buf.size, nccl_dtype, exchange_rank, stream_ptr)
                self.nccl_comm.recv(recv_buf.data.ptr, recv_buf.size, nccl_dtype, exchange_rank, stream_ptr)

        nccl.groupEnd()

        self.logger.debug("Rank %d completed NCCL exchange for transfer %d", self.comm.rank, self.step_number)
        self.step_number += 1


@dataclass
class Exchange:
    send: cp.ndarray
    recv: cp.ndarray
    copy: bool = field(init=False)

    def __post_init__(self):
        original = self.send
        self.send = cp.ascontiguousarray(self.send)

        self.copy = original.data.ptr != self.send.data.ptr

        if self.copy:
            self.recv = cp.ascontiguousarray(self.recv)


@dataclass
class ExchangeMPI(Exchange):
    comm: Comm
    rank: int
    send_tag: int
    recv_tag: int
    mpi_send: Prequest = field(init=False)
    mpi_recv: Prequest = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        self.mpi_send = self.comm.Send_init(self.send, self.rank, self.send_tag)
        self.mpi_recv = self.comm.Recv_init(self.recv, self.rank, self.recv_tag)

    def __del__(self):
        self.mpi_send.Free()
        self.mpi_recv.Free()


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
        if self.north is not None and self.north.copy:
            cp.copyto(self.north.send, self.array.download_boundary(gpu_stream, "south", copy=False))
        if self.east is not None and self.east.copy:
            cp.copyto(self.east.send, self.array.download_boundary(gpu_stream, "east", copy=False))
        if self.south is not None and self.south.copy:
            cp.copyto(self.south.send, self.array.download_boundary(gpu_stream, "north", copy=False))
        if self.west is not None and self.west.copy:
            cp.copyto(self.west.send, self.array.download_boundary(gpu_stream, "west", copy=False))

        # gpu_stream.synchronize()

    def upload_received(self, gpu_stream):
        """
        Updates the array with all the received boundary data.
        """
        if self.north is not None and self.north.copy:
            self.array.upload_boundary(gpu_stream, self.north.recv, "south")
        if self.east is not None and self.east.copy:
            self.array.upload_boundary(gpu_stream, self.east.recv, "east")
        if self.south is not None and self.south.copy:
            self.array.upload_boundary(gpu_stream, self.south.recv, "north")
        if self.west is not None and self.west.copy:
            self.array.upload_boundary(gpu_stream, self.west.recv, "west")

    def __getitem__(self, item: Direction) -> Exchange | None:
        """
        Gets a direction's Exchange.
        :param item: Direction of which variable to get from.
        """
        match item:
            case Direction.NORTH:
                return self.north
            case Direction.EAST:
                return self.east
            case Direction.SOUTH:
                return self.south
            case Direction.WEST:
                return self.west
            case _:
                raise RuntimeError(f"{item} is an invalid direction to get a variable from.")

    def __setitem__(self, key: Direction, value: Exchange | None) -> None:
        """
        Set a direction's Exchange.
        :param key: Direction of which variable to set to.
        """
        match key:
            case Direction.NORTH:
                self.north = value
            case Direction.EAST:
                self.east = value
            case Direction.SOUTH:
                self.south = value
            case Direction.WEST:
                self.west = value
            case _:
                raise RuntimeError(f"Attempted to access invalid key, `{key}`, in {self.__class__.__name__}.")

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


@dataclass
class MPIArrayExchange(ArrayExchange):
    north: ExchangeMPI | None = None
    east: ExchangeMPI | None = None
    south: ExchangeMPI | None = None
    west: ExchangeMPI | None = None

    def __getitem__(self, item) -> ExchangeMPI | None:
        return super().__getitem__(item)

    def __setitem__(self, key, value: ExchangeMPI | None) -> None:
        super().__setitem__(key, value)

    def upload_direction(self, direction: Direction, gpu_stream) -> None:
        """
        Uploads for a given direction.
        :param direction: Direction of which array to upload.
        :param gpu_stream: Stream for uploading data to device.
        """
        exchange = self.__getitem__(direction)

        if exchange is not None and exchange.copy:
            self.array.upload_boundary(gpu_stream, exchange.recv, direction.name.lower())
