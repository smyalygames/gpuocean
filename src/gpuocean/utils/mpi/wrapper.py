from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np
import numpy.typing as npt
import cupy as cp
from cupy.cuda import nccl
from mpi4py import MPI
from tqdm import tqdm

from gpuocean.utils.gpu import Array2D, GPUHandler, GPUStream
from gpuocean.utils.Common import BoundaryConditions, BoundaryType
from gpuocean.utils.dataclass import GhostCells

from .grid import Grid
from .exchange import MPIExchange

if TYPE_CHECKING:
    from gpuocean.SWEsimulators import SimulatorType
    from gpuocean.utils.types import AnySimulator
    from hip._util import types
    from mpi4py.MPI import Prequest


class MPIWrapper:
    """
    An MPI wrapper for the SWE simulator schemes.
    """

    def __init__(self, simulator_type: SimulatorType, global_nx: int, global_ny: int,
                 ghost_cells: tuple[int, int, int, int], strong_scale: bool = True,
                 comm=MPI.COMM_WORLD, use_nccl=False, mpi_persistent=True, boundary_conditions: BoundaryConditions | None =None, *args, **kwargs):
        """
        Creates a wrapper for the simulator, and the simulator chosen.
        :param simulator_type: Simulator type to create from the given arguments.
        :param global_nx: Size of the global domain in the x-axis.
        :param global_ny: Size of the global domain in the y-axis.
        :param ghost_cells: A tuple consisting of the number of ghost cells in the directions of (north, east, south, west).
        :param comm: MPI interface.
        :param use_nccl: Set to `True` to use NCCL, otherwise MPI will be used.
        :param mpi_persistent: When using MPI, will use persistent connections.
        :param args: Positional arguments for the specified simulator.
        :param kwargs: Keyword arguments for the specified simulator.
        """
        self.logger = logging.getLogger(__name__)
        self.comm = comm
        self.total_nodes = self.comm.size
        rank = self.comm.rank

        self.logger.info(f"Rank: {rank}, Total Ranks: {self.total_nodes}.")

        self.global_nx = global_nx
        self.global_ny = global_ny
        self.grid = Grid(global_nx, global_ny, self.total_nodes, rank, strong_scale=strong_scale, use_nccl=use_nccl)
        self.logger.debug(f"Decomposed domain is: ({self.grid.local_nx}, {self.grid.local_ny}) "
                          f"from global domain size ({self.global_nx}, {self.global_ny}).")

        if boundary_conditions is None:
            if "boundary_conditions" in kwargs:
                boundary_conditions = kwargs["boundary_conditions"]
            else:
                boundary_conditions = BoundaryConditions()
                self.logger.warning("No boundary conditions defined, using default boundary conditions %s", boundary_conditions)

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

        local_boundary_conditions = BoundaryConditions(**boundary_conditions_args)

        # Decompose the information
        kwargs.update({'comm': self.comm, 'boundary_conditions': local_boundary_conditions})

        self.ghost_cells = GhostCells(*ghost_cells)

        self._original_shape = (self.global_ny + self.ghost_cells.total_y, self.global_nx + self.ghost_cells.total_x)

        if strong_scale:
            kwargs.update({'nx': self.grid.local_nx, 'ny': self.grid.local_ny})
            self.logger.debug("New nx: %d, new ny: %d", kwargs['nx'], kwargs['ny'])
            self._splice_x0 = self.grid.x0
            self._splice_x1 = self.grid.x1 + self.ghost_cells.total_x
            self._splice_y0 = self.grid.y0
            self._splice_y1 = self.grid.y1 + self.ghost_cells.total_y

            self.logger.debug(
                f"Splicing arrays with [{self._splice_y0}:{self._splice_y1}, {self._splice_x0}:{self._splice_x1}] Original sape: {self._original_shape}.")

            # Go through arrays in args
            args = list(args)
            for i in range(len(args)):
                var = args[i]
                if isinstance(var, np.ndarray):
                    args[i] = self._decompose_array(var)

            args = tuple(args)

            # Go through arrays in kwargs
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    kwargs[key] = self._decompose_array(value)
        else:
            kwargs.update({'nx': self.global_nx, 'ny': self.global_ny})
            self.logger.debug("New nx: %d, new ny: %d", kwargs['nx'], kwargs['ny'])


        update_dt = False
        if kwargs['dt'] <= 0:
            update_dt = True
            kwargs['dt'] = 1

        # Create the simulator
        self.sim: AnySimulator = simulator_type.value(*args, **kwargs)
        # Write the global boundary conditions to netCDF
        if kwargs['write_netcdf']:
            self.sim.sim_writer.nc.boundary_conditions = str(boundary_conditions)
            self.sim.sim_writer.nc.boundary_conditions_sponge_mr = str(boundary_conditions.getSponge())

        # self.dt_stream = GPUStream(default_stream=False)
        # with self.dt_stream._cupy_stream:
        self.global_dt = cp.empty_like(self.sim.max_dt_buffer.data, shape=1)

        if self.total_nodes > 1:
            self.mpi_dt = None
            # Only use NCCL when each MPI rank has its own GPU device;
            # NCCL does not support multiple ranks sharing the same device.
            from gpuocean.utils.gpu import gpu_device
            nccl_id = None
            local_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
            # For jobs with multiple nodes (not processes)
            local_size = local_comm.Get_size()
            local_comm.Free()

            local_enough_gpus = 1 if gpu_device.get_device_count() >= local_size else 0

            global_enough_gpus = self.comm.allreduce(local_enough_gpus, op=MPI.MIN)

            if global_enough_gpus and use_nccl:
                if self.comm.rank == 0:
                    nccl_id = nccl.get_unique_id()
                nccl_id = self.comm.bcast(nccl_id, root=0)
            else:
                if use_nccl:
                    raise RuntimeError("Not enough GPUs allocated for each process.")
                # self.mpi_dt: Prequest = self.comm.Allreduce_init(self.sim.max_dt_buffer.data, self.global_dt, op=MPI.MIN)
                self.mpi_dt = None
            self.mpi_handler = MPIExchange(self.sim.gpu_stream, self.grid, self._get_domains(),
                                           self.comm, nccl_id, use_mpi=not use_nccl, mpi_persistent=mpi_persistent)
            # Add exchange function to GPUHandler
            GPUHandler.mpi_exchange_func = self.exchange_pointers
        else:
            self.mpi_handler = None

        # Check if dt needs to calculated
        if update_dt:
            self.update_dt()

    def _decompose_array(self, array: npt.NDArray) -> npt.NDArray:
        """
        Decomposes an array, also checks if the array is valid.
        """
        if array.shape == self._original_shape:
            return array[self._splice_y0:self._splice_y1, self._splice_x0:self._splice_x1]
        elif array.shape == (self._original_shape[0] + 1, self._original_shape[1] + 1):
            return array[self._splice_y0:self._splice_y1 + 1, self._splice_x0:self._splice_x1 + 1]

        # If the array is not something that can be decomposed, just return it anyway
        return array

    def reinit(self, eta: npt.NDArray, hu: npt.NDArray, hv: npt.NDArray, dt: float = 0):
        """
        Re-initializes the simulator to a given state.
        """
        if self.grid.strong_scale:
            eta_splice = self._decompose_array(eta)
            hu_splice = self._decompose_array(hu)
            hv_splice = self._decompose_array(hv)
        else:
            eta_splice = eta
            hu_splice = hu
            hv_splice = hv

        # Upload data to simulator
        self.sim.upload(eta_splice, hu_splice, hv_splice)
        self.sim.t = 0
        if dt <= 0:
            self.update_dt()
        else:
            self.sim.dt = dt

        self.sim.num_iterations = 0
        self.sim.wind_stress_timestamps = {}
        self.sim.atmospheric_pressure_timestamps = {}
        self.sim.init_count += 1

        if self.sim.write_netcdf:
            self.sim.sim_writer.sim_reinit(self.sim)


    def __getattr__(self, item):
        return getattr(self.sim, item)

    def _get_domains(self) -> list[Array2D]:
        """
        Gets the domains to exchange data through MPI.
        """
        return self.sim.arrays

    def step(self, t_end=0.0, update_dt=False, split_step=True, write_now=True, enable_progress_bar=True) -> float:
        """
        Start simulation for given amount of time.
        :param t_end: Amount of seconds to simulate for. Leave at `0` for only computing one step.
        :param update_dt: Set to `True` to dynamically update the dt per time-step.
        :param split_step: Compute halo first, then the inner domain.
        :param write_now: Write to netCDF after each time-step.
        :param enable_progress_bar: Enable progress bar.
        """
        t_now = 0.0

        if t_end == 0:
            self.sim.step(t_end, split_step=split_step)

        with tqdm(desc=f"Running a {t_end:.3f} second simulation", total=t_end, unit="ss", bar_format="{n:.3f}", disable=not enable_progress_bar) as pbar:
            while t_now < t_end:
                t_now += self.sim.dt
                self.sim.step(self.sim.dt, write_now=write_now, split_step=split_step)

                pbar.update(self.sim.dt)

                if update_dt:
                    self.update_dt()

        return self.sim.t

    def update_dt(self, courant_number: float = None):
        """
        Updates the time step self.dt by finding the maximum size of dt according to the
        CFL conditions, and scale it with the provided courant number (0.8 on default).
        """
        if self.sim.__class__.__name__ != "CDKLM16":
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
        if self.mpi_handler is None:
            with self.sim.gpu_stream._cupy_stream:
                cp.copyto(self.global_dt, self.sim.max_dt_buffer.data)
        elif self.mpi_handler.nccl_comm is not None:
            self.mpi_handler.nccl_comm.allReduce(self.sim.max_dt_buffer.data.data.ptr, self.global_dt.data.ptr, 1,
                                                 nccl.NCCL_FLOAT32, nccl.NCCL_MIN, self.sim.gpu_stream._cupy_stream.ptr)
        elif self.mpi_handler.nccl is not None:
            self.mpi_handler.nccl.all_reduce(self.sim.max_dt_buffer.data, self.global_dt, 'min', self.sim.gpu_stream._cupy_stream)
        else:
            self.sim.gpu_stream.synchronize()
            if self.mpi_dt is not None:
                self.mpi_dt.Start()
                self.mpi_dt.Wait()
            else:
                self.comm.Allreduce(self.sim.max_dt_buffer.data, self.global_dt, op=MPI.MIN)

        if self.global_dt == 0:
            raise RuntimeError(f"New timestep (dt) is zero. Received: {self.global_dt}, Local: {self.max_dt_buffer}")

        # TODO removed logging as it's unclear if it would degrade performance having to download from GPU.
        # self.logger.debug(f"New dt is: {self.global_dt[0]}. Local dt was: {dt_host[0][0]}.")

        self.sim.dt = courant_number * float(self.global_dt)

    def exchange_pointers(self, pointers: list[types.Pointer | cp.cuda.MemoryPointer]) -> None:
        """
        Exchanges a set `Array2D`s from their pointers.
        :param pointers: Pointers to the Array2D class.
        """
        arrays = [array for array in self.mpi_handler.exchange_arrays.keys() if array.pointer in pointers]

        self.mpi_handler.exchange(arrays)

    def cleanUp(self):
        self.sim.cleanUp()
        if self.mpi_dt is not None:
            self.mpi_dt.Free()

    def download(self, interior_domain_only=False, root: int = 0) -> tuple[
                                                                         npt.NDArray, npt.NDArray, npt.NDArray] | None:
        """
        Download the latest timestep from the GPU.
        :param interior_domain_only: ``False`` to include ghost cells, and ``True`` to not include them in the download.
        :param root: MPI rank to handle gathering all the domains.
        :returns: An array with all the data on the nodes combined on the root MPI process, otherwise nothing.
        """
        if not interior_domain_only:
            raise NotImplementedError("Array gathering currently does not support the extra ghost cells.")

        eta_local, hu_local, hv_local = self.sim.download(interior_domain_only)

        eta_arrays = self.comm.gather(eta_local, root=root)
        hu_arrays = self.comm.gather(hu_local, root=root)
        hv_arrays = self.comm.gather(hv_local, root=root)

        if self.comm.rank == root:
            eta = np.block(
                [[eta_arrays[x + (x * y)] for x in range(self.grid.nodes_x)] for y in range(self.grid.nodes_y)])
            hu = np.block(
                [[hu_arrays[x + (x * y)] for x in range(self.grid.nodes_x)] for y in range(self.grid.nodes_y)])
            hv = np.block(
                [[hv_arrays[x + (x * y)] for x in range(self.grid.nodes_x)] for y in range(self.grid.nodes_y)])

            return eta, hu, hv
        else:
            return None


