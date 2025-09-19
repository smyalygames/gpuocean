# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the Forward Backward Linear numerical 
scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Import packages we need
from __future__ import annotations
from typing import TYPE_CHECKING
import gc

import numpy as np
import numpy.typing as npt

from gpuocean.SWEsimulators import Simulator
from gpuocean.utils import SimWriter, SimReader, WindStress, AtmosphericPressure
from gpuocean.utils.Common import BoundaryConditions, BoundaryType
from gpuocean.utils.gpu import GPUHandler, Array2D, SWEDataArakawaC

if TYPE_CHECKING:
    from mpi4py import MPI

    from gpuocean.utils.gpu import KernelContext, GPUStream


class FBL(Simulator.Simulator):
    """
    Class that solves the SW equations using the Forward-Backward linear scheme
    """

    def __init__(self,
                 gpu_ctx: KernelContext,
                 H: npt.NDArray[np.float32], eta0: npt.NDArray[np.float32],
                 hu0: npt.NDArray[np.float32], hv0: npt.NDArray[np.float32],
                 nx: int, ny: int,
                 dx: float, dy: float, dt: float,
                 g: float, f: float, r: float,
                 t=0.0,
                 coriolis_beta=0.0,
                 y_zero_reference_cell=1,
                 wind=WindStress.WindStress(),
                 atmospheric_pressure=AtmosphericPressure.AtmosphericPressure(),
                 boundary_conditions=BoundaryConditions(),
                 write_netcdf=False,
                 comm: MPI.Intracomm = None,
                 ignore_ghostcells=False,
                 offset_x=0, offset_y=0,
                 block_width=16, block_height=16):
        """
        Initialization routine
        Args:
            H: Water depth incl ghost cells, (nx+2)*(ny+2) cells
            eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
            hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
            hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+3) cells
            nx: Number of cells along x-axis
            ny: Number of cells along y-axis
            dx: Grid cell spacing along x-axis (20 000 m)
            dy: Grid cell spacing along y-axis (20 000 m)
            dt: Size of each timestep (90 s)
            g: Gravitational accelleration (9.81 m/s^2)
            f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
            r: Bottom friction coefficient (2.4e-3 m/s)
            coriolis_beta: Coriolis linear factor -> f = f + beta*y
            y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell .
            wind: Wind stress parameters
            atmospheric_pressure: Object with values for atmospheric pressure
            boundary_conditions: Boundary condition object
            write_netcdf: Write the results after each superstep to a netCDF file
            comm: MPI communicator
        """

        #### THIS ALLOWS MAKES IT POSSIBLE TO GIVE THE OLD INPUT SHAPES TO NEW GHOST CELL REGIME: Only valid for benchmarking!
        if eta0.shape == (ny, nx):
            new_eta = np.zeros((ny + 2, nx + 2), dtype=np.float32)
            new_eta[:ny, :nx] = eta0.copy()
            eta0 = new_eta.copy()
        if H.shape == (ny, nx):
            new_H = np.ones((ny + 2, nx + 2), dtype=np.float32) * np.max(H)
            new_H[:ny, :nx] = H.copy()
            H = new_H.copy()
        if hu0.shape == (ny, nx + 1):
            new_hu = np.zeros((ny + 2, nx + 1), dtype=np.float32)
            new_hu[:ny, :nx + 1] = hu0.copy()
            hu0 = new_hu.copy()
        if hv0.shape == (ny + 1, nx):
            new_hv = np.zeros((ny + 3, nx + 2), dtype=np.float32)
            new_hv[:ny + 1, :nx] = hv0.copy()
            hv0 = new_hv.copy()

        # Create data by uploading to device
        ghost_cells_x = 1
        ghost_cells_y = 1
        y_zero_reference_cell = y_zero_reference_cell

        # Index range for interior domain (north, east, south, west)
        # so that interior domain of eta is
        # eta[self.interior_domain_indices[2]:self.interior_domain_indices[0], \
        #     self.interior_domain_indices[3]:self.interior_domain_indices[1] ]
        self.interior_domain_indices = np.array([-1, -1, 1, 1])

        self.boundary_conditions = boundary_conditions

        if boundary_conditions.isSponge():
            nx = nx - 2 + boundary_conditions.spongeCells.east + boundary_conditions.spongeCells.west
            ny = ny - 2 + boundary_conditions.spongeCells.north + boundary_conditions.spongeCells.south
            y_zero_reference_cell = y_zero_reference_cell + boundary_conditions.spongeCells.south

        rk_order = None
        theta = None
        A = None
        super(FBL, self).__init__(gpu_ctx,
                                  nx, ny,
                                  ghost_cells_x,
                                  ghost_cells_y,
                                  dx, dy, dt,
                                  g, f, r, A,
                                  t,
                                  theta, rk_order,
                                  coriolis_beta,
                                  y_zero_reference_cell,
                                  wind,
                                  atmospheric_pressure,
                                  write_netcdf,
                                  ignore_ghostcells,
                                  offset_x, offset_y,
                                  comm,
                                  block_width, block_height)
        self._set_interior_domain_from_sponge_cells()

        # Get kernels
        self.step_kernel = gpu_ctx.get_kernel("FBL_step_kernel",
                                              defines={'block_width': block_width, 'block_height': block_height,
                                                       'WIND_STRESS_X_NX': int(self.wind_stress.wind_u[0].shape[1]),
                                                       'WIND_STRESS_X_NY': int(self.wind_stress.wind_u[0].shape[0]),
                                                       'WIND_STRESS_Y_NX': int(self.wind_stress.wind_v[0].shape[1]),
                                                       'WIND_STRESS_Y_NY': int(self.wind_stress.wind_v[0].shape[0]), },
                                              compile_args={
                                                  'no_extern_c': True,
                                                  'options': ["--use_fast_math"],
                                                  # 'options': ["--generate-line-info"],
                                                  # 'options': ["--maxrregcount=32"]
                                                  # 'arch': "compute_50",
                                                  # 'code': "sm_50"
                                              },
                                              jit_compile_args={
                                                  # jit_options=[(cuda.jit_option.MAX_REGISTERS, 39)]
                                              }
                                              )

        # Get CUDA functions 
        self.fblStepKernel = GPUHandler(self.step_kernel, "fblStepKernel", "iiffffffffPiPiPiPiiPPPPf")

        # Set up textures
        self.update_wind_stress(self.step_kernel)

        self.H = Array2D(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, H)
        self.gpu_data = SWEDataArakawaC(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0, fbl=True)

        # Domain including ghost cells
        self.nx_halo = int(nx + 2)
        self.ny_halo = int(ny + 2)

        self.bc_kernel = FBLBoundaryConditions(self.gpu_ctx,
                                               self.nx,
                                               self.ny,
                                               self.boundary_conditions
                                               )

        # Bit-wise boolean for wall boundary conditions
        self.wall_bc = 0
        if self.boundary_conditions.north == BoundaryType.WALL:
            self.wall_bc += 1
        if self.boundary_conditions.east == BoundaryType.WALL:
            self.wall_bc += 2
        if self.boundary_conditions.south == BoundaryType.WALL:
            self.wall_bc += 4
        if self.boundary_conditions.west == BoundaryType.WALL:
            self.wall_bc += 8

        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self, ignore_ghostcells=self.ignore_ghostcells,
                                                        staggered_grid=True,
                                                        offset_x=self.offset_x, offset_y=self.offset_y)

    @classmethod
    def fromfilename(cls, gpu_ctx: KernelContext, filename: str, cont_write_netcdf=True):
        """
        Initialize and hotstart simulation from nc-file.
        cont_write_netcdf: Continue to write the results after each superstep to a new netCDF file
        filename: Continue simulation based on parameters and last timestep in this file
        """
        # open nc-file
        sim_reader = SimReader.SimNetCDFReader(filename, ignore_ghostcells=False)
        sim_name = str(sim_reader.get('simulator_short'))
        assert sim_name == cls.__name__, \
            "Trying to initialize a " + \
            cls.__name__ + " simulator with netCDF file based on " \
            + sim_name + " results."

        # read parameters
        nx = sim_reader.get("nx")
        ny = sim_reader.get("ny")

        dx = sim_reader.get("dx")
        dy = sim_reader.get("dy")

        width = nx * dx
        height = ny * dy

        dt = sim_reader.get("dt")
        g = sim_reader.get("g")
        r = sim_reader.get("bottom_friction_r")
        A = sim_reader.get("eddy_viscosity_coefficient")
        f = sim_reader.get("coriolis_force")
        beta = sim_reader.get("coriolis_beta")

        minmodTheta = sim_reader.get("minmod_theta")
        timeIntegrator = sim_reader.get("time_integrator")
        y_zero_reference_cell = sim_reader.get("y_zero_reference_cell")

        wind = WindStress.WindStress()

        boundaryConditions = sim_reader.getBC()

        h0 = sim_reader.getH()

        # get last timestep (including simulation time of last timestep)
        eta0, hu0, hv0, time0 = sim_reader.getLastTimeStep()

        return cls(gpu_ctx,
                   h0, eta0, hu0, hv0,
                   nx, ny,
                   dx, dy, dt,
                   g, f, r,
                   t=time0,
                   wind=wind,
                   boundary_conditions=boundaryConditions,
                   write_netcdf=cont_write_netcdf)

    def cleanUp(self):
        """
        Clean up function
        """
        self.closeNetCDF()

        self.gpu_data.release()

        self.H.release()

        self.gpu_ctx = None
        gc.collect()

    # Over-riding Simulator's download.
    def download(self, interior_domain_only=False):
        """
        Download the latest time step from the GPU
        """
        return self.gpu_data.download(self.gpu_stream,
                                      interior_domain_only=interior_domain_only)

    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        n = int(t_end / self.dt + 1)

        ## Populate all ghost cells before we start
        if self.t == 0:
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)
            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)

        for i in range(0, n):
            local_dt = float(min(self.dt, t_end - i * self.dt))

            # if self.totalNumIterations > 240:
            # if self.totalNumIterations > 5:
            #    self.t += t_end
            #    return self.t

            if local_dt <= 0.0:
                break

            wind_stress_t = float(self.update_wind_stress(self.step_kernel))

            self.fblStepKernel.async_call(self.global_size, self.local_size, self.gpu_stream,
                                          [self.nx, self.ny,
                                           self.dx, self.dy, local_dt,
                                           self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, self.r,
                                           self.H.pointer, self.H.pitch,
                                           self.gpu_data.hu0.pointer, self.gpu_data.hu0.pitch,
                                           self.gpu_data.hv0.pointer, self.gpu_data.hv0.pitch,
                                           self.gpu_data.h0.pointer, self.gpu_data.h0.pitch,
                                           self.wall_bc,
                                           self.wind_stress_x_current_arr.pointer,
                                           self.wind_stress_x_next_arr.pointer,
                                           self.wind_stress_y_current_arr.pointer,
                                           self.wind_stress_y_next_arr.pointer,
                                           wind_stress_t])

            # Fix U boundary
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)

            # Fix V boundary
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)

            # Fix eta boundary
            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)

            self.t += np.float64(local_dt)
            self.num_iterations += 1

        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)

        return self.t


class FBLBoundaryConditions:
    def __init__(self,
                 gpu_ctx: KernelContext,
                 nx: int, ny: int,
                 boundary_conditions: BoundaryConditions,
                 block_width=16, block_height=16):

        self.boundary_conditions = boundary_conditions
        self.ghostsX = 1
        self.ghostsY = 1

        self.bc_north = boundary_conditions.north
        self.bc_east = boundary_conditions.east
        self.bc_south = boundary_conditions.south
        self.bc_west = boundary_conditions.west

        self.nx = nx
        self.ny = ny
        self.nx_halo = nx + self.ghostsX
        self.ny_halo = ny + self.ghostsY

        # Debugging variables
        debug = False
        self.firstGhostU = debug
        self.firstGhostV = debug
        self.firstGhostEta = debug

        # Compute kernel launch parameters
        self.local_size = (block_width, block_height,
                           1)  # WARNING::: MUST MATCH defines of block_width/height in kernels!
        self.global_size = (
            int(np.ceil((self.nx + 2) / float(self.local_size[0]))),
            int(np.ceil((self.ny + 3) / float(self.local_size[1]))))

        self.local_size_NS = (64, 4, 1)
        self.global_size_NS = (int(np.ceil((self.nx + 2) / float(self.local_size_NS[0]))), 1)

        self.local_size_EW = (2, 64, 1)
        self.global_size_EW = (1, int(np.ceil((self.ny + 3) / float(self.local_size_EW[1]))))

        # Load kernel for periodic boundary.
        self.boundaryKernel_NS = gpu_ctx.get_kernel("FBL_boundary_NS",
                                                    defines={'block_width': self.local_size_NS[0],
                                                             'block_height': self.local_size_NS[1]})
        self.boundaryKernel_EW = gpu_ctx.get_kernel("FBL_boundary_EW",
                                                    defines={'block_width': self.local_size_EW[0],
                                                             'block_height': self.local_size_EW[1]})

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.closedBoundaryUKernel_EW = GPUHandler(self.boundaryKernel_EW, "closedBoundaryUKernel_EW", "iiiiiiPi")
        self.closedBoundaryUKernel_NS = GPUHandler(self.boundaryKernel_NS, "closedBoundaryUKernel_NS", "iiiiiiPi")
        self.periodicBoundaryUKernel_NS = GPUHandler(self.boundaryKernel_NS, "periodicBoundaryUKernel_NS", "iiPi")
        self.periodicBoundaryUKernel_EW = GPUHandler(self.boundaryKernel_EW, "periodicBoundaryUKernel_EW", "iiPi")

        self.closedBoundaryVKernel_NS = GPUHandler(self.boundaryKernel_NS, "closedBoundaryVKernel_NS", "iiiiiiPi")
        self.closedBoundaryVKernel_EW = GPUHandler(self.boundaryKernel_EW, "closedBoundaryVKernel_EW", "iiiiiiPi")
        self.periodicBoundaryVKernel_NS = GPUHandler(self.boundaryKernel_NS, "periodicBoundaryVKernel_NS", "iiPi")
        self.periodicBoundaryVKernel_EW = GPUHandler(self.boundaryKernel_EW, "periodicBoundaryVKernel_EW", "iiPi")

        self.closedBoundaryEtaKernel_NS = GPUHandler(self.boundaryKernel_NS, "closedBoundaryEtaKernel_NS", "iiiiiiPi")
        self.closedBoundaryEtaKernel_EW = GPUHandler(self.boundaryKernel_EW, "closedBoundaryEtaKernel_EW", "iiiiiiPi")
        self.periodicBoundaryEtaKernel_NS = GPUHandler(self.boundaryKernel_NS, "periodicBoundaryEtaKernel_NS", "iiPi")
        self.periodicBoundaryEtaKernel_EW = GPUHandler(self.boundaryKernel_EW, "periodicBoundaryEtaKernel_EW", "iiPi")

        # Reuse CTCS kernels for Flow Relaxation Scheme
        self.CTCSBoundaryKernels = gpu_ctx.get_kernel("CTCS_boundary", defines={'block_width': block_width,
                                                                                'block_height': block_height})
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.boundary_flowRelaxationScheme_NS = GPUHandler(self.CTCSBoundaryKernels,
                                                           "boundary_flowRelaxationScheme_NS", "iiiiiiiiiiPi")
        self.boundary_flowRelaxationScheme_EW = GPUHandler(self.CTCSBoundaryKernels, "boundary_flowRelaxationScheme_EW",
                                                           "iiiiiiiiiiPi")

    def boundaryConditionU(self, gpu_stream: GPUStream, hu0: Array2D):
        """
        Updates hu according to boundary conditions
        """

        # Start with fixing the potential sponge
        self.callSpongeNS(gpu_stream, hu0, 0, 0, nx_offset=-2)

        if self.boundary_conditions.east == BoundaryType.WALL or \
                self.boundary_conditions.west == BoundaryType.WALL or \
                self.boundary_conditions.north == BoundaryType.WALL or \
                self.boundary_conditions.south == BoundaryType.WALL:
            # With closed boundaries:
            # West is written to zero by everytime step in the step-function
            # East should be set to zero once, and is then never written to.
            # North and south must be updated after each timestep to U[ghost] = U[inner].

            # At this point, we call the BC kernel at all boundaries, just to be safe.
            # OBS! We use east-west before north south!

            self.closedBoundaryUKernel_EW.async_call(*self._closed_args(gpu_stream, hu0, False, False))

            # Not strictly needed
            self.closedBoundaryUKernel_NS.async_call(*self._closed_args(gpu_stream, hu0, True, False))

        if self.boundary_conditions.north == BoundaryType.PERIODIC:
            self.periodicBoundaryUKernel_NS.async_call(*self._closed_args(gpu_stream, hu0, True, True))
        if self.boundary_conditions.east == BoundaryType.PERIODIC:
            self.periodicBoundaryUKernel_EW.async_call(*self._closed_args(gpu_stream, hu0, False, True))

    def boundaryConditionV(self, gpu_stream: GPUStream, hv0: Array2D):
        """
        Updates hv according to periodic boundary conditions
        """

        # Start with fixing the potential sponge
        self.callSpongeNS(gpu_stream, hv0, 0, 1)

        if self.boundary_conditions.east == BoundaryType.WALL or \
                self.boundary_conditions.west == BoundaryType.WALL or \
                self.boundary_conditions.north == BoundaryType.WALL or \
                self.boundary_conditions.south == BoundaryType.WALL:
            # With closed boundaries:
            # V south is written to zero by the step-function,
            # Ghost cells in north and south must be updated with V[outer]=-V[inner]
            # Ghost cells in east and west must be updated with V[outer]=V[inner]
            #
            # North-south before east west.

            self.closedBoundaryVKernel_NS.async_call(*self._closed_args(gpu_stream, hv0, True, False))

            # Not strictly needed.
            self.closedBoundaryVKernel_EW.async_call(*self._closed_args(gpu_stream, hv0, False, False))

        if self.boundary_conditions.north == BoundaryType.PERIODIC:
            self.periodicBoundaryVKernel_NS.async_call(*self._closed_args(gpu_stream, hv0, True, True))
        if self.boundary_conditions.east == BoundaryType.PERIODIC:
            self.periodicBoundaryVKernel_EW.async_call(*self._closed_args(gpu_stream, hv0, False, True))

    def boundaryConditionEta(self, gpu_stream: GPUStream, eta0: Array2D):
        """
        Updates eta boundary conditions (ghost cells)
        """
        # Start with fixing the potential sponge
        self.callSpongeNS(gpu_stream, eta0, 0, 0)

        if self.boundary_conditions.east == BoundaryType.WALL or \
                self.boundary_conditions.west == BoundaryType.WALL or \
                self.boundary_conditions.north == BoundaryType.WALL or \
                self.boundary_conditions.south == BoundaryType.WALL:
            self.closedBoundaryEtaKernel_NS.async_call(*self._closed_args(gpu_stream, eta0, True, False))
            self.closedBoundaryEtaKernel_EW.async_call(*self._closed_args(gpu_stream, eta0, False, False))

        if self.boundary_conditions.north == BoundaryType.PERIODIC:
            self.periodicBoundaryEtaKernel_NS.async_call(*self._closed_args(gpu_stream, eta0, True, True))
        if self.boundary_conditions.east == BoundaryType.PERIODIC:
            self.periodicBoundaryEtaKernel_EW.async_call(*self._closed_args(gpu_stream, eta0, False, True))

    def _closed_args(self, gpu_stream: GPUStream, data: Array2D, ns: bool, periodic: bool):
        global_size = self.global_size_NS if ns else self.global_size_EW
        local_size = self.local_size_NS if ns else self.local_size_EW

        args = [self.nx, self.ny]
        if not periodic:
            args.extend([self.bc_north.value, self.bc_east.value, self.bc_south.value, self.bc_west.value])
        args.extend([data.pointer, data.pitch])

        return global_size, local_size, gpu_stream, args

    def callSpongeNS(self, gpu_stream: GPUStream, data: Array2D, staggered_x: int, staggered_y: int,
                     nx_offset=0, ny_offset=0):
        if (self.bc_north == BoundaryType.FLOW_RELAXATION_SCHEME) or (
                self.bc_south == BoundaryType.FLOW_RELAXATION_SCHEME):
            self.boundary_flowRelaxationScheme_NS.async_call(
                self.global_size, self.local_size, gpu_stream,
                [self.nx + ny_offset, self.ny + ny_offset,
                 self.ghostsX, self.ghostsY,
                 staggered_x, staggered_y,
                 self.boundary_conditions.spongeCells.north,
                 self.boundary_conditions.spongeCells.south,
                 self.bc_north.value, self.bc_south.value,
                 data.pointer, data.pitch])

        if (self.bc_east == BoundaryType.FLOW_RELAXATION_SCHEME) or (
                self.bc_west == BoundaryType.FLOW_RELAXATION_SCHEME):
            self.boundary_flowRelaxationScheme_EW.async_call(
                self.global_size, self.local_size, gpu_stream,
                [self.nx + ny_offset, self.ny + ny_offset,
                 self.ghostsX, self.ghostsY,
                 staggered_x, staggered_y,
                 self.boundary_conditions.spongeCells.east,
                 self.boundary_conditions.spongeCells.west,
                 self.bc_east.value, self.bc_west.value,
                 data.pointer, data.pitch])
