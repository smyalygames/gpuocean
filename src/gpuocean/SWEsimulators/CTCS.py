# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the Centered in Time, Centered in Space
(leapfrog) numerical scheme for the shallow water equations, 
described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .


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

import numpy as np
import gc

from gpuocean.utils import SimWriter, SimReader, WindStress, AtmosphericPressure
from gpuocean.utils.Common import BoundaryConditions, BoundaryType
from gpuocean.SWEsimulators import Simulator
from gpuocean.utils.gpu import GPUHandler, Array2D, SWEDataArakawaC

if TYPE_CHECKING:
    import numpy.typing as npt
    from mpi4py import MPI

    from gpuocean.utils.gpu import KernelContext, GPUStream


class CTCS(Simulator.Simulator):
    """
    Class that solves the SW equations using the Centered in time centered in space scheme
    """

    def __init__(self,
                 gpu_ctx: KernelContext,
                 H: npt.NDArray, eta0: npt.NDArray, hu0: npt.NDArray, hv0: npt.NDArray,
                 nx: int, ny: int,
                 dx: float, dy: float, dt: float,
                 g: float, f: float, r: float, A=0.0,
                 t=0.0,
                 coriolis_beta=0.0,
                 y_zero_reference_cell=0,
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
            hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
            nx: Number of cells along x-axis
            ny: Number of cells along y-axis
            dx: Grid cell spacing along x-axis (20 000 m)
            dy: Grid cell spacing along y-axis (20 000 m)
            dt: Size of each timestep (90 s)
            g: Gravitational accelleration (9.81 m/s^2)
            f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
            r: Bottom friction coefficient (2.4e-3 m/s)
            A: Eddy viscosity coefficient (O(dx))
            t: Start simulation at time t
            coriolis_beta: Coriolis linear factor -> f = f + beta*(y-y_0)
            atmospheric_pressure: Object with values for atmospheric pressure
            y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell .
            wind: Wind stress parameters
            boundary_conditions: Boundary condition object
            write_netcdf: Write the results after each superstep to a netCDF file
            comm: MPI communicator
        """

        # Sort out internally represented ghost_cells in the presence of given
        # boundary conditions
        halo_x = 1
        halo_y = 1
        ghost_cells_x = 1
        ghost_cells_y = 1
        y_zero_reference_cell = y_zero_reference_cell + 1

        self.boundary_conditions = boundary_conditions
        if boundary_conditions.isSponge():
            nx = nx + int(boundary_conditions.spongeCells.east) + int(boundary_conditions.spongeCells.west) - 2 * ghost_cells_x
            ny = ny + int(boundary_conditions.spongeCells.north) + int(boundary_conditions.spongeCells.south) - 2 * ghost_cells_y
            y_zero_reference_cell = y_zero_reference_cell + int(boundary_conditions.spongeCells.south)

        # self.<parameters> are sat in parent constructor:
        rk_order = None
        theta = None
        super(CTCS, self).__init__(gpu_ctx,
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

        # Index range for interior domain (north, east, south, west)
        # so that interior domain of eta is
        # eta[self.interior_domain_indices[2]:self.interior_domain_indices[0], \
        #     self.interior_domain_indices[3]:self.interior_domain_indices[1] ]
        self.interior_domain_indices = np.array([-1, -1, 1, 1])
        self._set_interior_domain_from_sponge_cells()

        self.step_kernel = gpu_ctx.get_kernel("CTCS_step_kernel",
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

        # Get GPU functions
        self.ctcsStepKernel = GPUHandler(self.step_kernel, "ctcsStepKernel", "iiifffffffffPiPiPiPiPiPiPiPPPPf")

        # Set up textures
        self.update_wind_stress(self.step_kernel)

        # Create data by uploading to device     
        self.H = Array2D(self.gpu_stream, nx, ny, halo_x, halo_y, H)
        self.gpu_data = SWEDataArakawaC(self.gpu_stream, nx, ny, halo_x, halo_y, eta0, hu0, hv0)

        # Global size needs to be larger than the default from parent.__init__
        self.global_size = (
            int(np.ceil((self.nx + 2 * halo_x) / float(self.local_size[0]))),
            int(np.ceil((self.ny + 2 * halo_y) / float(self.local_size[1])))
        )

        self.bc_kernel = CTCS_boundary_condition(gpu_ctx,
                                                 self.nx,
                                                 self.ny,
                                                 self.boundary_conditions,
                                                 halo_x, halo_y
                                                 )

        # "Beautify" code a bit by packing four bools into a single int
        # Note: Must match code in kernel!
        self.wall_bc = 0
        if self.boundary_conditions.north == 1:
            self.wall_bc = self.wall_bc | 0x01
        if self.boundary_conditions.east == 1:
            self.wall_bc = self.wall_bc | 0x02
        if self.boundary_conditions.south == 1:
            self.wall_bc = self.wall_bc | 0x04
        if self.boundary_conditions.west == 1:
            self.wall_bc = self.wall_bc | 0x08

        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self, ignore_ghostcells=self.ignore_ghostcells,
                                                        staggered_grid=True, offset_x=self.offset_x,
                                                        offset_y=self.offset_y)

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

    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        n = int(t_end / self.dt + 1)
        if n % 2 == 0:
            n += 1

        if self.t == 0:
            # print "N: ", n
            # print "np.float(min(self.dt, t_end-n*self.dt))", np.float32(min(self.dt, t_end-(n-1)*self.dt))

            # Ensure that the boundary conditions are satisfied before starting simulation
            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)
            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h1)
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu1)
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv1)

        for i in range(0, n):
            # Notation: 
            # gpu_data.u0 => U^{n-1} before U kernel, U^{n+1} after U kernel
            # gpu_data.u1 => U^{n}
            # When we call gpu_data.swap(), we swap these, so that
            # gpu_data.u0 => U^{n}
            # gpu_data.u1 => U^{n+1} (U kernel has been executed)
            # Now we are ready for the next time step

            # Add 1% of final timestep to this one
            # This makes final timestep 99% as large as the others
            # making sure that the last timestep is not incredibly small
            local_dt = (t_end / n)
            local_dt = local_dt + (local_dt / (100 * n))
            local_dt = float(min(local_dt, t_end - i * local_dt))

            if local_dt <= 0.0:
                break

            wind_stress_t = float(self.update_wind_stress(self.step_kernel))

            self.ctcsStepKernel.async_call(self.global_size, self.local_size, self.gpu_stream,
                                           [self.nx, self.ny,
                                      self.wall_bc,
                                      self.dx, self.dy, local_dt,
                                      self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell,
                                      self.r, self.A,

                                      self.gpu_data.h0.pointer, self.gpu_data.h0.pitch,
                                      # eta^{n-1} => eta^{n+1} \
                                      self.gpu_data.hu0.pointer, self.gpu_data.hu0.pitch,
                                      # U^{n-1} => U^{n+1} \
                                      self.gpu_data.hv0.pointer, self.gpu_data.hv0.pitch,
                                      # V^{n-1} => V^{n+1} \

                                      self.H.pointer, self.H.pitch,  # H (bathymetry) \
                                      self.gpu_data.h1.pointer, self.gpu_data.h1.pitch,  # eta^{n} \
                                      self.gpu_data.hu1.pointer, self.gpu_data.hu1.pitch,  # U^{n} \
                                      self.gpu_data.hv1.pointer, self.gpu_data.hv1.pitch,  # V^{n} \

                                      self.wind_stress_x_current_arr.pointer,
                                      self.wind_stress_x_next_arr.pointer,
                                      self.wind_stress_y_current_arr.pointer,
                                      self.wind_stress_y_next_arr.pointer,
                                      wind_stress_t])

            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)

            # After the kernels, swap the data pointers
            self.gpu_data.swap()

            self.t += np.float64(local_dt)
            self.num_iterations += 1

        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)

        return self.t

    def _call_all_boundary_conditions(self):
        self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)
        self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
        self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)
        self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h1)
        self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu1)
        self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv1)


class CTCS_boundary_condition:
    def __init__(self, gpu_ctx: KernelContext, nx: int, ny: int,
                 boundary_conditions: BoundaryConditions, halo_x: int, halo_y: int,
                 block_width=16, block_height=16):

        self.boundary_conditions = boundary_conditions

        self.bc_north = boundary_conditions.north.value
        self.bc_east = boundary_conditions.east.value
        self.bc_south = boundary_conditions.south.value
        self.bc_west = boundary_conditions.west.value

        self.nx = nx
        self.ny = ny
        self.halo_x = halo_x
        self.halo_y = halo_y
        self.nx_halo = nx + 2 * halo_x
        self.ny_halo = ny + 2 * halo_y

        # Set kernel launch parameters
        self.local_size = (block_width, block_height, 1)
        self.global_size = (
            int(np.ceil((self.nx_halo + 1) / float(self.local_size[0]))),
            int(np.ceil((self.ny_halo + 1) / float(self.local_size[1]))))

        self.local_size_NS = (64, 4, 1)
        self.global_size_NS = (int(np.ceil((self.nx_halo + 1) / float(self.local_size_NS[0]))), 1)

        self.local_size_EW = (4, 64, 1)
        self.global_size_EW = (1, int(np.ceil((self.ny_halo + 1) / float(self.local_size_EW[1]))))

        # Load kernel for periodic boundary
        self.boundaryKernels = gpu_ctx.get_kernel("CTCS_boundary", defines={'block_width': block_width,
                                                                            'block_height': block_height})
        self.boundaryKernels_NS = gpu_ctx.get_kernel("CTCS_boundary_NS", defines={'block_width': self.local_size_NS[0],
                                                                                  'block_height': self.local_size_NS[
                                                                                      1]})
        self.boundaryKernels_EW = gpu_ctx.get_kernel("CTCS_boundary_EW", defines={'block_width': self.local_size_EW[0],
                                                                                  'block_height': self.local_size_EW[
                                                                                      1]})

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.boundaryUKernel_NS = GPUHandler(self.boundaryKernels_NS, "boundaryUKernel_NS", "iiiiiiPi")
        self.boundaryUKernel_EW = GPUHandler(self.boundaryKernels_EW, "boundaryUKernel_EW", "iiiiiiPi")
        self.boundaryVKernel_NS = GPUHandler(self.boundaryKernels_NS, "boundaryVKernel_NS", "iiiiiiPi")
        self.boundaryVKernel_EW = GPUHandler(self.boundaryKernels_EW, "boundaryVKernel_EW", "iiiiiiPi")
        self.boundaryEtaKernel_NS = GPUHandler(self.boundaryKernels_NS, "boundaryEtaKernel_NS", "iiiiiiPi")
        self.boundaryEtaKernel_EW = GPUHandler(self.boundaryKernels_EW, "boundaryEtaKernel_EW", "iiiiiiPi")
        self.boundary_linearInterpol_NS = GPUHandler(self.boundaryKernels, "boundary_linearInterpol_NS", "iiiiiiiiiiPi")
        self.boundary_linearInterpol_EW = GPUHandler(self.boundaryKernels, "boundary_linearInterpol_EW", "iiiiiiiiiiPi")
        self.boundary_flowRelaxationScheme_NS = GPUHandler(self.boundaryKernels, "boundary_flowRelaxationScheme_NS",
                                                           "iiiiiiiiiiPi")
        self.boundary_flowRelaxationScheme_EW = GPUHandler(self.boundaryKernels, "boundary_flowRelaxationScheme_EW",
                                                           "iiiiiiiiiiPi")

    def boundaryConditionU(self, gpu_stream: GPUStream, hu0: Array2D):
        """
        Updates hu according periodic boundary conditions
        """

        if (self.bc_north < 3) or (self.bc_south < 3):
            self.boundaryUKernel_NS.async_call(
                self.global_size_NS, self.local_size_NS, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 self.bc_north, self.bc_south,
                 hu0.pointer, hu0.pitch])
        # self.callSpongeNS(gpu_stream, hu0, 0, 0)
        self.callSpongeNS(gpu_stream, hu0, 1, 0)

        if (self.bc_east < 3) or (self.bc_west < 3):
            self.boundaryUKernel_EW.async_call(
                self.global_size_EW, self.local_size_EW, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 self.bc_east, self.bc_west,
                 hu0.pointer, hu0.pitch])
        self.callSpongeEW(gpu_stream, hu0, 1, 0)
        # self.callSpongeEW(gpu_stream, hu0, 0, 0)

    def boundaryConditionV(self, gpu_stream: GPUStream, hv0: Array2D):
        """
        Updates hv according to periodic boundary conditions
        """

        if (self.bc_north < 3) or (self.bc_south < 3):
            self.boundaryVKernel_NS.async_call(
                self.global_size_NS, self.local_size_NS, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 self.bc_north, self.bc_south,
                 hv0.pointer, hv0.pitch])
        self.callSpongeNS(gpu_stream, hv0, 0, 1)
        # self.callSpongeNS(gpu_stream, hv0, 0, 0)

        if (self.bc_east < 3) or (self.bc_west < 3):
            self.boundaryVKernel_EW.async_call(
                self.global_size_EW, self.local_size_EW, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 self.bc_east, self.bc_west,
                 hv0.pointer, hv0.pitch])
        self.callSpongeEW(gpu_stream, hv0, 0, 1)
        # self.callSpongeEW(gpu_stream, hv0, 0, 0)

    def boundaryConditionEta(self, gpu_stream: GPUStream, eta0: Array2D):
        """
        Updates eta boundary conditions (ghost cells)
        """

        if (self.bc_north < 3) or (self.bc_south < 3):
            self.boundaryEtaKernel_NS.async_call(
                self.global_size_NS, self.local_size_NS, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 self.bc_north, self.bc_south,
                 eta0.pointer, eta0.pitch])
        self.callSpongeNS(gpu_stream, eta0, 0, 0)

        if (self.bc_east < 3) or (self.bc_west < 3):
            self.boundaryEtaKernel_EW.async_call(
                self.global_size_EW, self.local_size_EW, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 self.bc_east, self.bc_west,
                 eta0.pointer, eta0.pitch])
        self.callSpongeEW(gpu_stream, eta0, 0, 0)

    def callSpongeNS(self, gpu_stream: GPUStream, data: Array2D, staggered_x: int, staggered_y: int):
        """
        Call othe approporary sponge-like boundary condition with the given data
        """

        # print "callSpongeNS"
        if (self.bc_north == 3) or (self.bc_south == 3):
            self.boundary_flowRelaxationScheme_NS.async_call(
                self.global_size, self.local_size, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 staggered_x, staggered_y,
                 self.boundary_conditions.spongeCells.north,
                 self.boundary_conditions.spongeCells.south,
                 self.bc_north, self.bc_south,
                 data.pointer, data.pitch])
        if (self.bc_north == 4) or (self.bc_south == 4):
            self.boundary_linearInterpol_NS.async_call(
                self.global_size, self.local_size, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 staggered_x, staggered_y,
                 self.boundary_conditions.spongeCells.north,
                 self.boundary_conditions.spongeCells.south,
                 self.bc_north, self.bc_south,
                 data.pointer, data.pitch])

    def callSpongeEW(self, gpu_stream: GPUStream, data: Array2D, staggered_x: int, staggered_y: int):

        # print "CallSpongeEW"
        if (self.bc_east == 3) or (self.bc_west == 3):
            self.boundary_flowRelaxationScheme_EW.async_call(
                self.global_size, self.local_size, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 staggered_x, staggered_y,
                 self.boundary_conditions.spongeCells.east,
                 self.boundary_conditions.spongeCells.west,
                 self.bc_east, self.bc_west,
                 data.pointer, data.pitch])

        if (self.bc_east == 4) or (self.bc_west == 4):
            self.boundary_linearInterpol_EW.async_call(
                self.global_size, self.local_size, gpu_stream,
                [self.nx, self.ny,
                 self.halo_x, self.halo_y,
                 staggered_x, staggered_y,
                 self.boundary_conditions.spongeCells.east,
                 self.boundary_conditions.spongeCells.west,
                 self.bc_east, self.bc_west,
                 data.pointer, data.pitch])
