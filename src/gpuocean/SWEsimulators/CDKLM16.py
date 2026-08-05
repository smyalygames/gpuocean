# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the finite-volume scheme proposed by
Alina Chertock, Michael Dudzinski, A. Kurganov & Maria Lukacova-Medvidova (2016)
Well-Balanced Schemes for the Shallow Water Equations with Coriolis Forces

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
from typing import TYPE_CHECKING, Iterable
import gc
import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from gpuocean.utils import WindStress, AtmosphericPressure, OceanographicUtilities
from gpuocean.utils.netcdf import SimNetCDFWriter, SimNetCDFReader
from gpuocean.utils.Common import BoundaryConditions, BoundaryConditionsData
from gpuocean.utils.gpu.arrays.bathymetry import Bathymetry
from gpuocean.SWEsimulators import Simulator, OceanStateNoise, ModelErrorKL
from gpuocean.utils.gpu import Array2D, GPUHandler, SWEDataArakawaA, BoundaryConditionsArakawaA

if TYPE_CHECKING:
    from mpi4py import MPI

    from gpuocean.utils.gpu import KernelContext
    from gpuocean.utils.dataclass import GhostCells


class CDKLM16(Simulator.Simulator):
    """
    Class that solves the SW equations using the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
    """

    def __init__(self,
                 gpu_ctx: KernelContext,
                 eta0: npt.NDArray[np.float32 | np.float64],
                 hu0: npt.NDArray[np.float32 | np.float64], hv0: npt.NDArray[np.float32 | np.float64],
                 H: npt.NDArray[np.float32 | np.float64],
                 nx: int, ny: int,
                 dx: float, dy: float, dt: float,
                 g: float, f: float, r: float,
                 subsample_f=10,
                 angle=np.array([[0]], dtype=np.float32),
                 subsample_angle=10,
                 latitude: float | npt.NDArray = None,
                 rho_o=1025.0,
                 t=0.0,
                 theta=1.8, rk_order=2,
                 coriolis_beta=0.0,
                 max_wind_direction_perturbation=0,
                 wind=WindStress.WindStress(),
                 wind_stress_factor=1.0,
                 atmospheric_pressure=AtmosphericPressure.AtmosphericPressure(),
                 boundary_conditions=BoundaryConditions(),
                 boundary_conditions_data=BoundaryConditionsData(),
                 model_time_step: float = None,
                 reportGeostrophicEquilibrium=False,
                 write_netcdf=False,
                 comm: MPI.Intracomm = None,
                 local_particle_id=0,
                 super_dir_name: str = None,
                 netcdf_filename: str = None,
                 ignore_ghostcells=False,
                 courant_number=0.8,
                 offset_x=0, offset_y=0,
                 flux_slope_eps=1.0e-1,
                 desingularization_eps=1.0e-1,
                 depth_cutoff=1.0e-5,
                 one_dimensional=False,
                 flux_balancer=0.8,
                 block_width=12, block_height=32, num_threads_dt=256,
                 use_direct_lookup=False,
                 compile_opts: list[str]=None):
        """
        Initialization routine
        Args:
            eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
            hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
            hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
            H: Depth from equilibrium defined on cell corners, (nx+5)*(ny+5) corners
            nx: Number of cells along x-axis
            ny: Number of cells along y-axis
            dx: Grid cell spacing along x-axis (20 000 m)
            dy: Grid cell spacing along y-axis (20 000 m)
            dt: Size of each timestep (90 s)
            g: Gravitational accelleration (9.81 m/s^2)
            f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
            r: Bottom friction coefficient (2.4e-3 m/s)
            subsample_f: Subsample the coriolis f when creating texture by factor
            angle: Angle of rotation from North to y-axis as a texture (cuda.Array) or numpy array (in radians)
            subsample_angle: Subsample the angles given as input when creating texture by factor
            latitude: Specify latitude. This will override any f and beta plane already set (in radians)
            t: Start simulation at time t
            theta: MINMOD theta used the reconstructions of the derivatives in the numerical scheme
            rk_order: Order of Runge Kutta method {1,2*,3}
            coriolis_beta: Coriolis linear factor -> f = f + beta*(y-y_0)
            max_wind_direction_perturbation: Large-scale model error emulation by per-time-step perturbation of wind direction by +/- max_wind_direction_perturbation (degrees)
            wind: Wind stress parameters
            wind_stress_factor: artificial scaling of the wind stress acting on the water column. Won't affect drifters.
            atmospheric_pressure: Object with values for atmospheric pressure
            boundary_conditions: Boundary condition object
            model_time_step: The size of a data assimilation model step (default same as dt)
            reportGeostrophicEquilibrium: Calculate the Geostrophic Equilibrium variables for each superstep
            write_netcdf: Write the results after each superstep to a netCDF file
            comm: MPI communicator
            local_particle_id: Local (for each MPI process) particle id
            desingularization_eps: Used for desingularizing hu/h
            flux_slope_eps: Used for setting zero flux for symmetric Riemann fan
            flux_balancer: linear combination of upwind flux (value 1.0) and central-upwind flux (value 0.0)
            depth_cutoff: Used for defining dry cells
            super_dir_name: Directory to write netcdf files to
            netcdf_filename: Use this filename. (If not defined, a filename will be generated by SimWriter.)
            use_direct_lookup: Directly lookup the values in the data array rather than interpolate. Only valid for same sized arrays.
        """
        self.logger = logging.getLogger(__name__)

        if 4 <= rk_order <= 0:
            raise TypeError("Only 1st, 2nd and 3rd order Runge Kutta supported")
        if rk_order == 3 and r != 0.0:
            raise ValueError("3rd order Runge Kutta supported only without friction")
        if use_direct_lookup and (
                atmospheric_pressure.P[0].shape[0] != ny + 4 or atmospheric_pressure.P[0].shape[1] != nx + 4):
            raise TypeError("Direct lookup is only supported for same sized grids")

        # Sort out internally represented ghost_cells in the presence of given
        # boundary conditions
        ghost_cells_x = 2
        ghost_cells_y = 2

        # Coriolis at "first" cell
        x_zero_reference_cell = ghost_cells_x
        y_zero_reference_cell = ghost_cells_y  # In order to pass it to the super constructor

        # Boundary conditions
        self.boundary_conditions = boundary_conditions

        # Compensate f for reference cell (first cell in internal of domain)
        north = np.array([np.sin(angle[0, 0]), np.cos(angle[0, 0])])
        f = f - coriolis_beta * (
                x_zero_reference_cell * dx * north[0] + y_zero_reference_cell * dy * north[1])

        x_zero_reference_cell = 0
        y_zero_reference_cell = 0

        A = None
        self.max_wind_direction_perturbation = max_wind_direction_perturbation
        super(CDKLM16, self).__init__(gpu_ctx,
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
                                      block_width, block_height,
                                      local_particle_id=local_particle_id)

        # Index range for interior domain (north, east, south, west)
        # so that interior domain of eta is
        # eta[self.interior_domain_indices[2]:self.interior_domain_indices[0], \
        #     self.interior_domain_indices[3]:self.interior_domain_indices[1] ]
        self.interior_domain_indices = np.array([-2, -2, 2, 2])

        def subsample_texture(data: npt.NDArray, factor: int):
            ny, nx = data.shape
            dx, dy = 1 / nx, 1 / ny

            x_old = np.linspace(0.5 * dx, 1 - 0.5 * dx, nx)
            y_old = np.linspace(0.5 * dy, 1 - 0.5 * dy, ny)
            I = RegularGridInterpolator((y_old, x_old), data, method='linear')

            new_nx, new_ny = max(2, nx // factor), max(2, ny // factor)
            new_dx, new_dy = 1 / new_nx, 1 / new_ny

            x_new = np.linspace(0.5 * new_dx, 1 - 0.5 * new_dx, new_nx)
            y_new = np.linspace(0.5 * new_dy, 1 - 0.5 * new_dy, new_ny)

            mx_new, my_new = np.meshgrid(x_new, y_new)
            return I((my_new, mx_new))

        # Create the CPU coriolis
        if latitude is not None:
            if self.f != 0.0:
                raise RuntimeError("Cannot specify both latitude and f. Make your mind up.")
            coriolis_f, _ = OceanographicUtilities.calcCoriolisParams(latitude)
            coriolis_f = coriolis_f.astype(np.float32)
        else:
            if self.coriolis_beta != 0.0:
                if angle.size != 1:
                    raise RuntimeError("non-constant angle cannot be combined with beta plane model (makes no sense)")
                # Generate coordinates for all cells, including ghost cells from center to center
                # [-3/2dx, nx+3/2dx] for ghost_cells_x == 2
                x = np.linspace((-self.ghost_cells_x + 0.5) * self.dx, (self.nx + self.ghost_cells_x - 0.5) * self.dx,
                                self.nx + 2 * self.ghost_cells_x)
                y = np.linspace((-self.ghost_cells_y + 0.5) * self.dy, (self.ny + self.ghost_cells_y - 0.5) * self.dy,
                                self.ny + 2 * self.ghost_cells_x)
                self.logger.info(f"Using latitude to create Coriolis f texture ({x.size}x{y.size} cells)")
                x, y = np.meshgrid(x, y)
                n = x * np.sin(angle[0, 0]) + y * np.cos(angle[0, 0])  # North vector
                coriolis_f = self.f + self.coriolis_beta * n
            else:
                if self.f.size == 1:
                    coriolis_f = np.array([[self.f]], dtype=np.float32, order='C')
                elif self.f.shape == eta0.shape:
                    coriolis_f = np.array(self.f, dtype=np.float32, order='C')
                else:
                    raise RuntimeError("The shape of f should match up with eta or be scalar.")

        if subsample_f and coriolis_f.size > eta0.size:
            self.logger.info("Subsampling coriolis texture by factor " + str(subsample_f))
            self.logger.warning("This will give inaccurate coriolis along the border"
                                " as coriolis array is of size %d and eta0 is of size %d!",
                                coriolis_f.size, eta0.size)
            coriolis_f = subsample_texture(coriolis_f, subsample_f)

        # Initialize coriolis force GPU array
        self.coriolis_f_arr = Array2D(self.gpu_stream,
                                      coriolis_f.shape[1], coriolis_f.shape[0], 0, 0,
                                      coriolis_f, padded=False)

        # Subsample angle
        if subsample_angle and angle.size > eta0.size:
            self.logger.info("Subsampling angle texture by factor " + str(subsample_angle))
            self.logger.warning("This will give inaccurate angle along the border!")
            angle = subsample_texture(angle, subsample_angle)

        # Initialize angle GPU array
        self.angle_arr = Array2D(self.gpu_stream,
                                 angle.shape[1], angle.shape[0], 0, 0,
                                 angle)

        defines = CDKLMDefines(block_width, block_height,
                               desingularization_eps, flux_slope_eps, depth_cutoff,
                               self.theta,
                               self.rk_order,
                               self.nx, self.ny,
                               self.dx, self.dy,
                               self.g,
                               self.r,
                               rho_o,
                               wind_stress_factor,
                               one_dimensional,
                               flux_balancer,
                               coriolis_f,
                               angle,
                               self.atmospheric_pressure,
                               self.wind_stress,
                               use_direct_lookup,
                               self.ghost_cells
                               )

        if compile_opts is None:
            compile_opts = []

        compile_args = {  # default, fast_math, optimal
            'cuda': {
                'options': ["--ftz=true",  # false,   true,      true
                        "--prec-div=false",  # true,    false,     false,
                        "--prec-sqrt=false",  # true,    false,     false
                        "--fmad=false"] + compile_opts,  # true,    true,      false

                # 'options': ["--use_fast_math"]
                # 'options': ["--generate-line-info"],
                # nvcc_options=["--maxrregcount=39"],
                # 'arch': "compute_50",
                # 'code': "sm_50"
            },
            'hip': compile_opts
        }

        jit_compile_args = {
            # jit_options=[(cuda.jit_option.MAX_REGISTERS, 39)]
        }

        # Get kernels
        self.kernel = gpu_ctx.get_kernel("CDKLM16_kernel",
                                         defines=defines.default,
                                         compile_args=compile_args,
                                         jit_compile_args=jit_compile_args
                                         )

        self.kernel_ns = gpu_ctx.get_kernel("CDKLM16_kernel",
                                            defines=defines.north_south,
                                            compile_args=compile_args,
                                            jit_compile_args=jit_compile_args)

        self.kernel_ew = gpu_ctx.get_kernel("CDKLM16_kernel",
                                            defines=defines.east_west,
                                            compile_args=compile_args,
                                            jit_compile_args=jit_compile_args)

        self.kernel_inner = gpu_ctx.get_kernel("CDKLM16_kernel",
                                               defines=defines.inner,
                                               compile_args=compile_args,
                                               jit_compile_args=jit_compile_args)

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.cdklm_swe_2D = GPUHandler(self.kernel, "cdklm_swe_2D", "fiPiPiPiPiPiPiPiPifPPPPfPPPPfi")
        self.cdklm_swe_2D_nw = GPUHandler(self.kernel_ns, "cdklm_swe_2D", "fiPiPiPiPiPiPiPiPifPPPPfPPPPfi")
        self.cdklm_swe_2D_ew = GPUHandler(self.kernel_ew, "cdklm_swe_2D", "fiPiPiPiPiPiPiPiPifPPPPfPPPPfi")
        self.cdklm_swe_2D_inner = GPUHandler(self.kernel_inner, "cdklm_swe_2D", "fiPiPiPiPiPiPiPiPifPPPPfPPPPfi")
        self.update_wind_stress(self.kernel)
        self.update_atmospheric_pressure(self.kernel)

        # CUDA functions for finding max time step size:
        self.num_threads_dt = num_threads_dt
        self.num_blocks_dt = self.global_size[0] * self.global_size[1]
        self.update_dt_kernels = gpu_ctx.get_kernel("max_dt",
                                                    defines={'block_width': block_width,
                                                             'block_height': block_height,
                                                             'NUM_THREADS': self.num_threads_dt})
        self.per_block_max_dt_kernel = GPUHandler(self.update_dt_kernels, "per_block_max_dt", "iifffPiPiPiPifPi")
        self.max_dt_reduction_kernel = GPUHandler(self.update_dt_kernels, "max_dt_reduction", "iPP")

        # Bathymetry
        self.bathymetry = Bathymetry(gpu_ctx, self.gpu_stream, nx, ny,
                                     ghost_cells_x, ghost_cells_y, H,
                                     boundary_conditions)

        # Adjust eta for possible dry states
        Hm = self.downloadBathymetry()[1]
        eta0 = np.maximum(eta0, -Hm)

        # Create data by uploading to device
        self.gpu_data = SWEDataArakawaA(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0)

        # Allocate memory for calculating maximum timestep
        host_dt = np.zeros((self.global_size[1], self.global_size[0]), dtype=np.float32)
        self.device_dt = Array2D(self.gpu_stream, self.global_size[0], self.global_size[1],
                                 0, 0, host_dt, padded=False)
        host_max_dt_buffer = np.zeros((1, 1), dtype=np.float32)
        self.max_dt_buffer = Array2D(self.gpu_stream, 1, 1, 0, 0, host_max_dt_buffer)
        self.courant_number = courant_number

        ## Allocating memory for geostrophical equilibrium variables
        self.reportGeostrophicEquilibrium = int(reportGeostrophicEquilibrium)
        self.geoEq_uxpvy = None
        self.geoEq_Kx = None
        self.geoEq_Ly = None
        if self.reportGeostrophicEquilibrium:
            dummy_zero_array = np.zeros((ny + 2 * ghost_cells_y, nx + 2 * ghost_cells_x), dtype=np.float32, order='C')
            self.geoEq_uxpvy = Array2D(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)
            self.geoEq_Kx = Array2D(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)
            self.geoEq_Ly = Array2D(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)

        self.constant_equilibrium_depth = np.max(H)

        self.bc_kernel = BoundaryConditionsArakawaA(gpu_ctx,
                                                    self.gpu_stream,
                                                    self.nx,
                                                    self.ny,
                                                    ghost_cells_x,
                                                    ghost_cells_y,
                                                    self.boundary_conditions,
                                                    boundary_conditions_data,
                                                    )

        # Small scale perturbation:
        self.model_error = None

        # Data assimilation model step size
        self.model_time_step = model_time_step
        self.total_time_steps = 0
        if model_time_step is None:
            self.model_time_step = self.dt

        if self.write_netcdf:
            self.sim_writer = SimNetCDFWriter(self, super_dir=super_dir_name, filename=netcdf_filename,
                                              ignore_ghostcells=self.ignore_ghostcells,
                                              offset_x=self.offset_x, offset_y=self.offset_y)

        # Update timestep if dt is given as zero
        if self.dt <= 0:
            self.updateDt()

    @property
    def arrays(self) -> list[Array2D]:
        array = super().arrays

        if self.reportGeostrophicEquilibrium:
            array += [
                self.geoEq_uxpvy,
                self.geoEq_Kx,
                self.geoEq_Ly
            ]

        return array

    def cleanUp(self, do_gc=True):
        """
        Clean up function
        """
        self.closeNetCDF()

        self.gpu_data.release()

        if self.model_error is not None:
            self.model_error.cleanUp(do_gc=do_gc)

        if self.geoEq_uxpvy is not None:
            self.geoEq_uxpvy.release()
        if self.geoEq_Kx is not None:
            self.geoEq_Kx.release()
        if self.geoEq_Ly is not None:
            self.geoEq_Ly.release()
        self.bathymetry.release()

        self.device_dt.release()
        self.max_dt_buffer.release()

        self.gpu_ctx = None
        if do_gc:
            gc.collect()

    @classmethod
    def fromfilename(cls, gpu_ctx: KernelContext, filename: str, cont_write_netcdf=True,
                     new_netcdf_filename: str = None, time0: float = None):
        """
        Initialize and hotstart simulation from nc-file.
        cont_write_netcdf: Continue to write the results after each superstep to a new netCDF file
        filename: Continue simulation based on parameters and last timestep in this file
        new_netcdf_filename: If we want to continue to write netcdf, we should use this filename. Automatically generated if None.
        """
        # open nc-file
        sim_reader = SimNetCDFReader(filename, ignore_ghostcells=False)
        sim_name = str(sim_reader.get('simulator_short'))
        assert sim_name == cls.__name__, \
            "Trying to initialize a " + \
            cls.__name__ + " simulator with netCDF file based on " \
            + sim_name + " results."

        # read the most recent state 
        H = sim_reader.getH()

        # get last timestep (including simulation time of last timestep)
        if time0 is None:
            eta0, hu0, hv0, time0 = sim_reader.getLastTimeStep()
        else:
            eta0, hu0, hv0, time0 = sim_reader.getStateAtTime(time0)

        # For some reason, some old netcdf had 3-dimensional bathymetry.
        # This fix ensures that we only use a valid H
        if len(H.shape) == 3:
            print("norm diff H: ", np.linalg.norm(H[0, :, :] - H[1, :, :]))
            H = H[0, :, :]

        # Set simulation parameters
        sim_params = {
            'gpu_ctx': gpu_ctx,
            'eta0': eta0,
            'hu0': hu0,
            'hv0': hv0,
            'H': H,
            'nx': sim_reader.get("nx"),
            'ny': sim_reader.get("ny"),
            'dx': sim_reader.get("dx"),
            'dy': sim_reader.get("dy"),
            'dt': sim_reader.get("dt"),
            'g': sim_reader.get("g"),
            'f': sim_reader.get("coriolis_force"),
            'r': sim_reader.get("bottom_friction_r"),
            't': float(time0),
            'theta': sim_reader.get("minmod_theta"),
            'rk_order': sim_reader.get("time_integrator"),
            'coriolis_beta': sim_reader.get("coriolis_beta"),
            # 'y_zero_reference_cell': sim_reader.get("y_zero_reference_cell"), # TODO - UPDATE WITH NEW API
            'write_netcdf': cont_write_netcdf,
            # 'use_lcg': use_lcg,
            # 'xorwow_seed' : xorwow_seed,
            'netcdf_filename': new_netcdf_filename
        }

        # Wind stress
        wind = WindStress.WindStress()
        sim_params['wind'] = wind

        # Boundary conditions
        sim_params['boundary_conditions'] = sim_reader.getBC()

        # Data assimilation parameters:
        if sim_reader.has('model_time_step'):
            sim_params['model_time_step'] = sim_reader.get('model_time_step')

        return cls(**sim_params)

    def step(self, t_end=0.0, apply_stochastic_term=True, write_now=True, update_dt=False, split_step=False):
        """
        Function which steps n timesteps.
        apply_stochastic_term: Boolean value for whether the stochastic
            perturbation (if any) should be applied after every simulation time step
            by adding SOAR-generated random fields using OceanNoiseState.perturbSim(...)
        """

        if self.t == 0:
            self.bc_kernel.update_bc_values(self.gpu_stream, self.t)
            self.bc_kernel.boundaryCondition(self.gpu_stream,
                                             self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)

        t_now = 0.0

        while t_now < t_end:
            # Calculate dt if using automatic dt
            if update_dt:
                self.updateDt()
            local_dt = min(self.dt, float(t_end - t_now))

            wind_stress_t = self.update_wind_stress(self.kernel)
            atmospheric_pressure_t = self.update_atmospheric_pressure(self.kernel)
            self.bc_kernel.update_bc_values(self.gpu_stream, self.t)

            # 2nd order Runge Kutta
            if self.rk_order == 2:

                self.callKernel(self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0,
                                self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1,
                                local_dt, wind_stress_t, atmospheric_pressure_t, 0, split_step)

                self.bc_kernel.boundaryCondition(self.gpu_stream,
                                                 self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)

                self.callKernel(self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1,
                                self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0,
                                local_dt, wind_stress_t, atmospheric_pressure_t, 1, split_step)

                # Applying final boundary conditions after perturbation (if applicable)

            elif self.rk_order == 1:
                self.callKernel(self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0,
                                self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1,
                                local_dt, wind_stress_t, atmospheric_pressure_t, 0, split_step)

                self.gpu_data.swap()

                # Applying boundary conditions after perturbation (if applicable)

            # 3rd order RK method:
            elif self.rk_order == 3:

                self.callKernel(self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0,
                                self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1,
                                local_dt, wind_stress_t, atmospheric_pressure_t, 0, split_step)

                self.bc_kernel.boundaryCondition(self.gpu_stream,
                                                 self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)

                self.callKernel(self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1,
                                self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0,
                                local_dt, wind_stress_t, atmospheric_pressure_t, 1, split_step)

                self.bc_kernel.boundaryCondition(self.gpu_stream,
                                                 self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)

                self.callKernel(self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1,
                                self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0,
                                local_dt, wind_stress_t, atmospheric_pressure_t, 2, split_step)

                # Applying final boundary conditions after perturbation (if applicable)

            # Perturb ocean state with model error
            if self.model_error is not None and apply_stochastic_term:
                self.perturbState()

            # Apply boundary conditions
            self.bc_kernel.boundaryCondition(self.gpu_stream,
                                             self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)

            # Evolve drifters
            self.drifterStep(local_dt)

            self.t += local_dt
            t_now += local_dt
            self.num_iterations += 1

        if self.write_netcdf and write_now:
            self.writeState()

        return self.t

    def setSOARModelError(self, small_scale_perturbation_amplitude=None,
                          small_scale_perturbation_interpolation_factor=1,
                          use_lcg=False, xorwow_seed=None,
                          block_width_model_error=16, block_height_model_error=16):
        self.model_error = OceanStateNoise.OceanStateNoise.fromsim(self,
                                                                   soar_q0=small_scale_perturbation_amplitude,
                                                                   interpolation_factor=small_scale_perturbation_interpolation_factor,
                                                                   use_lcg=use_lcg, xorwow_seed=xorwow_seed,
                                                                   block_width=block_width_model_error,
                                                                   block_height=block_height_model_error)

        if self.write_netcdf:
            self.sim_writer.write_model_error(self)

    def setModelErrorFromFile(self, filename, use_lcg=False, xorwow_seed=None):
        """
        Initialize model error according to attributes in a netcdf file .
        filename: NetCDF file that has been written from a CDKLM simulator
        """
        # open nc-file
        sim_reader = SimNetCDFReader(filename, ignore_ghostcells=False)
        sim_name = str(sim_reader.get('simulator_short'))
        assert sim_name == self.__class__.__name__, \
            "Trying to initialize a " + \
            self.__class__.__name__ + " simulator with netCDF file based on " \
            + sim_name + " results."

        model_error_args = {}

        if sim_reader.has('small_scale_perturbation'):
            # For some backward compatibility (old version of SimWriter)
            if sim_reader.get('small_scale_perturbation') == 'True':
                model_error_args['small_scale_perturbation_amplitude'] = sim_reader.get(
                    'small_scale_perturbation_amplitude')
                model_error_args['small_scale_perturbation_interpolation_factor'] = sim_reader.get(
                    'small_scale_perturbation_interpolation_factor')
                self.setSOARModelError(**model_error_args, use_lcg=use_lcg, xorwow_seed=xorwow_seed)
        elif sim_reader.has('has_model_error'):
            # New version of SimWriter
            if sim_reader.get('has_model_error') == "True":
                if sim_reader.get('model_error_name') == "OceanStateNoise":
                    model_error_args['small_scale_perturbation_amplitude'] = sim_reader.get(
                        'small_scale_perturbation_amplitude')
                    model_error_args['small_scale_perturbation_interpolation_factor'] = sim_reader.get(
                        'small_scale_perturbation_interpolation_factor')
                    self.setSOARModelError(**model_error_args, use_lcg=use_lcg, xorwow_seed=xorwow_seed)
                elif sim_reader.get('model_error_name') == "ModelErrorKL":
                    model_error_args['kl_decay'] = sim_reader.get('kl_decay')
                    model_error_args['kl_scaling'] = sim_reader.get('kl_scaling')
                    model_error_args['include_cos'] = sim_reader.get('include_cos')
                    model_error_args['include_sin'] = sim_reader.get('include_sin')
                    model_error_args['basis_x_start'] = sim_reader.get('basis_x_start')
                    model_error_args['basis_y_start'] = sim_reader.get('basis_y_start')
                    model_error_args['basis_x_end'] = sim_reader.get('basis_x_end')
                    model_error_args['basis_y_end'] = sim_reader.get('basis_y_end')
                    self.setKLModelError(**model_error_args, use_lcg=use_lcg, xorwow_seed=xorwow_seed)

    def setKLModelError(self, kl_decay=1.2, kl_scaling=1.0,
                        include_cos=True, include_sin=True,
                        basis_x_start=1, basis_y_start=1,
                        basis_x_end=10, basis_y_end=10,
                        use_lcg=False, xorwow_seed=None, np_seed=None,
                        block_width=16, block_height=16):
        self.model_error = ModelErrorKL.ModelErrorKL.fromsim(self,
                                                             kl_decay=kl_decay, kl_scaling=kl_scaling,
                                                             include_cos=include_cos, include_sin=include_sin,
                                                             basis_x_start=basis_x_start, basis_y_start=basis_y_start,
                                                             basis_x_end=basis_x_end, basis_y_end=basis_y_end,
                                                             use_lcg=use_lcg, xorwow_seed=xorwow_seed, np_seed=np_seed,
                                                             block_width=block_width, block_height=block_height)

    def setKLModelErrorSimilarAs(self, otherSim):
        self.model_error = ModelErrorKL.ModelErrorKL.fromsim(self,
                                                             kl_decay=otherSim.model_error.kl_decay,
                                                             kl_scaling=otherSim.model_error.kl_scaling,
                                                             include_cos=otherSim.model_error.include_cos,
                                                             include_sin=otherSim.model_error.include_sin,
                                                             basis_x_start=otherSim.model_error.basis_x_start,
                                                             basis_y_start=otherSim.model_error.basis_y_start,
                                                             basis_x_end=otherSim.model_error.basis_x_end,
                                                             basis_y_end=otherSim.model_error.basis_y_end,
                                                             use_lcg=otherSim.model_error.use_lcg,
                                                             block_width=otherSim.model_error.local_size[0],
                                                             block_height=otherSim.model_error.local_size[1])

    def drifterStep(self, dt: float) -> float | None:
        # Evolve drifters
        if self.hasCrossProductDrifter:
            for d in range(len(self.CrossProductDrifter)):
                if self.CPsims[d] is not None:
                    self.CrossProductDrifter[d].drift(self.CPsims[d].gpu_data.h0,
                                                      self.CPsims[d].gpu_data.hu0,
                                                      self.CPsims[d].gpu_data.hv0,
                                                      self.CPsims[d].bathymetry.Bm,
                                                      self.nx, self.ny, self.t, self.dx, self.dy,
                                                      dt,
                                                      2, 2)
                self.CrossProductDrifter[d].drift(self.gpu_data.h0, self.gpu_data.hu0,
                                                  self.gpu_data.hv0,
                                                  self.bathymetry.Bm,
                                                  self.nx, self.ny, self.t, self.dx, self.dy,
                                                  dt,
                                                  2, 2)
            self.CPdrifter_t += dt

        if self.hasDrifters:
            self.drifters.driftFromSim(self, dt)
            self.drifter_t += dt
            return self.drifter_t
        return None

    def callKernel(self,
                   h_in: Array2D, hu_in: Array2D, hv_in: Array2D,
                   h_out: Array2D, hu_out: Array2D, hv_out: Array2D,
                   local_dt: float, wind_stress_t: float, atmospheric_pressure_t: float, rk_step: int,
                   split_steps=False):

        exchange_exclude = (h_in.pointer, hu_in.pointer, hv_in.pointer,
                            self.bathymetry.Bi.pointer, self.bathymetry.Bm.pointer, # TODO added this later 10/07
                            self.wind_stress_x_current_arr.pointer, self.wind_stress_y_current_arr.pointer)

        if split_steps:
            self.split_step(h_in, hu_in, hv_in, h_out, hu_out, hv_out, local_dt, wind_stress_t, atmospheric_pressure_t,
                            rk_step, exchange_exclude)
            return

        # "Beautify" code a bit by packing four int8s into a single int32
        # Note: Must match code in kernel!
        boundary_conditions = 0
        boundary_conditions = boundary_conditions | (int(self.boundary_conditions.north) << 24)
        boundary_conditions = boundary_conditions | (int(self.boundary_conditions.south) << 16)
        boundary_conditions = boundary_conditions | (int(self.boundary_conditions.east) << 8)
        boundary_conditions = boundary_conditions | (int(self.boundary_conditions.west) << 0)

        self.cdklm_swe_2D.async_call(self.global_size, self.local_size, self.gpu_stream,
                                     [local_dt,
                                      rk_step,
                                      h_in.pointer, h_in.pitch,
                                      hu_in.pointer, hu_in.pitch,
                                      hv_in.pointer, hv_in.pitch,
                                      h_out.pointer, h_out.pitch,
                                      hu_out.pointer, hu_out.pitch,
                                      hv_out.pointer, hv_out.pitch,
                                      self.bathymetry.Bi.pointer, self.bathymetry.Bi.pitch,
                                      self.bathymetry.Bm.pointer, self.bathymetry.Bm.pitch,
                                      self.bathymetry.mask_value,
                                      self.coriolis_f_arr.pointer,
                                      self.angle_arr.pointer,
                                      self.atmospheric_pressure_current_arr.pointer,
                                      self.atmospheric_pressure_next_arr.pointer,
                                      atmospheric_pressure_t,
                                      self.wind_stress_x_current_arr.pointer,
                                      self.wind_stress_x_next_arr.pointer,
                                      self.wind_stress_y_current_arr.pointer,
                                      self.wind_stress_y_next_arr.pointer,
                                      wind_stress_t,
                                      boundary_conditions], exchange_exclude=exchange_exclude)

    def split_step(self,
                   h_in: Array2D, hu_in: Array2D, hv_in: Array2D,
                   h_out: Array2D, hu_out: Array2D, hv_out: Array2D,
                   local_dt: float, wind_stress_t: float, atmospheric_pressure_t: float, rk_step: int,
                   exchange_exclude: Iterable):
        """
        Calculates the border first, exchanges the data, then calculates the inner data.
        """
        self.call_outer_kernel(h_in, hu_in, hv_in,
                               h_out, hu_out, hv_out,
                               local_dt, wind_stress_t, atmospheric_pressure_t, rk_step)
        self.logger.info("Finished outer, synchronizing...")

        self.gpu_stream.synchronize()
        self.logger.info("Finished synchronizing")
        self.call_inner_kernel(h_in, hu_in, hv_in,
                               h_out, hu_out, hv_out,
                               local_dt, wind_stress_t, atmospheric_pressure_t, rk_step, exchange_exclude)
        self.logger.info("Finished split computation")

    def call_inner_kernel(self,
                          h_in: Array2D, hu_in: Array2D, hv_in: Array2D,
                          h_out: Array2D, hu_out: Array2D, hv_out: Array2D,
                          local_dt: float, wind_stress_t: float, atmospheric_pressure_t: float, rk_step: int,
                          exchange_exclude: Iterable):
        """
        Used for splitting up computing the entire domain. This function is used for calculating the inner domain.
        Only computes interior cells that haven't been computed by call_outer_kernel.
        """
        # As it's the inner domain, boundary conditions should be ignored
        boundary_conditions = 0

        # Inner domain offsets: skip ghost cells on all sides
        # Row offset: skip north ghost cells
        offset = (self.ghost_cells.y, self.ghost_cells.x)

        self.logger.info("Starting Inner")
        self.cdklm_swe_2D_inner.async_call(self.global_size, self.local_size, self.gpu_stream,
                                           [local_dt,
                                            rk_step,
                                            h_in.offset_pointer(offset), h_in.pitch,
                                            hu_in.offset_pointer(offset), hu_in.pitch,
                                            hv_in.offset_pointer(offset), hv_in.pitch,
                                            h_out.offset_pointer(offset),
                                            h_out.pitch,
                                            hu_out.offset_pointer(offset),
                                            hu_out.pitch,
                                            hv_out.offset_pointer(offset),
                                            hv_out.pitch,
                                            self.bathymetry.Bi.offset_pointer(offset),
                                            self.bathymetry.Bi.pitch,
                                            self.bathymetry.Bm.offset_pointer(offset),
                                            self.bathymetry.Bm.pitch,
                                            self.bathymetry.mask_value,
                                            self.coriolis_f_arr.offset_pointer(offset),
                                            self.angle_arr.offset_pointer(offset),
                                            self.atmospheric_pressure_current_arr.pointer,
                                            self.atmospheric_pressure_next_arr.pointer,
                                            atmospheric_pressure_t,
                                            self.wind_stress_x_current_arr.pointer,
                                            self.wind_stress_x_next_arr.pointer,
                                            self.wind_stress_y_current_arr.pointer,
                                            self.wind_stress_y_next_arr.pointer,
                                            wind_stress_t,
                                            boundary_conditions],
                                           exchange=True,
                                           exchange_exclude=exchange_exclude)

    def call_outer_kernel(self,
                          h_in: Array2D, hu_in: Array2D, hv_in: Array2D,
                          h_out: Array2D, hu_out: Array2D, hv_out: Array2D,
                          local_dt: float, wind_stress_t: float, atmospheric_pressure_t: float, rk_step: int):

        boundary_ns = 0
        boundary_ew = 0
        boundary_ns = boundary_ns | (int(self.boundary_conditions.east) << 8)
        boundary_ns = boundary_ns | (int(self.boundary_conditions.west) << 0)
        boundary_north = boundary_ns | (int(self.boundary_conditions.north) << 24)
        boundary_south = boundary_ns | (int(self.boundary_conditions.south) << 16)
        boundary_ew = boundary_ew | (int(self.boundary_conditions.north) << 24)
        boundary_ew = boundary_ew | (int(self.boundary_conditions.south) << 16)
        boundary_east = boundary_ew | (int(self.boundary_conditions.east) << 8)
        boundary_west = boundary_ew | (int(self.boundary_conditions.west) << 0)

        # North boundary: rows [halo_y + ny, ny_halo)
        north_offset = (self.ny - self.ghost_cells.south, 0)
        north_offset_bathymetry = (self.ny - (1 + self.ghost_cells.south), 0)
        if self.coriolis_f_arr.shape[0] < self.ghost_cells.y:
            north_offset_coriolis = (0, 0)
        else:
            north_offset_coriolis = (self.ghost_cells.south, 0)
        if self.angle_arr.shape[0] < self.ghost_cells.y:
            north_offset_angle = (0, 0)
        else:
            north_offset_angle = (self.ghost_cells.south, 0)
        self.logger.info("Starting North")

        self.cdklm_swe_2D_nw.async_call((self.global_size[0], 1), self.local_size, self.gpu_stream,
                                        [local_dt,
                                         rk_step,
                                         h_in.offset_pointer(north_offset), h_in.pitch,
                                         hu_in.offset_pointer(north_offset), hu_in.pitch,
                                         hv_in.offset_pointer(north_offset), hv_in.pitch,
                                         h_out.offset_pointer(north_offset), h_out.pitch,
                                         hu_out.offset_pointer(north_offset), hu_out.pitch,
                                         hv_out.offset_pointer(north_offset), hv_out.pitch,
                                         self.bathymetry.Bi.offset_pointer(north_offset_bathymetry),
                                         self.bathymetry.Bi.pitch,
                                         self.bathymetry.Bm.offset_pointer(north_offset_bathymetry),
                                         self.bathymetry.Bm.pitch,
                                         self.bathymetry.mask_value,
                                         self.coriolis_f_arr.offset_pointer(north_offset_coriolis),
                                         self.angle_arr.offset_pointer(north_offset_angle),
                                         self.atmospheric_pressure_current_arr.pointer,
                                         self.atmospheric_pressure_next_arr.pointer,
                                         atmospheric_pressure_t,
                                         self.wind_stress_x_current_arr.pointer,
                                         self.wind_stress_x_next_arr.pointer,
                                         self.wind_stress_y_current_arr.pointer,
                                         self.wind_stress_y_next_arr.pointer,
                                         wind_stress_t,
                                         boundary_north],
                                        exchange=False)

        self.logger.info("Starting South")

        # South boundary: rows [0, halo_y)
        self.cdklm_swe_2D_nw.async_call((self.global_size[0], 1), self.local_size, self.gpu_stream,
                                        [local_dt,
                                         rk_step,
                                         h_in.pointer, h_in.pitch,
                                         hu_in.pointer, hu_in.pitch,
                                         hv_in.pointer, hv_in.pitch,
                                         h_out.pointer, h_out.pitch,
                                         hu_out.pointer, hu_out.pitch,
                                         hv_out.pointer, hv_out.pitch,
                                         self.bathymetry.Bi.pointer, self.bathymetry.Bi.pitch,
                                         self.bathymetry.Bm.pointer, self.bathymetry.Bm.pitch,
                                         self.bathymetry.mask_value,
                                         self.coriolis_f_arr.pointer,
                                         self.angle_arr.pointer,
                                         self.atmospheric_pressure_current_arr.pointer,
                                         self.atmospheric_pressure_next_arr.pointer,
                                         atmospheric_pressure_t,
                                         self.wind_stress_x_current_arr.pointer,
                                         self.wind_stress_x_next_arr.pointer,
                                         self.wind_stress_y_current_arr.pointer,
                                         self.wind_stress_y_next_arr.pointer,
                                         wind_stress_t,
                                         boundary_south],
                                        exchange=False)

        # East boundary: cols [halo_x + nx, nx_halo)
        # Skip north and south boundaries
        east_offset = (self.ghost_cells.south, self.nx - self.ghost_cells.west)
        east_bathymetry_offset = (self.ghost_cells.south + 1, self.nx - (1 + self.ghost_cells.west))
        if self.coriolis_f_arr.shape[0] < self.ghost_cells.y:
            east_coriolis_offset = (0, 0)
        else:
            east_coriolis_offset = (self.ghost_cells.south,
                                    max(0, self.coriolis_f_arr.shape[1] - self.ghost_cells.west))
        if self.angle_arr.shape[0] < self.ghost_cells.y:
            east_angle_offset = (0, 0)
        else:
            east_angle_offset = (self.ghost_cells.south, max(0, self.angle_arr.shape[1] - self.ghost_cells.west))

        self.logger.info("Starting East")

        self.cdklm_swe_2D_ew.async_call((1, self.global_size[1]), self.local_size, self.gpu_stream,
                                        [local_dt,
                                         rk_step,
                                         h_in.offset_pointer(east_offset), h_in.pitch,
                                         hu_in.offset_pointer(east_offset), hu_in.pitch,
                                         hv_in.offset_pointer(east_offset), hv_in.pitch,
                                         h_out.offset_pointer(east_offset), h_out.pitch,
                                         hu_out.offset_pointer(east_offset), hu_out.pitch,
                                         hv_out.offset_pointer(east_offset), hv_out.pitch,
                                         self.bathymetry.Bi.offset_pointer(east_bathymetry_offset),
                                         self.bathymetry.Bi.pitch,
                                         self.bathymetry.Bm.offset_pointer(east_bathymetry_offset),
                                         self.bathymetry.Bm.pitch,
                                         self.bathymetry.mask_value,
                                         self.coriolis_f_arr.offset_pointer(east_coriolis_offset),
                                         self.angle_arr.offset_pointer(east_angle_offset),
                                         self.atmospheric_pressure_current_arr.pointer,
                                         self.atmospheric_pressure_next_arr.pointer,
                                         atmospheric_pressure_t,
                                         self.wind_stress_x_current_arr.pointer,
                                         self.wind_stress_x_next_arr.pointer,
                                         self.wind_stress_y_current_arr.pointer,
                                         self.wind_stress_y_next_arr.pointer,
                                         wind_stress_t,
                                         boundary_east],
                                        exchange=False)

        # West boundary: cols [0, halo_x)
        # Skip north and south boundaries

        self.logger.info("Starting West")

        west_offset = (0, self.nx - self.ghost_cells.west)
        west_bathymetry_offset = (0 + 1, self.nx - (1 + self.ghost_cells.west))
        if self.coriolis_f_arr.shape[0] < self.ghost_cells.y:
            west_coriolis_offset = (0, 0)
        else:
            west_coriolis_offset = (0,
                                    max(0, self.coriolis_f_arr.shape[1] - self.ghost_cells.west))
        if self.angle_arr.shape[0] < self.ghost_cells.y:
            west_angle_offset = (0, 0)
        else:
            west_angle_offset = (0, max(0, self.angle_arr.shape[1] - self.ghost_cells.west))

        self.cdklm_swe_2D_ew.async_call((1, self.global_size[1]), self.local_size, self.gpu_stream,
                                        [local_dt,
                                         rk_step,
                                         h_in.offset_pointer(west_offset),
                                         int(hu_in.offset_pointer(west_offset)), hu_in.pitch,
                                         int(hv_in.offset_pointer(west_offset)), hv_in.pitch,
                                         int(h_out.offset_pointer(west_offset)), h_out.pitch,
                                         int(hu_out.offset_pointer(west_offset)), hu_out.pitch,
                                         int(hv_out.offset_pointer(west_offset)), hv_out.pitch,
                                         int(self.bathymetry.Bi.offset_pointer(west_bathymetry_offset)),
                                         self.bathymetry.Bi.pitch,
                                         int(self.bathymetry.Bm.offset_pointer(west_bathymetry_offset)),
                                         self.bathymetry.Bm.pitch,
                                         self.bathymetry.mask_value,
                                         int(self.coriolis_f_arr.offset_pointer(west_coriolis_offset)),
                                         int(self.angle_arr.offset_pointer(west_angle_offset)),
                                         self.atmospheric_pressure_current_arr.pointer,
                                         self.atmospheric_pressure_next_arr.pointer,
                                         atmospheric_pressure_t,
                                         self.wind_stress_x_current_arr.pointer,
                                         self.wind_stress_x_next_arr.pointer,
                                         self.wind_stress_y_current_arr.pointer,
                                         self.wind_stress_y_next_arr.pointer,
                                         wind_stress_t,
                                         boundary_west],
                                        exchange=False)

    def perturbState(self, perturbation_scale=1.0, update_random_field=True, q0_scale=1):
        if not q0_scale == 1:
            Warning(
                "CDKLM16.perturbState argument 'q0_scale' will be deprecated. Please use 'perturbation_scale' instead")
            perturbation_scale = q0_scale

        if self.model_error is not None:
            self.model_error.perturbSim(self, perturbation_scale=perturbation_scale,
                                        update_random_field=update_random_field)

    def perturbSimilarAs(self, other_sim, perturbation_scale=1.0):
        self.model_error.perturbSimSimilarAs(self, modelError=other_sim.model_error,
                                             perturbation_scale=perturbation_scale)

    def applyBoundaryConditions(self):
        self.bc_kernel.boundaryCondition(self.gpu_stream,
                                         self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)

    def dataAssimilationStep(self, observation_time, other_sim=None, model_error_final_step=True, write_now=True,
                             courant_number=0.8):
        """
        The model runs until self.t = observation_time - self.model_time_step with model error.
        If model_error_final_step is true, another stochastic model_time_step is performed, 
        otherwise a deterministic model_time_step.
        """
        # For the IEWPF scheme, it is important that the final timestep before the
        # observation time is a full time step (fully deterministic). 
        # We therefore make sure to take the (potential) small timestep first in this function,
        # followed by appropriately many full time steps.

        full_model_time_steps = int(round(observation_time - self.t) / self.model_time_step)
        leftover_step_size = observation_time - self.t - full_model_time_steps * self.model_time_step

        # Avoid a too small extra timestep
        if leftover_step_size / self.model_time_step < 0.1 and full_model_time_steps > 1:
            leftover_step_size += self.model_time_step
            full_model_time_steps -= 1

        # Force leftover_step_size to zero if it is very small compared to the model_time_step
        if leftover_step_size / self.model_time_step < 0.00001:
            leftover_step_size = 0

        assert (full_model_time_steps > 0), "There is less than CDKLM16.model_time_step until the observation"

        # Start by updating the timestep size.
        self.updateDt(courant_number=courant_number)
        if other_sim is not None:
            other_sim.updateDt(courant_number=courant_number)

        # Loop standard steps:
        for i in range(full_model_time_steps + 1):

            if i == 0 and leftover_step_size == 0:
                continue
            elif i == 0:
                # Take the leftover step
                self.step(leftover_step_size, apply_stochastic_term=False, write_now=False)
                self.perturbState(perturbation_scale=np.sqrt(leftover_step_size / self.model_time_step))
                if other_sim is not None:
                    other_sim.step(leftover_step_size, apply_stochastic_term=False, write_now=False)
                    other_sim.perturbSimilarAs(self,
                                               perturbation_scale=np.sqrt(leftover_step_size / self.model_time_step))

            else:
                # Take standard steps
                self.step(self.model_time_step, apply_stochastic_term=False, write_now=False)
                if (i < full_model_time_steps) or model_error_final_step:
                    self.perturbState()
                if other_sim is not None:
                    other_sim.step(self.model_time_step, apply_stochastic_term=False, write_now=False)
                    if (i < full_model_time_steps) or model_error_final_step:
                        other_sim.perturbSimilarAs(self)

            self.total_time_steps += 1

            # Update dt now and then
            if self.total_time_steps % 5 == 0:
                self.updateDt(courant_number=courant_number)

        if self.write_netcdf and write_now:
            self.sim_writer.write_timestep(self)

        assert (round(observation_time) == round(
            self.t)), 'The simulation time is not the same as observation time after dataAssimilationStep! \n' + \
                      '(self.t, observation_time, diff): ' + str((self.t, observation_time, self.t - observation_time))

    def writeState(self):
        if self.write_netcdf:
            self.sim_writer.write_timestep(self)

    def updateDt(self, courant_number: float = None):
        """
        Updates the time step self.dt by finding the maximum size of dt according to the 
        CFL conditions, and scale it with the provided courant number (0.8 on default).
        """
        if courant_number is None:
            courant_number = self.courant_number

        self.per_block_max_dt_kernel.async_call(self.global_size, self.local_size, self.gpu_stream,
                                                [self.nx, self.ny,
                                                 self.dx, self.dy,
                                                 self.g,
                                                 self.gpu_data.h0.pointer, self.gpu_data.h0.pitch,
                                                 self.gpu_data.hu0.pointer, self.gpu_data.hu0.pitch,
                                                 self.gpu_data.hv0.pointer, self.gpu_data.hv0.pitch,
                                                 self.bathymetry.Bm.pointer, self.bathymetry.Bm.pitch,
                                                 self.bathymetry.mask_value,
                                                 self.device_dt.pointer, self.device_dt.pitch])

        self.max_dt_reduction_kernel.async_call((1, 1),
                                                (self.num_threads_dt, 1, 1),
                                                self.gpu_stream,
                                                [self.num_blocks_dt,
                                                 self.device_dt.pointer,
                                                 self.max_dt_buffer.pointer])

        dt_host = float(self.max_dt_buffer.download(self.gpu_stream)[0, 0])

        if dt_host == 0:
            raise RuntimeError("Change in time (dt) is zero.")

        self.dt = courant_number * float(dt_host)

    def _getMaxTimestepHost(self, courant_number=0.8) -> float:
        """
        Calculates the maximum allowed time step according to the CFL conditions and scales the
        result with the provided courant number (0.8 on default).
        This function is for reference only, and suboptimally implemented on the host.
        """
        eta, hu, hv = self.download(interior_domain_only=True)
        Hm = self.downloadBathymetry()[1][2:-2, 2:-2]
        # print(eta.shape, Hm.shape)

        h = eta + Hm
        gravity_waves = np.sqrt(self.g * h)
        u = hu / h
        v = hv / h

        max_dt = 0.25 * min(self.dx / np.max(np.abs(u) + gravity_waves),
                            self.dy / np.max(np.abs(v) + gravity_waves))

        return courant_number * max_dt

    def downloadBathymetry(self, interior_domain_only=False):
        Bi, Bm = self.bathymetry.download(self.gpu_stream)

        if interior_domain_only:
            Bi = Bi[self.interior_domain_indices[2]:self.interior_domain_indices[0] + 1,
            self.interior_domain_indices[3]:self.interior_domain_indices[1] + 1]
            Bm = Bm[self.interior_domain_indices[2]:self.interior_domain_indices[0],
            self.interior_domain_indices[3]:self.interior_domain_indices[1]]

        return [Bi, Bm]

    def getLandMask(self, interior_domain_only=True):
        if self.gpu_data.h0.mask is None:
            return None

        if interior_domain_only:
            return self.gpu_data.h0.mask[2:-2, 2:-2]
        else:
            return self.gpu_data.h0.mask

    def downloadDt(self):
        return self.device_dt.download(self.gpu_stream)

    def downloadGeoEqNorm(self):

        uxpvy_cpu = self.geoEq_uxpvy.download(self.gpu_stream)
        Kx_cpu = self.geoEq_Kx.download(self.gpu_stream)
        Ly_cpu = self.geoEq_Ly.download(self.gpu_stream)

        uxpvy_norm = np.linalg.norm(uxpvy_cpu)
        Kx_norm = np.linalg.norm(Kx_cpu)
        Ly_norm = np.linalg.norm(Ly_cpu)

        return uxpvy_norm, Kx_norm, Ly_norm


@dataclass
class CDKLMDefines:
    block_width: int
    block_height: int
    desingularization_eps: float
    flux_slope_eps: float
    depth_cutoff: float
    theta: float
    rk_order: int
    nx: int
    ny: int
    dx: float
    dy: float
    gravity: float
    friction: float
    rho_o: float
    wind_stress_factor: float
    one_dimensional: bool
    flux_balancer: float
    coriolis_f: npt.NDArray
    angle: npt.NDArray
    atmospheric_pressure: AtmosphericPressure.AtmosphericPressure
    wind_stress: WindStress.WindStress
    use_direct_lookup: bool
    ghost_cells: GhostCells

    def __compute_defines(self, nx: int, ny: int):
        return {
            'block_width': self.block_width, 'block_height': self.block_height,
            'KPSIMULATOR_DESING_EPS': "{:.12f}f".format(self.desingularization_eps),
            'KPSIMULATOR_FLUX_SLOPE_EPS': "{:.12f}f".format(self.flux_slope_eps),
            'KPSIMULATOR_DEPTH_CUTOFF': "{:.12f}f".format(self.depth_cutoff),
            'THETA': "{:.12f}f".format(self.theta),
            'RK_ORDER': self.rk_order,
            'NX': nx,
            'NY': ny,
            'DX': "{:.12f}f".format(self.dx),
            'DY': "{:.12f}f".format(self.dy),
            'GRAV': "{:.12f}f".format(self.gravity),
            'FRIC': "{:.12f}f".format(self.friction),
            'RHO_O': "{:.12f}f".format(self.rho_o),
            'WIND_STRESS_FACTOR': "{:.12f}f".format(self.wind_stress_factor),
            'ONE_DIMENSIONAL': int(self.one_dimensional),
            'FLUX_BALANCER': "{:.12f}f".format(self.flux_balancer),
            # These minimums here are a bit hacky
            # TODO understand the sizes of the arrays below, because my assumption is that they can be smaller than nx and ny
            'CORIOLIS_F_NX': self.coriolis_f.shape[1],
            'CORIOLIS_F_NY': self.coriolis_f.shape[0],
            'ANGLE_NX': self.angle.shape[1],
            'ANGLE_NY': self.angle.shape[0],
            'ATMOS_PRES_NX': self.atmospheric_pressure.P[0].shape[1],
            'ATMOS_PRES_NY': self.atmospheric_pressure.P[0].shape[0],
            'WIND_STRESS_X_NX': self.wind_stress.stress_u[0].shape[1],
            'WIND_STRESS_X_NY': self.wind_stress.stress_u[0].shape[0],
            'WIND_STRESS_Y_NX': self.wind_stress.stress_v[0].shape[1],
            'WIND_STRESS_Y_NY': self.wind_stress.stress_v[0].shape[0],
            'USE_DIRECT_LOOKUP': self.use_direct_lookup
        }

    @property
    def default(self):
        return self.__compute_defines(self.nx, self.ny)

    @property
    def north_south(self):
        """
        Gets the definitions used for a North/South kernel.
        """
        ny = self.ghost_cells.y + self.ghost_cells.north

        return self.__compute_defines(self.nx, ny)

    @property
    def east_west(self):
        """
        Gets the definitions used for an East/West kernel.
        """
        nx = self.ghost_cells.x + self.ghost_cells.east
        # Avoid double computation
        ny = self.ny - self.ghost_cells.y

        return self.__compute_defines(nx, ny)

    @property
    def inner(self):
        """
        Gets the definition used for the inner cells that do not require ghost cells.
        """
        nx = self.nx - self.ghost_cells.x
        ny = self.ny - self.ghost_cells.y

        return self.__compute_defines(nx, ny)
