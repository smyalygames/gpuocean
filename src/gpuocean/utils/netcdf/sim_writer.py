from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import logging
from datetime import datetime
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from git import Repo, InvalidGitRepositoryError, GitError
from netCDF4 import Dataset
import numpy as np
from mpi4py import MPI

from gpuocean.utils.mpi import Grid

if TYPE_CHECKING:
    from netCDF4 import Variable
    import numpy.typing as npt
    from matplotlib.pyplot import Axes

    from gpuocean.utils.types import AnySimulator


class SimNetCDFWriter:
    """
    Handles writing to a netCDF file.
    """

    def __init__(self, sim: AnySimulator,
                 super_dir: Optional[str | os.PathLike[str]] = None, filename: Optional[str] = None,
                 num_layers: int = 1, staggered_grid: bool = False, ignore_ghostcells: Optional[bool] = False,
                 offset_x: int = 0, offset_y: int = 0,
                 write_parallel: bool = True, write_async: bool = True):
        """
        Writes simulator output to a netCDF file.
        :param sim: Simulator that will be used for the netCDF output.
        :param super_dir: Directory to output the netCDF data.
        :param filename: Name of the netCDF file. If none, will use a generated filename.
        :param num_layers: Number of layers in the simulator.
        :param staggered_grid: Set if the simulator uses a staggered grid.
        :param ignore_ghostcells: Will not write ghost cells to the file if set to ``True``.
        :param offset_x: Offset x-axis from the origin of the simulator in the netCDF file.
            The offset is `offset_x * dx`.
        :param offset_x: Offset y-axis from the origin of the simulator in the netCDF file.
            The offset is `offset_y * dx`.
        :param write_parallel: Writes data in parallel for a timestep using HDF5 and MPI.
        :param write_async: Makes each call to `self.write_timestep` asynchronous.
        """

        self.logger = logging.getLogger(__name__)
        self.current_directory = os.getcwd()

        # Parallel netCDF4 write?
        # TODO: Implement check/test for feature or take as an argument
        self.write_parallel = write_parallel

        # Asynchronous writes
        self.write_async = write_async
        self.executor = ThreadPoolExecutor(max_workers=1)

        # GPU compute queue:
        self.gpu_stream = sim.gpu_stream

        # Write options for netCDF
        self.ignore_ghostcells = ignore_ghostcells
        self.staggered_grid = staggered_grid
        self.num_layers = num_layers

        self.text_pos = -1

        # Simulator Information
        self.sim_name = str(sim.__class__.__name__)
        self.minmod_theta = sim.theta
        self.coriolis_force = sim.f
        self.coriolis_beta = sim.coriolis_beta
        self.y_zero_reference_cell = sim.y_zero_reference_cell - 2
        self.wind_stress = sim.wind_stress.source_filename
        self.eddy_viscosity_coefficient = sim.A
        self.bottom_friction_r = sim.r
        g = sim.g

        self.dx = sim.dx
        self.dy = sim.dy
        self.dt = sim.dt
        auto_dt = False  # TODO add check in the simulator

        # Identification of the simulator run
        timestamp = datetime.now()
        self.timestamp = timestamp.strftime("%Y_%m_%d-%H_%M_%S")
        self.date = timestamp.strftime("%Y_%m_%d")

        ## Get git information
        try:
            repo_path = os.path.dirname(os.path.abspath(__file__))
            repo = Repo(repo_path, search_parent_directories=True)
            self.git_hash = str(repo.head.commit.hexsha)
        except (InvalidGitRepositoryError, GitError) as e:
            self.logger.warning("Could not get git information", exc_info=e)
            self.git_hash = "git info missing..."

        # Handle file locations
        self.super_dir: str = super_dir
        if super_dir is None:
            self.super_dir = self.current_directory

        if filename is None:
            self.dir_name = "netcdf_" + self.date
            self.dir_name = os.path.join(self.super_dir, self.dir_name)
            file = f"{self.sim_name}_{self.timestamp}.nc"
            self.filename = os.path.join(self.dir_name, file)
        elif os.path.dirname(filename) == '':
            self.dir_name = "netcdf_" + self.date
            self.dir_name = os.path.join(self.super_dir, self.dir_name)
            self.filename = os.path.join(self.dir_name, filename)
        else:
            self.filename = filename
            self.dir_name = os.path.dirname(filename)

        # Make sure that the suffix is for a netCDF file
        if Path(self.filename).suffix != '.nc':
            self.filename += '.nc'

        # Simulator data
        self.boundary_conditions = str(sim.boundary_conditions)
        ## Machine readable boundary conditions
        self.boundary_conditions_sponge_mr = str(sim.boundary_conditions.getSponge())

        self.dt = sim.dt
        if self.staggered_grid:
            Hm = sim.H.download(self.gpu_stream)
            Hi = None
        else:
            Hi, Hm = sim.bathymetry.download(self.gpu_stream)

        if self.sim_name == 'KP07':
            if sim.use_rk2:
                self.time_integrator = 2
            else:
                self.time_integrator = 1
        else:
            self.time_integrator: int = sim.rk_order

        self.ghost_cells = sim.ghost_cells

        # Checks if domain decomposition is occurring if MPI is being used.
        if self.write_parallel:
            try:
                nx = Grid.global_nx
                ny = Grid.global_ny
                x_pos = Grid.x_pos
                y_pos = Grid.y_pos
                nodes_x = Grid.nodes_x
                nodes_y = Grid.nodes_y

                # Coordinates for parallel netCDF
                self.x0 = Grid.x0
                self.x1 = Grid.x1
                self.y0 = Grid.y0
                self.y1 = Grid.y1
            except AttributeError as e:
                error_message = "Grid was not initialised for writing netCDF files with MPI."
                self.logger.error(error_message, exc_info=e)
                raise RuntimeError(error_message)
        else:
            x_pos = 0
            y_pos = 0
            nx = sim.nx
            ny = sim.ny
            nodes_x = 1
            nodes_y = 1

            self.x0 = 0
            self.x1 = nx
            self.y0 = 0
            self.y1 = ny

        ## Append ghost cells to the edges if those are being included
        if not self.ignore_ghostcells:
            nx += self.ghost_cells.total_x * nodes_x
            self.x0 += self.ghost_cells.total_x * x_pos
            self.x1 += self.ghost_cells.total_x * (x_pos + 1)

            ny += self.ghost_cells.total_y * nodes_y
            self.y0 += self.ghost_cells.total_y * y_pos
            self.y1 += self.ghost_cells.total_y * (y_pos + 1)


        local_nx = self.x1 - self.x0
        local_ny = self.y1 - self.y0

        # netCDF file initialization
        os.makedirs(self.dir_name, exist_ok=True)
        self.nc = Dataset(self.filename, 'w', format='NETCDF4', clobber=True,
                          parallel=self.write_parallel, comm=MPI.COMM_WORLD, info=MPI.Info())

        # Write netCDF global attributes
        self.nc.git_hash = self.git_hash
        self.nc.ignore_ghostcells = str(self.ignore_ghostcells)
        self.nc.num_layers = self.num_layers
        self.nc.staggered_grid = str(self.staggered_grid)
        self.nc.simulator_short = self.sim_name
        # NOTE if writing in parallel, a parent class will have to write the global boundary conditions
        if not self.write_parallel:
            self.nc.boundary_conditions = self.boundary_conditions
            self.nc.boundary_conditions_sponge_mr = self.boundary_conditions_sponge_mr
        self.nc.time_integrator = self.time_integrator
        self.nc.minmod_theta = self.minmod_theta
        self.nc.coriolis_force = self.coriolis_force
        self.nc.coriolis_beta = self.coriolis_beta
        self.nc.y_zero_reference_cell = self.y_zero_reference_cell
        if self.wind_stress is not None:
            self.nc.wind_stress_source = self.wind_stress
        self.nc.eddy_viscosity_coefficient = self.eddy_viscosity_coefficient
        self.nc.g = g
        self.nc.nx = nx
        self.nc.ny = ny
        self.nc.dx = self.dx
        self.nc.dy = self.dy
        self.nc.dt = self.dt
        self.nc.auto_dt = str(auto_dt)
        self.nc.bottom_friction_r = self.bottom_friction_r
        self.nc.ghost_cells_north = self.ghost_cells.north
        self.nc.ghost_cells_east = self.ghost_cells.east
        self.nc.ghost_cells_south = self.ghost_cells.south
        self.nc.ghost_cells_west = self.ghost_cells.west

        # TODO continue filling out the rest of the attributes required.

        forecast_group = self.nc.createGroup(f"forecasts/{self.sim_name}")
        moment_group = forecast_group.createGroup("moments")
        moment_u_group = moment_group.createGroup("u")
        moment_v_group = moment_group.createGroup("v")

        # Create netCDF dimensions and variables
        forecast_group.createDimension('time', None)

        nc_vars: dict[str, list[Variable]] = {'x': [], 'y': []}
        self.nc.createDimension('x', nx)
        self.nc.createDimension('y', ny)
        nc_vars['x'].append(self.nc.createVariable('x', np.float64, ('x',)))
        nc_vars['y'].append(self.nc.createVariable('y', np.float64, ('y',)))

        references_group = self.nc.createGroup("references")

        if (not self.ignore_ghostcells) and self.staggered_grid:
            if self.sim_name == 'FBL':
                x_hu_nx = nx - 1
            else:
                x_hu_nx = nx + 1
            moment_u_group.createDimension('x', x_hu_nx)
            moment_u_group.createDimension('y', ny)
            moment_v_group.createDimension('x', nx)
            moment_v_group.createDimension('y', ny + 1)
            nc_vars['x'].append(self.nc.createVariable('x', np.float64, ('x',)))
            nc_vars['y'].append(self.nc.createVariable('y', np.float64, ('y',)))
            nc_vars['x'].append(self.nc.createVariable('x', np.float64, ('x',)))
            nc_vars['y'].append(self.nc.createVariable('y', np.float64, ('y',)))
        else:
            Hi_group = references_group.createGroup("Hi")
            Hi_group.createDimension('x', nx + 1)
            Hi_group.createDimension('y', ny + 1)
            nc_vars['x'].append(Hi_group.createVariable('x', np.float64, ('x',)))
            nc_vars['y'].append(Hi_group.createVariable('y', np.float64, ('y',)))

        # TODO manage writing ensembles

        x_ghost_cell_buffer = 0
        y_ghost_cell_buffer = 0
        if not self.ignore_ghostcells:
            x_ghost_cell_buffer = self.ghost_cells.west * (x_pos + 1)
            y_ghost_cell_buffer = self.ghost_cells.south * (y_pos + 1)

        for var in nc_vars['x']:
            buffer = 0
            if x_pos == nodes_x - 1:
                buffer = var.shape[0] - nx
            x_start_stop = (((self.x0 - x_ghost_cell_buffer) * self.dx) + self.dx / 2.0,
                            ((self.x1 + buffer - x_ghost_cell_buffer) * self.dx) - self.dx / 2.0)
            var[self.x0:self.x1 + buffer] = np.linspace(*x_start_stop, local_nx + buffer, dtype=np.float64)
            var.standard_name = "projection_x_coordinate"
            var.axis = "X"
            var.units = 'meter'

        for var in nc_vars['y']:
            buffer = 0
            if y_pos == nodes_y - 1:
                buffer = var.shape[0] - ny
            y_start_stop = (((self.y0 - y_ghost_cell_buffer) * self.dy) + self.dy / 2.0,
                            ((self.y1 + buffer - y_ghost_cell_buffer) * self.dy) - self.dy / 2.0)
            var[self.y0:self.y1 + buffer] = np.linspace(*y_start_stop, local_ny + buffer, dtype=np.float64)
            var.standard_name = "projection_y_coordinate"
            var.axis = "Y"
            var.units = 'meter'

        ## Create bogus projection variable
        self.projection = self.nc.createVariable('projection_stere', np.float32)
        self.projection.grid_mapping_name = 'polar_stereographic'
        self.projection.scale_factor_at_projection_origin = 0.9330127018922193
        self.projection.straight_vertical_longitude_from_pole = 70.0
        self.projection.latitude_of_projection_origin = 90.0
        self.projection.earth_radius = 6371000.0
        self.projection.proj4 = '+proj=stere +lat_0=90 +lon_0=70 +lat_ts=60 +units=m +a=6.371e+06 +e=0 +no_defs'

        ## Create a land mask
        self.land_mask = self.nc.createVariable('land_binary_mask', np.int8, ('y', 'x'))
        self.land_mask.standard_name = 'land_binary_mask'
        self.land_mask.units = '1'
        self.land_mask[:] = 0

        ## Create bathymetry/equilibrium depth
        self.Hm = references_group.createVariable('Hm', np.float32, ('y', 'x'), zlib=True)
        self.Hm.standard_name = 'water_surface_reference_datum_altitude'
        self.Hm.grid_mapping = 'projection_stere'
        self.Hm.coordinates = 'y x'
        self.Hm.units = 'meter'
        Hm = self._handle_ghost_cells(Hm)
        self.Hm[self.y0:self.y1, self.x0:self.x1] = Hm

        if not self.staggered_grid:
            self.Hi = Hi_group.createVariable("Hi", np.float32, ('y', 'x'), zlib=True)
            self.Hi.standard_name = 'water_surface_reference_datum_altitude'
            self.Hi.grid_mapping = 'projection_stere'
            self.Hi.coordinates = 'y x'
            self.Hi.units = 'meter'
            append_x = int(nx == self.x1)
            append_y = int(ny == self.y1)
            Hi = self._handle_ghost_cells(Hi)
            # TODO make these if statements nicer, feels unnecessary
            if nx != self.x1:
                Hi = Hi[:, :-1]
            if ny != self.y1:
                Hi = Hi[:-1]
            self.Hi[self.y0:self.y1 + append_y, self.x0:self.x1 + append_x] = Hi

        self.time = forecast_group.createVariable('time', np.float64, ('time',))
        self.time.units = 'seconds since 1970-01-01 00:00:00'
        self.time.set_collective(self.write_parallel)

        self.eta = forecast_group.createVariable('eta', np.float32, ('time', 'y', 'x'), zlib=True)
        self.eta.set_collective(self.write_parallel)
        moment_dims = ('time', 'y', 'x')
        self.hu = moment_u_group.createVariable('hu', np.float32, moment_dims, zlib=True)
        self.hu.set_collective(self.write_parallel)
        self.hv = moment_v_group.createVariable('hv', np.float32, moment_dims, zlib=True)
        self.hv.set_collective(self.write_parallel)

        self.eta.standard_name = 'water_surface_height_above_reference_datum'
        self.hu.standard_name = 'x_sea_water_velocity'
        self.hv.standard_name = 'y_sea_water_velocity'
        self.eta.grid_mapping = 'projection_stere'
        self.hu.grid_mapping = 'projection_stere'
        self.hv.grid_mapping = 'projection_stere'
        self.eta.coordinates = 'y x'
        self.hu.coordinates = 'hu_y hu_x'
        self.hv.coordinates = 'hv_y hv_x'

        self.eta.units = 'meter'
        self.hu.units = 'meter second-1'
        self.hv.units = 'meter second-1'

        self.i = 0
        # Initial conditions of the simulator should be added as the first element to the above arrays
        self.write_timestep(sim)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(f"Closing netCDF file {self.filename}...")
        self.nc.close()

    def write_timestep(self, sim: AnySimulator) -> None:
        """
        Writes the state of the simulator to the netCDF file using a simulator.
        :param sim: Simulator to get the state from.
        """
        eta, hu, hv = sim.download()
        time = sim.t

        args=(time, eta, hu, hv)

        if self.write_async:
            self.executor.submit(self.write, *args)
        else:
            self.write(*args)

    def write(self, t: int | float, eta: npt.NDArray, hu: npt.NDArray, hv: npt.NDArray,
              eta2: Optional[npt.NDArray] = None, hu2: Optional[npt.NDArray] = None,
              hv2: Optional[npt.NDArray] = None) -> None:
        """
        Writes the state of the simulator to the netCDF file.
        """
        eta = self._handle_ghost_cells(eta)
        hu = self._handle_ghost_cells(hu)
        hv = self._handle_ghost_cells(hv)

        self.time[self.i] = t
        self.eta[self.i, self.y0:self.y1, self.x0:self.x1] = eta
        self.hu[self.i, self.y0:self.y1, self.x0:self.x1] = hu
        self.hv[self.i, self.y0:self.y1, self.x0:self.x1] = hv

        if self.num_layers == 2:
            raise NotImplementedError("NetCDF file is not set up for 2 layers yet.")
            if eta2 is None or hu2 is None or hv2 is None:
                raise RuntimeError("One of the 2nd layers provided is of type None.")

            eta2 = self._handle_ghost_cells(eta2)
            hu2 = self._handle_ghost_cells(hu2)
            hv2 = self._handle_ghost_cells(hv2)

            self.eta2[self.i, self.y0:self.y1, self.x0:self.x1] = eta2
            self.hu2[self.i, self.y0:self.y1, self.x0:self.x1] = hu2
            self.hv2[self.i, self.y0:self.y1, self.x0:self.x1] = hv2

        self.i += 1

    def _handle_ghost_cells(self, array: npt.NDArray) -> npt.NDArray:
        """
        Handles including or removing ghost cells from an array.
        :param array: NumPy array to handle the ghost cells.
        :returns: Array with or without ghost cells depending on if `ignore_ghostcells` is set.
        """
        if not self.ignore_ghostcells:
            return array

        return array[self.ghost_cells.north:-self.ghost_cells.south, self.ghost_cells.east:-self.ghost_cells.west]

    def _add_text(self, ax: Axes, message: str) -> None:
        """
        Helper function to add text to a plot on matplotlib.
        :param ax: Plot to add text to.
        :param message: Text to add to the plot.
        """
        break_point = 70
        if len(message) > break_point:
            rest = '     ' + message[break_point:]
            ax.text(0.1, self.text_pos, message[0:break_point])
            self.text_pos -= 0.2
            self._add_text(ax, rest)
        else:
            ax.text(0.1, self.text_pos, message)
            self.text_pos -= 0.2

    def info_plot(self, ax: Axes) -> None:
        """
        Adds textual information from of the simulator to a plot.
        :param ax: Plot to add information to.
        """
        self.text_pos = 2.3
        ax.text(1, 2.8, 'NetCDF INFO')

        self._add_text(ax, 'working directory: ' + self.current_directory)
        self._add_text(ax, 'filename: ' + self.filename)
        self._add_text(ax, '')
        self._add_text(ax, 'git hash: ' + self.git_hash)
        self._add_text(ax, '')
        self._add_text(ax, 'Simulator: ' + self.sim_name)
        self._add_text(ax, 'BC: ' + str(self.boundary_conditions))
        self._add_text(ax, 'f: ' + str(self.boundary_conditions))
        self._add_text(ax, f"dt: {self.dt}, dx: {self.dx}, dy: {self.dy}")
        self._add_text(ax, 'wind type: ' + str(self.wind_stress))

        ax.axis((0, 6, 0, 3))
