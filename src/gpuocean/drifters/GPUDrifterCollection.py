# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018, 2023  SINTEF Digital

This python class implements a DrifterCollection living on the GPU.

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

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from gpuocean.drifters.BaseDrifterCollection import BaseDrifterCollection
from gpuocean.utils.WindStress import WindStress
from gpuocean.utils.Common import BoundaryConditions
from gpuocean.utils.gpu import GPUHandler, GPUStream, Array2D

if TYPE_CHECKING:
    from gpuocean.utils.gpu import KernelContext, module_t


class GPUDrifterCollection(BaseDrifterCollection):
    def __init__(self, gpu_ctx: KernelContext, num_drifters: int,
                 observation_variance=0.01,
                 boundary_conditions=BoundaryConditions(),
                 initialization_cov_drifters=None,
                 domain_size_x=1.0, domain_size_y=1.0,
                 gpu_stream: GPUStream = None,
                 initialize=False,
                 wind=WindStress(),
                 wind_drift_factor=0.0,
                 block_width=64):

        super(GPUDrifterCollection, self).__init__(num_drifters,
                                                   observation_variance=observation_variance,
                                                   boundaryConditions=boundary_conditions,
                                                   domain_size_x=domain_size_x,
                                                   domain_size_y=domain_size_y)

        # Define CUDA environment:
        self.gpu_ctx = gpu_ctx
        self.block_width = block_width
        self.block_height = 1
        self.wind = wind
        self.wind_drift_factor = float(wind_drift_factor)

        self.gpu_stream = gpu_stream
        if self.gpu_stream is None:
            self.gpu_stream = GPUStream()

        self.sensitivity = 1.0

        self.driftersHost = np.zeros((self.getNumDrifters() + 1, 2), dtype=np.float32, order='C')
        self.driftersDevice = Array2D(self.gpu_stream,
                                      2, self.getNumDrifters() + 1, 0, 0,
                                      self.driftersHost)

        self.drift_kernels = gpu_ctx.get_kernel("driftKernels",
                                                defines={'block_width': self.block_width,
                                                         'block_height': self.block_height,
                                                         'WIND_X_NX': self.wind.wind_u[0].shape[1],
                                                         'WIND_X_NY': self.wind.wind_u[0].shape[0],
                                                         'WIND_Y_NX': self.wind.wind_v[0].shape[1],
                                                         'WIND_Y_NY': self.wind.wind_v[0].shape[0]
                                                         })
        # Define wind arrays
        t = 0  # TODO: check if this is correct
        t_max_index = len(self.wind.t) - 1
        t0_index = max(0, np.searchsorted(self.wind.t, t) - 1)
        t1_index = min(t_max_index, int(np.searchsorted(self.wind.t, t)))
        self.wind_x_current_arr = Array2D(self.gpu_stream,
                                          self.wind.wind_u[t0_index].shape[1], self.wind.wind_u[t0_index].shape[0],
                                          0,0,
                                          self.wind.wind_u[t0_index], padded=False)
        self.wind_y_current_arr = Array2D(self.gpu_stream,
                                          self.wind.wind_v[t0_index].shape[1], self.wind.wind_v[t0_index].shape[0],
                                          0, 0,
                                          self.wind.wind_v[t0_index], padded=False)
        self.wind_x_next_arr = Array2D(self.gpu_stream,
                                       self.wind.wind_u[t1_index].shape[1], self.wind.wind_u[t1_index].shape[0],
                                       0, 0,
                                       self.wind.wind_u[t1_index], padded=False)
        self.wind_y_next_arr = Array2D(self.gpu_stream,
                                       self.wind.wind_v[t1_index].shape[1], self.wind.wind_v[t1_index].shape[0],
                                       0, 0,
                                       self.wind.wind_v[t1_index], padded=False)

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.passiveDrifterKernel = GPUHandler(self.drift_kernels, "passiveDrifterKernel",
                                               "iifffiiPiPiPiPiiiiPifPPPPff")
        self.enforceBoundaryConditionsKernel = GPUHandler(self.drift_kernels, "enforceBoundaryConditions", "ffiiiPi")
        if self.wind_drift_factor:
            # Initialize wind parameters
            self.wind_timestamps = {}

            self.update_wind(self.drift_kernels, 0.0)

        self.local_size = (self.block_width, self.block_height, 1)
        self.global_size = (int(np.ceil((self.getNumDrifters() + 2) / float(self.block_width))), 1)

        # Initialize drifters:
        if initialize:
            self.uniformly_distribute_drifters(initialization_cov_drifters=initialization_cov_drifters)

    def copy(self):
        """
        Makes an independent indentical copy of the current object
        """

        copyOfSelf = GPUDrifterCollection(self.gpu_ctx,
                                          self.getNumDrifters(),
                                          observation_variance=self.observation_variance,
                                          boundary_conditions=self.boundaryConditions,
                                          domain_size_x=self.domain_size_x,
                                          domain_size_y=self.domain_size_y,
                                          gpu_stream=self.gpu_stream,
                                          block_width=self.block_width)

        copyOfSelf.setDrifterPositions(self.getDrifterPositions())
        copyOfSelf.setObservationPosition(self.getObservationPosition())

        return copyOfSelf

    def update_wind(self, kernel_module: module_t, t: float) -> float:
        # Key used to access the hashmaps
        key = str(kernel_module)

        # Compute new t0 and t1
        t_max_index = len(self.wind.t) - 1
        t0_index = max(0, np.searchsorted(self.wind.t, t) - 1)
        t1_index = min(t_max_index, int(np.searchsorted(self.wind.t, t)))
        new_t0 = self.wind.t[t0_index]
        new_t1 = self.wind.t[t1_index]

        # Find the old (and update)
        old_t0 = None
        old_t1 = None
        if key in self.wind_timestamps:
            old_t0 = self.wind_timestamps[key][0]
            old_t1 = self.wind_timestamps[key][1]
        self.wind_timestamps[key] = [new_t0, new_t1]

        # If time interval has changed, upload new data
        if new_t0 != old_t0:
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.wind_x_current_arr.upload(self.gpu_stream, self.wind.wind_u[t0_index])
            self.wind_y_current_arr.upload(self.gpu_stream, self.wind.wind_v[t0_index])
            self.gpu_ctx.synchronize()

        if new_t1 != old_t1:
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.wind_x_next_arr.upload(self.gpu_stream, self.wind.wind_u[t1_index])
            self.wind_y_next_arr.upload(self.gpu_stream, self.wind.wind_v[t1_index])
            self.gpu_ctx.synchronize()

        # Compute the wind_stress_t linear interpolation coefficient
        wind_t = 0.0
        elapsed_since_t0 = (t - new_t0)
        time_interval = max(1.0e-10, (new_t1 - new_t0))
        wind_t = max(0.0, min(1.0, elapsed_since_t0 / time_interval))

        return wind_t

    def setDrifterPositions(self, new_drifter_positions):
        ### Need to attach the observation to the new_drifter_positions, and then upload
        # to the GPU
        new_positions_all = np.concatenate((new_drifter_positions, np.array([self.getObservationPosition()])),
                                           axis=0)
        # print new_positions_all
        self.driftersDevice.upload(self.gpu_stream, new_positions_all)

    def setObservationPosition(self, new_observation_position):
        new_positions_all = np.concatenate((self.getDrifterPositions(), np.array([new_observation_position])))
        self.driftersDevice.upload(self.gpu_stream, new_positions_all)

    def setSensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def getDrifterPositions(self):
        all_drifters = self.driftersDevice.download(self.gpu_stream)
        return all_drifters[:-1, :]

    def getObservationPosition(self):
        all_drifters = self.driftersDevice.download(self.gpu_stream)
        return all_drifters[self.obs_index, :]

    def driftFromSim(self, sim, dt: float):
        self.drift(sim.gpu_data.h0, sim.gpu_data.hu0,
                   sim.gpu_data.hv0,
                   sim.bathymetry.Bm,
                   sim.nx, sim.ny, sim.t, sim.dx, sim.dy,
                   dt,
                   2, 2)

    def drift(self, eta: Array2D, hu: Array2D, hv: Array2D, Hm: Array2D, nx: int, ny: int, t: float, dx: float,
              dy: float, dt: float,
              x_zero_ref: int, y_zero_ref: int):
        if self.wind_drift_factor:
            wind_t = self.update_wind(self.drift_kernels, t)
        else:
            wind_t = 0.0
        self.passiveDrifterKernel.async_call(self.global_size, self.local_size, self.gpu_stream,
                                             [nx, ny, dx, dy, dt, x_zero_ref, y_zero_ref,
                                              eta.pointer, eta.pitch,
                                              hu.pointer, hu.pitch,
                                              hv.pointer, hv.pitch,
                                              Hm.pointer, Hm.pitch,
                                              int(self.boundaryConditions.isPeriodicNorthSouth()),
                                              int(self.boundaryConditions.isPeriodicEastWest()),
                                              self.getNumDrifters(),
                                              self.driftersDevice.pointer,
                                              self.driftersDevice.pitch,
                                              self.sensitivity,
                                              self.wind_x_current_arr.pointer,
                                              self.wind_x_next_arr.pointer,
                                              self.wind_y_current_arr.pointer,
                                              self.wind_y_next_arr.pointer,
                                              wind_t, self.wind_drift_factor])

    def setGPUStream(self, gpu_stream):
        self.gpu_stream = gpu_stream

    def cleanUp(self):
        if self.driftersDevice is not None:
            self.driftersDevice.release()
        self.gpu_ctx = None

    def enforceBoundaryConditions(self):
        if self.boundaryConditions.isPeriodicNorthSouth or self.boundaryConditions.isPeriodicEastWest:
            self.enforceBoundaryConditionsKernel.async_call(self.global_size, self.local_size, self.gpu_stream,
                                                            [self.domain_size_x,
                                                             self.domain_size_y,
                                                             int(self.boundaryConditions.isPeriodicNorthSouth()),
                                                             int(self.boundaryConditions.isPeriodicEastWest()),
                                                             self.numDrifters,
                                                             self.driftersDevice.pointer,
                                                             self.driftersDevice.pitch])
