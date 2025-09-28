# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements a Ensemble of particles living on the GPU,
each consisting of a single drifter in its own ocean state. The 
perturbation parameter is the wind direction.


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

from gpuocean.SWEsimulators import CDKLM16
from gpuocean.ensembles.BaseDrifterEnsemble import BaseDrifterEnsemble
from gpuocean.drifters.GPUDrifterCollection import GPUDrifterCollection

if TYPE_CHECKING:
    from gpuocean.utils.gpu import KernelContext


class DrifterEnsemble(BaseDrifterEnsemble):

    def __init__(self, gpu_ctx: KernelContext, num_particles: int, observation_variance=0.0):

        super(DrifterEnsemble, self).__init__(num_particles,
                                              observation_variance)

        self.gpu_ctx = gpu_ctx

    # ---------------------------------------
    # Implementing abstract function
    # ---------------------------------------
    def init(self):

        self.sim = CDKLM16.CDKLM16(self.gpu_ctx,
                                   self.base_eta, self.base_hu, self.base_hv,
                                   self.base_H,
                                   self.nx, self.ny, self.dx, self.dy, self.dt,
                                   self.g, self.f, self.r,
                                   wind=self.wind,
                                   boundary_conditions=self.boundaryConditions,
                                   write_netcdf=False)

        # TO CHECK! Is it okay to have drifters as self.drifters - in order to easier reach its member functions?
        self.drifters = GPUDrifterCollection(self.gpu_ctx, self.numParticles,
                                             observation_variance=self.observation_variance,
                                             boundary_conditions=self.boundaryConditions,
                                             domain_size_x=self.nx * self.dx,
                                             domain_size_y=self.ny * self.dy)

        self.drifters.initializeUniform()
        self.sim.attachDrifters(self.drifters)

    def cleanUp(self):
        if self.sim is not None:
            self.sim.cleanUp()
        if self.drifters is not None:
            self.drifters.cleanUp()
        self.gpu_ctx = None
