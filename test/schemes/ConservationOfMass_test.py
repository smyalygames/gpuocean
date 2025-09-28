# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2022 SINTEF Digital

This python module implements regression tests conservation of mass
related to Kelvin waves in a domain with mixed boundary conditions.

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

import typing
import unittest
import gc

from testUtils import *

from gpuocean.SWEsimulators import CDKLM16, KP07, FBL, CTCS
from gpuocean.utils.Common import BoundaryConditions, BoundaryType
from gpuocean.utils.gpu import KernelContext

class ConservationOfMassTest(unittest.TestCase):
    def setUp(self):
        self.gpu_ctx = KernelContext()

        self.sim_args = {
            "gpu_ctx": self.gpu_ctx,
            "nx": int(500), "ny": 200,
            "dx": 10000.0, "dy": 10000.0,
            "dt": 50,
            "g": 9.81,
            "f": 1.2e-4,
            "coriolis_beta": 0.0,
            "r": 0.0,
            "boundary_conditions": BoundaryConditions(BoundaryType.WALL, BoundaryType.PERIODIC,
                                                      BoundaryType.WALL, BoundaryType.PERIODIC)
        }
        self.ctcs_args = {
            "A": 50
        }
        self.depth = 100
        self.rossby_radius = np.sqrt(self.sim_args["g"]*self.depth)/self.sim_args["f"]
        self.phase_speed = np.sqrt(self.sim_args['g']*self.depth)
        self.period = self.sim_args['dx']*self.sim_args['nx']/self.phase_speed
        self.geoconst = self.sim_args['g']*self.depth/self.sim_args['f']
        
        self.T = 50
        self.total_periodes = 2 # Number of simulated periods
        self.sub_dt = self.total_periodes*self.period/(self.sim_args['dt']*(self.T-1))

        self.sim = None

        self.initMass = 0.0

    def tearDown(self):
        if self.sim is not None:
            self.sim.cleanUp()
            self.sim = None

        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.Hi = None

        if self.sim_args["gpu_ctx"] is not None:
            self.gpu_ctx = None

        if self.gpu_ctx is not None:
            # TODO Check if this is broken or what value it should be
            # self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
        gc.collect() # Force run garbage collection to free up memory


    def setupSimAndRun(self, sim_name):

        ghosts = [2,2,2,2]
        if sim_name == "FBL" or sim_name == "CTCS":
            ghosts = [1,1,1,1]

        dataShape = (self.sim_args["ny"] + ghosts[0]+ghosts[2], 
                     self.sim_args["nx"] + ghosts[1]+ghosts[3])

        eta = np.zeros(dataShape, dtype=np.float32, order='C')
        hu, hv, H = None, None, None
        if sim_name == "FBL":
            hu = np.zeros((dataShape[0], dataShape[1]-1), dtype=np.float32)
            hv = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32)  
            H = np.ones(dataShape, dtype=np.float32)*self.depth
        elif sim_name == "CTCS":
            hu = np.zeros((dataShape[0], dataShape[1]+1), dtype=np.float32)
            hv = np.zeros((dataShape[0]+1, dataShape[1]), dtype=np.float32)  
            H = np.ones(dataShape, dtype=np.float32)*self.depth
        else:
            hu = np.zeros(dataShape, dtype=np.float32, order='C')
            hv = np.zeros(dataShape, dtype=np.float32, order='C')
            H = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32)*self.depth

        self.init_eta(eta, ghosts, self.rossby_radius, self.sim_args)
        self.init_hu( hu,  ghosts, self.rossby_radius, self.geoconst, self.sim_args)
        
        self.initMass = np.sum(eta[ghosts[0]:-ghosts[2], ghosts[1]:-ghosts[3]])

        init_args =  {
            "eta0" : eta,
            "hu0"  : hu,
            "hv0"  : hv,
            "H"    : H
        }
        
        accuracy = 5
        if sim_name == "FBL":
            self.sim = FBL.FBL(**self.sim_args, **init_args)
            accuracy = 3
        elif sim_name == "CTCS":
            self.sim = CTCS.CTCS(**self.sim_args, **init_args, **self.ctcs_args)
        elif sim_name == "KP":
            self.sim = KP07.KP07(**self.sim_args, **init_args)
        elif sim_name == "CDKLM":
            self.sim = CDKLM16.CDKLM16(**self.sim_args, **init_args)
        
        for k in range(self.T):
            if k > 0:
                t = self.sim.step(self.sub_dt*self.sim.dt)
            else:
                t = 0.0

            eta1, u1, v1 = self.sim.download(interior_domain_only=True)
            mass = eta1.sum()
            relDiff = (mass-self.initMass)/self.initMass
            self.assertAlmostEqual(0.0, relDiff, places=accuracy,
                               msg='Unexpected mass difference sim '+sim_name+' iteration '+str(k)+'! Max rel diff: ' + str(relDiff) + ', diff: ' + str(mass-self.initMass))
    
    
    def eta_cell(self, r_i: float, r_j: float, rossby_radius: float, coriolis: bool) -> float:
        f_func = 1.0 + np.tanh((-r_i + rossby_radius)/(rossby_radius/3))

        if coriolis:
            return np.exp(-r_j/rossby_radius)*f_func
        else:
            return f_func
        
    def init_eta(self, eta: npt.NDArray, ghost_cells: list[int], rossby_radius: float, sim_args: dict[str, typing.Any], coriolis=True):
        ny, nx = eta.shape
        x_0 = nx/2
        y_0 = ghost_cells[2]-0.5
        for j in range(0, ny):
            for i in range(0, nx):
                r_j = np.sqrt((j-y_0)**2)*sim_args["dy"]
                r_i = np.sqrt((i-x_0)**2)*sim_args["dx"]
                
                eta[j,i] = self.eta_cell(r_i, r_j, rossby_radius, coriolis)
                
                
    def init_hu(self, hu: npt.NDArray, ghost_cells: list[int], rossby_radius: float, geoconst: float, sim_args: dict[str, typing.Any], coriolis=True):
        if not coriolis:
            return
        ny, nx = hu.shape
        x_0 = nx/2
        y_0 = ghost_cells[2]-0.5
        for j in range(0, ny):
            for i in range(0, nx):
                r_j = np.sqrt((j-y_0)**2)*sim_args["dy"]
                r_i = np.sqrt((i-x_0)**2)*sim_args["dx"]
                
                eta_c = self.eta_cell(r_i, r_j, rossby_radius, coriolis)
                hu[j,i] = -geoconst*(-1.0/rossby_radius)*np.sign(j-y_0)*eta_c

    def test_conservationOfMass_FBL_KelvinWaves(self):
        self.setupSimAndRun("FBL")

    def test_conservationOfMass_CTCS_KelvinWaves(self):
        self.setupSimAndRun("CTCS")

    # FIXME causes fatal error for the next test
    def notest_conservationOfMass_KP_KelvinWaves(self):
        self.setupSimAndRun("KP")

    def test_conservationOfMass_CDKLM_KelvinWaves(self):
        self.setupSimAndRun("CDKLM")

        