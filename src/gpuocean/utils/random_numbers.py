# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019, 2024  SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This class implements some simple random number generators on the GPU

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
import gc

import numpy as np

from gpuocean.utils.gpu import GPUHandler, Array2D, XORWOWHandler

if TYPE_CHECKING:
    from gpuocean.utils.gpu import KernelContext, GPUStream


class RandomNumbers(object):
    """
    Class for generating random numbers within GPU Ocean.

    Arrays for holding the random numbers must be held by other objects, and only objects related to
    the random number generation itself is held by this class.
    """

    def __init__(self, gpu_ctx: KernelContext, gpu_stream: GPUStream, nx: int, ny: int,
                 use_lcg=False,
                 seed=None, xorwow_seed: np.random.SeedSequence = None,
                 block_width=16, block_height=16):
        """
        Class for generating random numbers within GPU Ocean.

        Args:
            ny: y-shape of the random number array that will be generated
            nx: x-shape of the random number array that will be generated
            use_lcg: LCG is a linear algorithm for generating a series of pseudo-random numbers
            seed: Integer seed for the random number generators in this class. If we use LCG, this seed is
                passed to numpy before generating one seeded seed per thread. If we use curand (XORWOW),
                and xorwow_seed is None, we seed the XORWOW generator based on this seed.
            xorwow_seed: Seed for the curand XORWOW random number generator, same value as 'seed' by default
            block_width: The size of the width in the GPU block
            block_height: The size of the height in the GPU block
        """

        self.use_lcg = use_lcg
        self.gpu_stream = gpu_stream

        # Set numpy random state
        self.random_state = np.random.RandomState(seed=seed)

        # Make sure that all variables initialized within ifs are defined
        self.rng = None
        self.seed = None
        self.host_seed = None

        self.nx = nx
        self.ny = ny

        # Since normal distributed numbers are generated in pairs, we need to store half the number of
        # seed values compared to the number of random numbers.
        # This split is in x-direction, and the dimension in y is kept as is
        self.seed_ny = self.ny
        self.seed_nx = int(np.ceil(self.nx / 2))

        # Generate seed:
        self.floatMax = float(np.iinfo(np.int32).max)
        if self.use_lcg:
            self.host_seed = self.random_state.rand(self.seed_ny, self.seed_nx) * self.floatMax
            self.host_seed = self.host_seed.astype(np.uint64, order='C')
            self.seed = Array2D(self.gpu_stream, self.seed_nx, self.seed_ny, 0, 0, self.host_seed,
                                double_precision=True, integers=True)
        else:
            self.rng = XORWOWHandler(seed, xorwow_seed)

        # Generate kernels
        self.kernels = gpu_ctx.get_kernel("random_number_generators",
                                          defines={'block_width': block_width, 'block_height': block_height,
                                                   },
                                          compile_args={
                                              'options': ["--use_fast_math",
                                                          "--maxrregcount=32"]
                                          })

        # Get CUDA functions and define data types for prepared_{async_}call()
        # Generate kernels
        self.uniformDistributionKernel = GPUHandler(self.kernels, "uniformDistribution", "iiiPiPi")

        self.normalDistributionKernel = None
        if self.use_lcg:
            self.normalDistributionKernel = GPUHandler(self.kernels, "normalDistribution", "iiiPiPi")

        # Compute kernel launch parameters
        self.local_size = (block_width, block_height, 1)

        # Launch one thread for each seed, which in turns generates two iid N(0,1)
        self.global_size_random_numbers = (
            int(np.ceil(self.seed_nx / float(self.local_size[0]))),
            int(np.ceil(self.seed_ny / float(self.local_size[1])))
        )

    def __del__(self):
        self.cleanUp()

    def cleanUp(self):
        if self.seed is not None:
            self.seed.release()
        self.gpu_ctx = None
        gc.collect()

    def getSeed(self):
        assert self.use_lcg, "getSeed is only valid if LCG is used as pseudo-random generator."

        return self.seed.download(self.gpu_stream)

    def resetSeed(self):
        assert self.use_lcg, "resetSeed is only valid if LCG is used as pseudo-random generator."

        # Generate seed:
        self.host_seed = self.random_state.rand(self.seed_ny, self.seed_nx) * self.floatMax
        self.host_seed = self.host_seed.astype(np.uint64, order='C')
        self.seed.upload(self.gpu_stream, self.host_seed)

    def _check_input(self, random_numbers):
        if not isinstance(random_numbers, Array2D):
            raise TypeError(f"Expected random_numbers of type Array2D but got {str(type(random_numbers))}")
        shape_input = (random_numbers.ny, random_numbers.nx)
        shape_expected = (self.ny, self.nx)
        if shape_input != shape_expected:
            raise ValueError(f"Expected random_numbers with shape {shape_expected} but got {shape_input}")

    def generateNormalDistribution(self, random_numbers: Array2D):
        self._check_input(random_numbers)
        if not self.use_lcg:
            self.rng.fill_normal(random_numbers, gpu_stream=self.gpu_stream)
        else:
            self.normalDistributionKernel.async_call(self.global_size_random_numbers, self.local_size, self.gpu_stream,
                                                     [self.seed_nx, self.seed_ny,
                                                      self.nx,
                                                      self.seed.pointer, self.seed.pitch,
                                                      random_numbers.pointer, random_numbers.pitch])

    def generateUniformDistribution(self, random_numbers: Array2D):
        self._check_input(random_numbers)
        if not self.use_lcg:
            self.rng.fill_uniform(random_numbers, gpu_stream=self.gpu_stream)
        else:
            self.uniformDistributionKernel.async_call(self.global_size_random_numbers, self.local_size, self.gpu_stream,
                                                      [self.seed_nx, self.seed_ny,
                                                       self.nx,
                                                       self.seed.pointer, self.seed.pitch,
                                                       random_numbers.pointer, random_numbers.pitch])
