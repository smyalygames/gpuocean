from __future__ import annotations
from typing import TYPE_CHECKING
import math

import numpy as np
from hip import hiprand

from ..random_numbers import BaseRandomNumbers
from ...hip_utils import hip_check

if TYPE_CHECKING:
    from ... import Array2D, GPUStream


class HIPRandomNumbers(BaseRandomNumbers):
    def __init__(self, seed=None, xorwow_seed: np.random.SeedSequence = None):
        super().__init__(seed, xorwow_seed)

        self.generator: hiprand.rocrand_generator_base_type = hip_check(
            hiprand.hiprandCreateGenerator(hiprand.hiprandRngType_t.HIPRAND_RNG_PSEUDO_XORWOW))

        if self._xorwow_seed is not None:
            hip_check(hiprand.hiprandSetPseudoRandomGeneratorSeed(self.generator, self._xorwow_seed))

    def __del__(self):
        hip_check(hiprand.hiprandDestroyGenerator(self.generator))

    def fill_normal(self, array: Array2D, gpu_stream: GPUStream = None):
        # Set the GPU Stream
        hip_check(hiprand.hiprandSetStream(self.generator, gpu_stream.pointer))

        # Parameters for generator
        elements = math.prod(array.shape)
        pointer = array.pointer
        mean = 0
        stddev = 1

        # Check if the array is 64/32-bit and generate with normal distribution
        if array.double_precision:
            hip_check(hiprand.hiprandGenerateNormalDouble(self.generator, pointer, elements, mean, stddev))
        else:
            hip_check(hiprand.hiprandGenerateNormal(self.generator, pointer, elements, mean, stddev))

    def fill_uniform(self, array: Array2D, gpu_stream: GPUStream = None):
        # Set the GPU Stream
        hip_check(hiprand.hiprandSetStream(self.generator, gpu_stream.pointer))

        # Parameters for generator
        elements = math.prod(array.shape)
        pointer = array.pointer

        # Check if the array is 64/32-bit and generate uniform
        if array.double_precision:
            hip_check(hiprand.hiprandGenerateUniformDouble(self.generator, pointer, elements))
        else:
            hip_check(hiprand.hiprandGenerateUniform(self.generator, pointer, elements))
