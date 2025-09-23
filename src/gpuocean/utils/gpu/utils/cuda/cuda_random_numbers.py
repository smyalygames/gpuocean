from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pycuda.gpuarray
from pycuda.curandom import XORWOWRandomNumberGenerator

from ..random_numbers import BaseRandomNumbers

if TYPE_CHECKING:
    from ... import Array2D, GPUStream


class CudaRandomNumbers(BaseRandomNumbers):
    def __init__(self, seed=None, xorwow_seed: np.random.SeedSequence = None):
        super().__init__(seed, xorwow_seed)

        if xorwow_seed is not None:
            def set_seeder(shape: tuple[int, ...], seed_sequence: np.random.SeedSequence):
                seed_arr = pycuda.gpuarray.ones_like(pycuda.gpuarray.zeros(shape, dtype=np.int32), dtype=np.int32) * seed_sequence
                return seed_arr

            self.rng = XORWOWRandomNumberGenerator(lambda shape: set_seeder(shape, xorwow_seed))
        else:
            self.rng = XORWOWRandomNumberGenerator()

    def fill_normal(self, array: Array2D, gpu_stream: GPUStream = None):
        self.rng.fill_normal(array.pointer, stream=gpu_stream.pointer)

    def fill_uniform(self, array: Array2D, gpu_stream: GPUStream = None):
        self.rng.fill_uniform(array.pointer, stream=gpu_stream.pointer)
