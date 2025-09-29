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
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from gpuocean.utils.gpu import Array2D, GPUStream


class BaseRandomNumbers(ABC):
    """
    A class for generating XORWOW for pycuda and HIP.
    """

    def __init__(self, seed=None, xorwow_seed: np.random.SeedSequence = None):
        self._xorwow_seed = xorwow_seed
        if xorwow_seed is None:
            self._xorwow_seed = seed

    @abstractmethod
    def fill_normal(self, array: Array2D, gpu_stream: GPUStream = None):
        """
        Fills an array with random numbers using a standard normal distribution.
        Args:
            array: 2D device array to fill with random numbers.
            gpu_stream: (Optional) The stream for the GPU.
        """


    @abstractmethod
    def fill_uniform(self, array: Array2D, gpu_stream: GPUStream = None):
        """
        Fills an array with random numbers using a uniform distribution.
        Args:
            array: 2D device array to fill with random numbers
            gpu_stream: (Optional) The stream for the GPU.
        """
