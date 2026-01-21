"""
This software is part of GPU Ocean.

Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute
Copyright (C) 2025 Anthony Berg

This python module implements the different helper functions and
classes that are shared through out all elements of the code.

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

from typing import TypeVar, Union

import numpy as np
import numpy.typing as npt

from .. import Array2D, GPUStream

T = TypeVar('T', np.float32, np.float64)
data_t = Union[npt.NDArray[T], np.ma.MaskedArray]


class SWEDataArakawaA:
    """
    A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
    """

    def __init__(self, gpu_stream: GPUStream, nx: int, ny: int, halo_x: int, halo_y: int, h0: data_t, hu0: data_t,
                 hv0: data_t):
        """
        Uploads initial data to the GPU device
        """
        self.h0 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu0 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, hu0)
        self.hv0 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, hv0)

        self.h1 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu1 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, hu0)
        self.hv1 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, hv0)

    def swap(self) -> None:
        """
        Swaps the variables after a timestep has been completed
        """
        self.h1, self.h0 = self.h0, self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1

    def download(self, gpu_stream: GPUStream):
        """
        Enables downloading data from CUDA device to Python
        """
        h_cpu = self.h0.download(gpu_stream)
        hu_cpu = self.hu0.download(gpu_stream)
        hv_cpu = self.hv0.download(gpu_stream)

        return h_cpu, hu_cpu, hv_cpu

    @property
    def arrays(self) -> tuple[Array2D, Array2D, Array2D, Array2D, Array2D, Array2D]:
        """
        Gets all the arrays used in this object.
        """
        return self.h0, self.hu0, self.hv0, self.h1, self.hu1, self.hv1

    def release(self) -> None:
        """
        Frees the allocated memory buffers on the GPU
        """
        for array in self.arrays:
            array.release()


class SWEDataArakawaC:
    """
    A class representing an Arakawa C type (staggered, u fluxes on east/west faces, v fluxes on north/south faces) grid
    We use h as cell centers
    """

    def __init__(self, gpu_stream: GPUStream, nx: int, ny: int, halo_x: int, halo_y: int, h0: data_t, hu0: data_t,
                 hv0: data_t, fbl=False):
        """
        Uploads initial data to the GPU device
        asymHalo needs to be on the form [north, east, south, west]
        """
        # FIXME: This at least works for 0 and 1 ghost cells, but not convinced it generalizes
        if halo_x > 1 and halo_y > 1:
            raise ValueError("There can only be 0 or 1 ghost cells.")

        self.fbl = fbl

        hv_ny = ny + 1
        if fbl:
            hu_nx = nx - 1
        else:
            hu_nx = nx + 1

        self.h0 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu0 = Array2D(gpu_stream, hu_nx, ny, halo_x, halo_y, hu0)
        self.hv0 = Array2D(gpu_stream, nx, hv_ny, halo_x, halo_y, hv0)

        self.h1 = Array2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu1 = Array2D(gpu_stream, hu_nx, ny, halo_x, halo_y, hu0)
        self.hv1 = Array2D(gpu_stream, nx, hv_ny, halo_x, halo_y, hv0)

    def swap(self) -> None:
        """
        Swaps the variables after a timestep has been completed
        """
        # h is assumed to be constant (bottom topography really)
        self.h1, self.h0 = self.h0, self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1

    def download(self, gpu_stream: GPUStream, interior_domain_only=False):
        """
        Enables downloading data from GPU device to Python (CPU)
        """
        h_cpu = self.h0.download(gpu_stream)
        hu_cpu = self.hu0.download(gpu_stream)
        hv_cpu = self.hv0.download(gpu_stream)

        if interior_domain_only and self.fbl:
            # print("Sneaking in some FBL specific functionality")
            return h_cpu[1:-1, 1:-1], hu_cpu[1:-1, :], hv_cpu[1:-1, 1:-1]

        return h_cpu, hu_cpu, hv_cpu

    def download_prev_timestep(self, gpu_stream: GPUStream):
        """
        Enables downloading data from the additional buffer of the GPU device to Python (CPU)
        """
        h_cpu = self.h1.download(gpu_stream)
        hu_cpu = self.hu1.download(gpu_stream)
        hv_cpu = self.hv1.download(gpu_stream)

        return h_cpu, hu_cpu, hv_cpu

    @property
    def arrays(self) -> tuple[Array2D, Array2D, Array2D, Array2D, Array2D, Array2D]:
        """
        Gets all the arrays used in this object.
        """
        return self.h0, self.hu0, self.hv0, self.h1, self.hu1, self.hv1

    def release(self) -> None:
        """
        Frees the allocated memory buffers on the GPU
        """
        for array in self.arrays:
            array.release()
