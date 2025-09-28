from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gpuocean.utils.Common import BoundaryConditions, BoundaryType
from .. import GPUHandler, Array2D

if TYPE_CHECKING:
    from .. import KernelContext, GPUStream

class Bathymetry:
    """
    Class for holding bathymetry defined on cell intersections (cell corners) and reconstructed on
    cell mid-points.
    """

    def __init__(self, gpu_ctx: KernelContext, gpu_stream: GPUStream,
                 nx: int, ny: int, halo_x: int, halo_y: int,
                 Bi_host: Union[npt.NDArray, np.ma.MaskedArray],
                 boundary_conditions=BoundaryConditions(),
                 block_width=16, block_height=16):
        # Convert scalar data to int32
        self.gpu_stream = gpu_stream
        self.nx = nx
        self.ny = ny
        self.halo_x = halo_x
        self.halo_y = halo_y
        self.halo_nx = nx + 2 * halo_x
        self.halo_ny = ny + 2 * halo_y
        self.boundary_conditions = boundary_conditions

        # Set land value (if masked array)
        self.mask_value = 1.0e20
        self.use_mask = False
        if np.ma.is_masked(Bi_host):
            Bi_host = Bi_host.copy().filled(self.mask_value).astype(np.float32)
            self.use_mask = True

        # Check that Bi has the size corresponding to number of cell intersections
        BiShapeY, BiShapeX = Bi_host.shape
        assert (BiShapeX == nx + 1 + 2 * halo_x and BiShapeY == ny + 1 + 2 * halo_y), \
            "Wrong size of bottom bathymetry, should be defined on cell intersections, not cell centers. " + \
            str((BiShapeX, BiShapeY)) + " vs " + str((nx + 1 + 2 * halo_x, ny + 1 + 2 * halo_y))

        # Upload Bi to device
        self.Bi = Array2D(gpu_stream, nx + 1, ny + 1, halo_x, halo_y, Bi_host)

        # Define OpenCL parameters
        self.local_size = (block_width, block_height, 1)
        self.global_size = (
            int(np.ceil((self.halo_nx + 1) / float(self.local_size[0]))),
            int(np.ceil((self.halo_ny + 1) / float(self.local_size[1])))
        )

        # Check boundary conditions and make Bi periodic if necessary
        # Load CUDA module for periodic boundary
        self.boundaryKernels = gpu_ctx.get_kernel("boundary_kernels",
                                                  defines={'block_width': block_width, 'block_height': block_height,
                                                           # setting dummy variables as they need to be defined
                                                           'BC_NS_NX': -1,
                                                           'BC_NS_NY': -1,
                                                           'BC_EW_NX': -1,
                                                           'BC_EW_NY': -1,
                                                           })

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.periodic_boundary_intersections_NS = GPUHandler(self.boundaryKernels,
                                                             "periodic_boundary_intersections_NS", "iiiiPi")
        self.periodic_boundary_intersections_EW = GPUHandler(self.boundaryKernels,
                                                             "periodic_boundary_intersections_EW", "iiiiPi")
        self.closed_boundary_intersections_NS = GPUHandler(self.boundaryKernels, "closed_boundary_intersections_NS",
                                                           "iiiiPi")
        self.closed_boundary_intersections_EW = GPUHandler(self.boundaryKernels, "closed_boundary_intersections_EW",
                                                           "iiiiPi")

        self._boundaryConditions()

        # Allocate Bm
        Bm_host = np.zeros((self.halo_ny, self.halo_nx), dtype=np.float32, order='C')
        self.Bm = Array2D(gpu_stream, nx, ny, halo_x, halo_y, Bm_host)

        # Load kernel for finding Bm from Bi
        self.initBm_kernel = gpu_ctx.get_kernel("initHm_kernel",
                                                defines={'block_width': block_width, 'block_height': block_height})

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.initBm = GPUHandler(self.initBm_kernel, "initBm", "iiPifPi")
        self.waterElevationToDepth = GPUHandler(self.initBm_kernel, "waterElevationToDepth", "iiPiPi")

        # Call kernel
        self.initBm.async_call(self.global_size, self.local_size, self.gpu_stream,
                               [self.halo_nx, self.halo_ny,
                                self.Bi.pointer, self.Bi.pitch,
                                self.mask_value,
                                self.Bm.pointer, self.Bm.pitch])

    def download(self, gpu_stream: GPUStream) -> (tuple[npt.NDArray, npt.NDArray] |
                                                  tuple[np.ma.MaskedArray, np.ma.MaskedArray]):
        Bm_cpu = self.Bm.download(gpu_stream)
        Bi_cpu = self.Bi.download(gpu_stream)

        # Mask land values in output
        if self.use_mask:
            Bi_cpu = np.ma.array(data=Bi_cpu, mask=(Bi_cpu == self.mask_value), fill_value=0.0)
            Bm_cpu = np.ma.array(data=Bm_cpu, mask=(Bm_cpu == self.mask_value), fill_value=0.0)

        return Bi_cpu, Bm_cpu

    def release(self) -> None:
        """
        Frees the allocated memory buffers on the GPU
        """
        self.Bm.release()
        self.Bi.release()

    # Transforming water elevation into water depth
    def waterElevationToDepth(self, h: Array2D) -> None:

        assert ((h.ny_halo, h.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "h0 not the correct shape: " + str(h.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))

        # Call kernel
        self.waterElevationToDepth.async_call(self.global_size, self.local_size, self.gpu_stream,
                                              [self.halo_nx, self.halo_ny,
                                               h.pointer, h.pitch,
                                               self.Bm.pointer, self.Bm.pitch])

    # Transforming water depth into water elevation
    def waterDepthToElevation(self, w: Array2D, h: Array2D) -> None:

        assert ((h.ny_halo, h.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "h0 not the correct shape: " + str(h.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))
        assert ((w.ny_halo, w.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "w not the correct shape: " + str(w.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))
        # Call kernel
        self.waterDepthToElevation.async_call(self.global_size, self.local_size, self.gpu_stream,
                                              [self.halo_nx, self.halo_ny,
                                               w.pointer, w.pitch,
                                               h.pointer, h.pitch,
                                               self.Bm.pointer, self.Bm.pitch])

    def _boundaryConditions(self) -> None:
        args = (self.global_size, self.local_size, self.gpu_stream,
                [self.nx, self.ny, self.halo_x, self.halo_y,
                 self.Bi.pointer, self.Bi.pitch])

        # North-south:
        if (self.boundary_conditions.north == BoundaryType.PERIODIC) and (self.boundary_conditions.south == BoundaryType.PERIODIC):
            self.periodic_boundary_intersections_NS.async_call(*args)
        elif (self.boundary_conditions.north == BoundaryType.WALL) and (self.boundary_conditions.south == BoundaryType.WALL):
            self.closed_boundary_intersections_NS.async_call(*args)

        # East-west:
        if (self.boundary_conditions.east == BoundaryType.PERIODIC) and (self.boundary_conditions.west == BoundaryType.PERIODIC):
            self.periodic_boundary_intersections_EW.async_call(*args)
        elif (self.boundary_conditions.north == BoundaryType.WALL) and (self.boundary_conditions.south == BoundaryType.WALL):
            self.closed_boundary_intersections_EW.async_call(*args)
