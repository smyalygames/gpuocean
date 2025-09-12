from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np

from gpuocean.utils.Common import BoundaryConditionsData

from .. import Array3D, GPUHandler

if TYPE_CHECKING:
    from gpuocean.utils.Common import BoundaryConditions

    from .. import KernelContext, GPUStream


class BoundaryConditionsArakawaA:
    """
    Class that checks boundary conditions and calls the required kernels for Arakawa A type grids.
    """

    def __init__(self, gpu_ctx: KernelContext, gpu_stream: GPUStream,
                 nx: int, ny: int,
                 halo_x: int, halo_y: int,
                 boundary_conditions: BoundaryConditions,
                 bc_data=BoundaryConditionsData(),
                 block_width=16, block_height=16):
        self.logger = logging.getLogger(__name__)

        self.boundary_conditions = boundary_conditions

        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.halo_x = np.int32(halo_x)
        self.halo_y = np.int32(halo_y)
        self.bc_data = bc_data
        # print("boundary (ny, nx: ", (self.ny, self.nx))
        # print("boundary (halo_y, halo_x): ", (self.halo_y, self.halo_x))
        # print("numerical sponge cells (n,e,s,w): ", self.boundary_conditions.spongeCells)

        # Load CUDA module for periodic boundary
        self.boundaryKernels = gpu_ctx.get_kernel("boundary_kernels.cu",
                                                  defines={'block_width': block_width, 'block_height': block_height,
                                                           'BC_NS_NX': int(self.bc_data.north.h[0].shape[0]),
                                                           'BC_NS_NY': int(2),
                                                           'BC_EW_NX': int(2),
                                                           'BC_EW_NY': int(self.bc_data.east.h[0].shape[0]),
                                                           })

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.periodicBoundary_NS = GPUHandler(self.boundaryKernels, "periodicBoundary_NS", "iiiiPiPiPi")
        self.periodicBoundary_EW = GPUHandler(self.boundaryKernels, "periodicBoundary_EW", "iiiiPiPiPi")
        self.linearInterpolation_NS = GPUHandler(self.boundaryKernels, "linearInterpolation_NS", "iiiiiiiiPiPiPi")
        self.linearInterpolation_EW = GPUHandler(self.boundaryKernels, "linearInterpolation_EW", "iiiiiiiiPiPiPi")
        self.flowRelaxationScheme_NS = GPUHandler(self.boundaryKernels, "flowRelaxationScheme_NS", "iiiiiiiiPiPiPiPPf")
        self.flowRelaxationScheme_EW = GPUHandler(self.boundaryKernels, "flowRelaxationScheme_EW", "iiiiiiiiPiPiPiPPf")

        self.bc_timestamps = [None, None]

        # Set kernel launch parameters
        self.local_size = (block_width, block_height, 1)
        self.global_size = (
            int(np.ceil((self.nx + 2 * self.halo_x + 1) / float(self.local_size[0]))),
            int(np.ceil((self.ny + 2 * self.halo_y + 1) / float(self.local_size[1]))))
        components = 3
        self.bc_NS_current_arr = Array3D(gpu_stream,
                                         self.bc_data.north.h[0].shape[0], 2, components,
                                         np.zeros((2, self.bc_data.north.h[0].shape[0], components)))
        self.bc_NS_next_arr = Array3D(gpu_stream,
                                      self.bc_data.north.h[0].shape[0], 2, components,
                                      np.zeros((2, self.bc_data.north.h[0].shape[0], components)))
        self.bc_EW_current_arr = Array3D(gpu_stream,
                                         2, self.bc_data.east.h[0].shape[0], components,
                                         np.zeros((self.bc_data.east.h[0].shape[0], 2, components)))
        self.bc_EW_next_arr = Array3D(gpu_stream,
                                      2, self.bc_data.east.h[0].shape[0], components,
                                      np.zeros((self.bc_data.east.h[0].shape[0], 2, components)))

    """
    Function which updates the external solution for the boundary conditions
    """

    def update_bc_values(self, gpu_stream: GPUStream, t: float):
        # Only if we use flow relaxation
        if not (self.boundary_conditions.north == 3 or
                self.boundary_conditions.south == 3 or
                self.boundary_conditions.east == 3 or
                self.boundary_conditions.west == 3):
            return

        # Compute new t0 and t1
        t_max_index = len(self.bc_data.t) - 1
        t0_index = max(0, np.searchsorted(self.bc_data.t, t) - 1)
        t1_index = min(t_max_index, np.searchsorted(self.bc_data.t, t))
        new_t0 = self.bc_data.t[t0_index]
        new_t1 = self.bc_data.t[t1_index]

        # Find the old (and update)
        old_t0 = self.bc_timestamps[0]
        old_t1 = self.bc_timestamps[1]
        self.bc_timestamps = [new_t0, new_t1]

        # Log some debug info
        self.logger.debug(f"Times: {str(self.bc_data.t)}")
        self.logger.debug(f"Time indices: [{t0_index}, {t1_index}]")
        self.logger.debug(f"Time: {t}  New interval is [{new_t0}, {new_t1}], old was [{old_t0}, {old_t1}]")

        def pack_data_ns(data, t_index):
            h = data.h[t_index]
            hu = data.hu[t_index]
            hv = data.hv[t_index]

            h = np.squeeze(h)
            hu = np.squeeze(hu)
            hv = np.squeeze(hv)

            if len(h.shape) != 1:
                raise ValueError("NS-data must be one row")

            nx = h.shape[0]

            components = 3
            NS_data = np.vstack((h, hu, hv))
            NS_data = np.transpose(NS_data)
            NS_data = np.reshape(NS_data, (1, nx, components))
            NS_data = np.ascontiguousarray(NS_data)
            # print(NS_data)

            return NS_data

        def pack_data_ew(data, t_index):
            h = data.h[t_index]
            hu = data.hu[t_index]
            hv = data.hv[t_index]

            h = np.squeeze(h)
            hu = np.squeeze(hu)
            hv = np.squeeze(hv)
            assert (len(h.shape) == 1), "EW-data must be one column"
            ny = h.shape[0]

            components = 3
            EW_data = np.vstack((h, hu, hv))
            EW_data = np.transpose(EW_data)
            EW_data = np.reshape(EW_data, (ny, 1, components))
            EW_data = np.ascontiguousarray(EW_data)

            return EW_data

        def upload(name: str, t_index: int):
            gpu_stream.synchronize()
            self.logger.debug(f"Updating {name}")

            N_data = pack_data_ns(self.bc_data.north, t_index)
            S_data = pack_data_ns(self.bc_data.south, t_index)
            NS_data = np.vstack((S_data, N_data))
            NS_data = np.ascontiguousarray(NS_data)
            self.bc_NS_current_arr.upload(gpu_stream, NS_data)

            E_data = pack_data_ew(self.bc_data.east, t_index)
            W_data = pack_data_ew(self.bc_data.west, t_index)
            EW_data = np.hstack((W_data, E_data))
            EW_data = np.ascontiguousarray(EW_data)
            self.bc_EW_current_arr.upload(gpu_stream, EW_data)

            self.logger.debug(f"NS-Data is set to {str(NS_data)}, {str(NS_data.shape)}")
            self.logger.debug(f"EW-Data is set to {str(EW_data)}, {str(EW_data.shape)}")

            gpu_stream.synchronize()

        # If time interval has changed, upload new data
        if new_t0 != old_t0:
            upload("T0", t0_index)

        if new_t1 != old_t1:
            upload("T1", t1_index)

        # Update the bc_t linear interpolation coefficient
        elapsed_since_t0 = (t - new_t0)
        time_interval = max(1.0e-10, (new_t1 - new_t0))
        self.bc_t = np.float32(max(0.0, min(1.0, elapsed_since_t0 / time_interval)))
        self.logger.debug("Interpolation t is %f", self.bc_t)

    def boundaryCondition(self, gpu_stream, h, u, v):
        if self.boundary_conditions.north == 2:
            self.periodic_boundary_NS(gpu_stream, h, u, v)
        else:
            if self.boundary_conditions.north == 3 or self.boundary_conditions.south == 3:
                self.flow_relaxation_NS(gpu_stream, h, u, v)
            if self.boundary_conditions.north == 4 or self.boundary_conditions.south == 4:
                self.linear_interpolation_NS(gpu_stream, h, u, v)

        if self.boundary_conditions.east == 2:
            self.periodic_boundary_EW(gpu_stream, h, u, v)
        else:
            if self.boundary_conditions.east == 3 or self.boundary_conditions.west == 3:
                self.flow_relaxation_EW(gpu_stream, h, u, v)
            if self.boundary_conditions.east == 4 or self.boundary_conditions.west == 4:
                self.linear_interpolation_EW(gpu_stream, h, u, v)

    def periodic_boundary_NS(self, gpu_stream: GPUStream, h, u, v):
        self.__periodic_boundary(self.periodicBoundary_NS, gpu_stream, h, u, v)

    def periodic_boundary_EW(self, gpu_stream: GPUStream, h, v, u):
        self.__periodic_boundary(self.periodicBoundary_EW, gpu_stream, h, u, v)

    def __periodic_boundary(self, funct: GPUHandler, gpu_stream: GPUStream, h, u, v):
        funct.async_call(
            self.global_size, self.local_size, gpu_stream,
            [self.nx, self.ny,
            self.halo_x, self.halo_y,
            h.data.gpudata, h.pitch,
            u.data.gpudata, u.pitch,
            v.data.gpudata, v.pitch])

    def linear_interpolation_NS(self, gpu_stream, h, u, v):
        self.linearInterpolation_NS.async_call(
            self.global_size, self.local_size, gpu_stream,
            [self.boundary_conditions.north, self.boundary_conditions.south,
            self.nx, self.ny,
            self.halo_x, self.halo_y,
            self.boundary_conditions.spongeCells['north'],
            self.boundary_conditions.spongeCells['south'],
            h.data.gpudata, h.pitch,
            u.data.gpudata, u.pitch,
            v.data.gpudata, v.pitch])

    def linear_interpolation_EW(self, gpu_stream, h, u, v):
        self.linearInterpolation_EW.async_call(
            self.global_size, self.local_size, gpu_stream,
            [self.boundary_conditions.east, self.boundary_conditions.west,
            self.nx, self.ny,
            self.halo_x, self.halo_y,
            self.boundary_conditions.spongeCells['east'],
            self.boundary_conditions.spongeCells['west'],
            h.data.gpudata, h.pitch,
            u.data.gpudata, u.pitch,
            v.data.gpudata, v.pitch])

    def flow_relaxation_NS(self, gpu_stream, h, u, v):
        self.flowRelaxationScheme_NS.async_call(
            self.global_size, self.local_size, gpu_stream,
            [self.boundary_conditions.north, self.boundary_conditions.south,
            self.nx, self.ny,
            self.halo_x, self.halo_y,
            self.boundary_conditions.spongeCells['north'],
            self.boundary_conditions.spongeCells['south'],
            h.data.gpudata, h.pitch,
            u.data.gpudata, u.pitch,
            v.data.gpudata, v.pitch,
            self.bc_NS_current_arr.data.gpudata,
            self.bc_NS_next_arr.data.gpudata,
            self.bc_t])

    def flow_relaxation_EW(self, gpu_stream, h, u, v):
        self.flowRelaxationScheme_EW.async_call(
            self.global_size, self.local_size, gpu_stream,
            [self.boundary_conditions.east, self.boundary_conditions.west,
            self.nx, self.ny,
            self.halo_x, self.halo_y,
            self.boundary_conditions.spongeCells['east'],
            self.boundary_conditions.spongeCells['west'],
            h.data.gpudata, h.pitch,
            u.data.gpudata, u.pitch,
            v.data.gpudata, v.pitch,
            self.bc_EW_current_arr.data.gpudata,
            self.bc_EW_next_arr.data.gpudata,
            self.bc_t])
