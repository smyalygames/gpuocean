"""Opt-in RK2 update accounting; deliberately separate from the normal solver path."""

import numpy as np

from gpuocean.utils.gpu import Array2D


class CDKLM16Budget:
    def __init__(self, sim, wet_mask=None):
        self.sim = sim
        depth = np.ma.filled(sim.downloadBathymetry(interior_domain_only=True)[1], np.nan)
        self.wet = np.isfinite(depth) & (depth < 1e10)
        if wet_mask is not None:
            if wet_mask.shape != (sim.ny + 4, sim.nx + 4):
                raise ValueError("budget_wet_mask must include the two-cell halos")
            self.wet &= wet_mask[2:-2, 2:-2]
        self.zeros = np.zeros((12 * sim.ny, sim.nx), dtype=np.float64)
        self.increments = Array2D(sim.gpu_stream, sim.nx, 12 * sim.ny, 0, 0, self.zeros,
                                  double_precision=True, padded=False)
        self.relaxation = np.zeros(3, dtype=np.float64)

    def state_totals(self, arrays):
        self.sim.gpu_stream.synchronize()
        return np.array([np.sum(a.download(self.sim.gpu_stream)[2:-2, 2:-2][self.wet],
                                dtype=np.float64) for a in arrays])

    def reset(self):
        self.sim.gpu_stream.synchronize()
        self.increments.upload(self.sim.gpu_stream, self.zeros)
        self.relaxation[:] = 0

    def totals(self):
        self.sim.gpu_stream.synchronize()
        values = self.increments.download(self.sim.gpu_stream).reshape(4, 3, self.sim.ny, self.sim.nx)
        return np.vstack((np.sum(values[:, :, self.wet], axis=-1, dtype=np.float64), self.relaxation))

    def release(self):
        self.increments.release()
