"""Manual GPU/MPI integration checks: GPUOCEAN_RUN_GPU_TESTS=1 pytest -q this_file."""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("GPUOCEAN_RUN_GPU_TESTS") != "1",
                                reason="Requires an explicitly allocated GPU/MPI environment")


@pytest.mark.parametrize("open_boundary", [False, True])
@pytest.mark.parametrize("ranks", [1, 2])
def test_budget_and_decomposition(tmp_path, open_boundary, ranks):
    results = []
    for processes, budget in ((1, False), (1, True), (ranks, True)):
        output = tmp_path / f"result-{len(results)}.npz"
        command = [sys.executable, __file__, "--worker", str(output), str(int(open_boundary)), str(int(budget))]
        if processes > 1:
            launcher = shutil.which("mpiexec")
            if launcher is None:
                pytest.skip("mpiexec is unavailable; run the two-rank worker through srun on LUMI")
            command = [launcher, "-n", str(processes), *command]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        with np.load(output) as data:
            results.append({name: data[name] for name in data.files})
    for actual in results[1:]:
        np.testing.assert_allclose(actual["state"], results[0]["state"], rtol=2e-6, atol=2e-6)
        delta = np.sum(actual["state"] - actual["initial"], axis=(1, 2), dtype=np.float64)
        expected = actual["budget"].sum(axis=0)
        scale = np.maximum(1, np.sum(np.abs(actual["initial"]), axis=(1, 2), dtype=np.float64))
        assert np.all(np.abs(delta - expected) / scale < 1e-5)
        if not open_boundary:
            assert np.all(np.abs(expected) / scale < 1e-5)


def worker():
    from mpi4py import MPI
    from gpuocean.SWEsimulators import SimulatorType
    from gpuocean.utils.Common import (
        BoundaryConditions, BoundaryConditionsData, BoundaryType,
        SingleBoundaryConditionData, SpongeCells,
    )
    from gpuocean.utils.gpu import KernelContext, gpu_device
    from gpuocean.utils.mpi import MPIWrapper
    from gpuocean.utils.WindStress import WindStress
    from gpuocean.utils.AtmosphericPressure import AtmosphericPressure

    output, open_boundary, budget = sys.argv[2:]
    open_boundary, budget = bool(int(open_boundary)), bool(int(budget))
    nx, ny = 32, 24
    x, y = np.meshgrid(np.arange(nx + 4), np.arange(ny + 4))
    eta = (0.1 * np.sin(2 * np.pi * (x - 2) / nx)).astype(np.float32)
    hu, hv = np.full_like(eta, 1.0), np.full_like(eta, -0.5)
    xc, yc = np.meshgrid(np.arange(nx + 5), np.arange(ny + 5))
    depth = np.full_like(xc, 60, dtype=np.float32)
    if open_boundary:
        depth += np.sin(2 * np.pi * xc / nx).astype(np.float32)
    boundary_type = BoundaryType.FLOW_RELAXATION_SCHEME if open_boundary else BoundaryType.PERIODIC
    boundaries = BoundaryConditions(*([boundary_type] * 4), sponge_cells=SpongeCells(5, 5, 5, 5))

    def boundary(length):
        h = np.stack((np.linspace(-0.1, 0.1, length), np.linspace(0.1, 0.2, length))).astype(np.float32)
        return SingleBoundaryConditionData(h, np.ones_like(h), np.full_like(h, -0.5))

    data = BoundaryConditionsData([0, 0.5], north=boundary(nx + 4), south=boundary(nx + 4),
                                  east=boundary(ny + 4), west=boundary(ny + 4))
    forcing = {}
    if open_boundary:
        wind = (2 + 0.01 * x).astype(np.float32)
        forcing["wind"] = WindStress(t=[0, 0.5], wind_u=[wind, wind], wind_v=[wind, wind])
        pressure = (100000 + x + y).astype(np.float32)
        forcing["atmospheric_pressure"] = AtmosphericPressure(t=[0, 0.5], P=[pressure.copy(), pressure.copy()])
    comm = MPI.COMM_WORLD
    ctx = KernelContext(device=gpu_device.get_device_for_rank(comm.rank))
    sim = MPIWrapper(SimulatorType.CDKLM16, nx, ny, (2, 2, 2, 2), comm=comm,
                     gpu_ctx=ctx, eta0=eta, hu0=hu, hv0=hv, H=depth, dx=10.0, dy=10.0,
                     dt=0.01, g=9.81, f=1e-4 if open_boundary else 0,
                     r=0.003 if open_boundary else 0, write_netcdf=False,
                     boundary_conditions=boundaries, boundary_conditions_data=data,
                     budget_diagnostics=budget, **forcing)
    initial = sim.download(interior_domain_only=True)
    sim.sim.t = np.float64(0)
    while sim.sim.t < 0.5 - 1e-8:
        sim.update_dt()
        step = float(np.float32(min(0.01, float(sim.sim.dt), 0.5 - float(sim.sim.t))))
        sim.sim.dt = step
        sim.sim.step(step, write_now=False, split_step=False, apply_stochastic_term=False)
    state = sim.download(interior_domain_only=True)
    totals = np.zeros((5, 3))
    if budget:
        comm.Allreduce(sim.sim.budget.totals(), totals)
    if comm.rank == 0:
        np.savez(output, initial=np.asarray(initial), state=np.asarray(state), budget=totals)
    if budget:
        sim.reinit(eta, hu, hv, dt=0.01)
        np.testing.assert_array_equal(sim.sim.budget.totals(), np.zeros((5, 3)))
    sim.cleanUp()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    worker()
