"""Integration tests for periodic MPI halos and gathered solution fields."""

import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest


def run_worker(ranks, output, nx, ny, periodic_x, periodic_y, dynamic, nonpersistent,
               variable_depth=False):
    executable = shutil.which("mpiexec")
    if executable is None:
        pytest.skip("mpiexec is not available")
    command = [sys.executable, __file__, "--worker", str(output), str(nx), str(ny),
               str(int(periodic_x)), str(int(periodic_y)), str(int(dynamic)),
               str(int(nonpersistent)), str(int(variable_depth))]
    if ranks > 1:
        command = [executable, "-n", str(ranks), *command]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(output) as data:
        return [data[name] for name in ("eta", "hu", "hv")]


@pytest.mark.parametrize(
    "nx,ny,periodic_x,periodic_y,ranks,dynamic,nonpersistent",
    [(32, 24, True, True, 2, False, False),
     (24, 32, True, True, 2, True, True),
     (32, 32, True, True, 4, True, False),
     (32, 24, True, False, 2, False, True),
     (24, 32, False, True, 2, False, False),
     (32, 24, False, True, 2, False, True),
     (32, 24, False, False, 4, False, False)])
def test_periodic_mpi_matches_single_rank(tmp_path, nx, ny, periodic_x, periodic_y,
                                          ranks, dynamic, nonpersistent):
    reference = run_worker(1, tmp_path / "reference.npz", nx, ny, periodic_x,
                           periodic_y, dynamic, nonpersistent)
    actual = run_worker(ranks, tmp_path / "parallel.npz", nx, ny, periodic_x,
                        periodic_y, dynamic, nonpersistent)
    for expected, result in zip(reference, actual):
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(np.sum(result, dtype=np.float64),
                                   np.sum(expected, dtype=np.float64),
                                   rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("ranks", [2, 4])
def test_periodic_global_bathymetry_halos(tmp_path, ranks):
    reference = run_worker(1, tmp_path / "reference.npz", 32, 32, True, True,
                           False, False, variable_depth=True)
    actual = run_worker(ranks, tmp_path / "parallel.npz", 32, 32, True, True,
                        False, False, variable_depth=True)
    for expected, result in zip(reference, actual):
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("ranks,dynamic", [(1, False), (1, True), (2, False), (2, True)])
def test_conservation_runner(tmp_path, ranks, dynamic):
    executable = shutil.which("mpiexec")
    if ranks > 1 and executable is None:
        pytest.skip("mpiexec is not available")
    script = os.path.join(os.path.dirname(__file__), "..", "validate_conservation.py")
    output = tmp_path / "conservation.csv"
    command = [sys.executable, script, "--nx", "32", "--ny", "32", "--duration", "1",
               "--interval", "0.3", "--output", str(output)]
    if dynamic:
        command.append("--dynamic-dt")
    if ranks > 1:
        command = [executable, "-n", str(ranks), *command]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with output.open() as file:
        rows = list(csv.DictReader(file))
    assert float(rows[0]["time_s"]) == 0
    assert float(rows[-1]["time_s"]) == pytest.approx(1.0, abs=1e-6)
    assert len(rows) >= 4
    for row in rows:
        for field in ("mass_relative_drift", "mass_anomaly_relative_drift",
                      "momentum_x_relative_drift", "momentum_y_relative_drift"):
            assert abs(float(row[field])) < 1e-5


def test_conservation_rejects_unstable_fixed_step():
    script = os.path.join(os.path.dirname(__file__), "..", "validate_conservation.py")
    result = subprocess.run([sys.executable, script, "--nx", "32", "--ny", "32",
                             "--duration", "1", "--dt", "0.2"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "exceeds CFL limit" in result.stderr


def test_conservation_fails_when_tolerance_is_exceeded():
    script = os.path.join(os.path.dirname(__file__), "..", "validate_conservation.py")
    result = subprocess.run([sys.executable, script, "--nx", "32", "--ny", "32",
                             "--duration", "1", "--tolerance", "1e-15"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "FAIL: maximum relative drift" in result.stderr


def test_conservation_does_not_overwrite_output(tmp_path):
    executable = shutil.which("mpiexec")
    if executable is None:
        pytest.skip("mpiexec is not available")
    script = os.path.join(os.path.dirname(__file__), "..", "validate_conservation.py")
    output = tmp_path / "existing.csv"
    output.write_text("existing data")
    result = subprocess.run([executable, "-n", "2", sys.executable, script,
                             "--output", str(output)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "Cannot open diagnostic output" in result.stderr
    assert output.read_text() == "existing data"


def test_weak_scaling_periodic_is_rejected():
    from gpuocean.utils.Common import BoundaryConditions, BoundaryType
    from gpuocean.utils.mpi import MPIWrapper

    class TwoRanks:
        size = 2
        rank = 0

    periodic = BoundaryConditions(*([BoundaryType.PERIODIC] * 4))
    with pytest.raises(NotImplementedError, match="require strong scaling"):
        MPIWrapper(None, 32, 32, (2, 2, 2, 2), strong_scale=False,
                   comm=TwoRanks(), boundary_conditions=periodic)


def worker():
    from mpi4py import MPI
    from gpuocean.SWEsimulators import SimulatorType
    from gpuocean.utils.Common import BoundaryConditions, BoundaryType
    from gpuocean.utils.gpu import KernelContext, gpu_device
    from gpuocean.utils.mpi import MPIWrapper

    output, nx, ny, periodic_x, periodic_y, dynamic, nonpersistent, variable_depth = sys.argv[2:]
    nx, ny = int(nx), int(ny)
    periodic_x, periodic_y, dynamic, nonpersistent = (
        bool(int(v)) for v in (periodic_x, periodic_y, dynamic, nonpersistent))
    variable_depth = bool(int(variable_depth))
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    eta = (0.3 * np.sin(2 * np.pi * x / nx) +
           0.2 * np.cos(2 * np.pi * y / ny)).astype(np.float32)
    hu = (0.4 + 0.1 * np.sin(2 * np.pi * y / ny)).astype(np.float32)
    hv = (-0.2 + 0.1 * np.cos(2 * np.pi * x / nx)).astype(np.float32)
    eta, hu, hv = (np.pad(a, 2, mode="wrap") for a in (eta, hu, hv))
    H = np.full((ny + 5, nx + 5), 60, dtype=np.float32)
    if variable_depth:
        H[:] = 99
        H[2:ny + 2, 2:nx + 2] = 60 + np.sin(2 * np.pi * x / nx) + np.cos(2 * np.pi * y / ny)
    bc_x = BoundaryType.PERIODIC if periodic_x else BoundaryType.WALL
    bc_y = BoundaryType.PERIODIC if periodic_y else BoundaryType.WALL
    comm = MPI.COMM_WORLD
    ctx = KernelContext(device=gpu_device.get_device_for_rank(comm.rank))
    sim = MPIWrapper(SimulatorType.CDKLM16, nx, ny, (2, 2, 2, 2),
                     comm=comm, mpi_persistent=not nonpersistent,
                     boundary_conditions=BoundaryConditions(bc_y, bc_x, bc_y, bc_x),
                     gpu_ctx=ctx, eta0=eta, hu0=hu, hv0=hv,
                     H=H,
                     dx=10.0, dy=10.0, dt=0 if dynamic else 0.05,
                     g=9.81, f=0.0, r=0.0, write_netcdf=False)
    sim.step(0.5, update_dt=dynamic, split_step=False, write_now=False,
             enable_progress_bar=False)
    sim.sim.gpu_stream.synchronize()
    result = sim.download(interior_domain_only=True)
    if comm.rank == 0:
        np.savez(output, eta=result[0], hu=result[1], hv=result[2])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
