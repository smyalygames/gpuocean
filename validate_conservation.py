"""Run a periodic, unforced mass and momentum conservation experiment."""

import argparse
import csv
import math
import sys

import numpy as np
from mpi4py import MPI

from gpuocean.SWEsimulators import SimulatorType
from gpuocean.utils.Common import BoundaryConditions, BoundaryType
from gpuocean.utils.gpu import KernelContext, gpu_device
from gpuocean.utils.mpi import MPIWrapper


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def positive_int(value):
    result = int(value)
    if result < 8:
        raise argparse.ArgumentTypeError("must be at least 8")
    return result


def initial_conditions(nx, ny, dx, dy, depth):
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    lx, ly = nx * dx, ny * dy
    rx = (x - 0.37 * lx + lx / 2) % lx - lx / 2
    ry = (y - 0.43 * ly + ly / 2) % ly - ly / 2
    eta = np.exp(-((rx[None, :] / (0.08 * lx)) ** 2 +
                   (ry[:, None] / (0.08 * ly)) ** 2)).astype(np.float32)
    h = depth + eta
    return tuple(np.pad(a, 2, mode="wrap") for a in (eta, 0.1 * h, -0.08 * h))


def totals(sim, comm, depth, area, density):
    sim.sim.gpu_stream.synchronize()
    eta, hu, hv = sim.sim.download(interior_domain_only=True)
    local = np.array([np.sum(eta, dtype=np.float64),
                      np.sum(hu, dtype=np.float64),
                      np.sum(hv, dtype=np.float64)], dtype=np.float64)
    global_sums = np.empty_like(local)
    comm.Allreduce(local, global_sums, op=MPI.SUM)
    global_sums *= density * area
    mass_anomaly = global_sums[0]
    global_sums[0] += density * area * depth * sim.grid.global_nx * sim.grid.global_ny
    return np.append(global_sums, mass_anomaly)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=positive_int, default=256)
    parser.add_argument("--ny", type=positive_int, default=256)
    parser.add_argument("--duration", type=positive_float, default=1000.0, help="Simulation seconds")
    parser.add_argument("--interval", type=positive_float, default=10.0, help="Diagnostic interval in seconds")
    parser.add_argument("--dt", type=positive_float, default=0.05, help="Fixed timestep in seconds")
    parser.add_argument("--dynamic-dt", action="store_true", help="Update the CFL timestep every step")
    parser.add_argument("--mpi-nonpersistent", action="store_true", help="Use non-persistent MPI exchange")
    parser.add_argument("--tolerance", type=positive_float, default=1e-5,
                        help="Maximum relative drift for each integral")
    parser.add_argument("--output", help="Write CSV to a new file instead of stdout")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    output = None
    error = None
    if rank == 0:
        try:
            output = open(args.output, "x", newline="") if args.output else sys.stdout
        except OSError as exc:
            error = f"Cannot open diagnostic output: {exc}"
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    dx = dy = 10.0
    depth, gravity, density = 60.0, 9.81, 1025.0
    eta, hu, hv = initial_conditions(args.nx, args.ny, dx, dy, depth)
    bathymetry = np.full((args.ny + 5, args.nx + 5), depth, dtype=np.float32)
    periodic = BoundaryConditions(*([BoundaryType.PERIODIC] * 4))

    gpu_ctx = KernelContext(device=gpu_device.get_device_for_rank(rank))
    sim = MPIWrapper(SimulatorType.CDKLM16, args.nx, args.ny, (2, 2, 2, 2),
                     comm=comm, mpi_persistent=not args.mpi_nonpersistent,
                     boundary_conditions=periodic, gpu_ctx=gpu_ctx, eta0=eta,
                     hu0=hu, hv0=hv, H=bathymetry, dx=dx, dy=dy,
                     dt=0 if args.dynamic_dt else args.dt, g=gravity, f=0.0, r=0.0,
                     write_netcdf=False)
    initial = totals(sim, comm, depth, dx * dy, density)
    max_drift = np.zeros(4)
    minimum_dt, maximum_dt = math.inf, 0.0
    writer = csv.writer(output) if rank == 0 else None

    def write_row(row):
        error = None
        if rank == 0:
            try:
                writer.writerow(row)
                output.flush()
            except OSError as exc:
                error = f"Cannot write diagnostic output: {exc}"
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)

    write_row(["time_s", "iterations", "min_dt_s", "max_dt_s", "mass_kg",
               "momentum_x_kg_m_s", "momentum_y_kg_m_s", "mass_relative_drift",
               "momentum_x_relative_drift", "momentum_y_relative_drift",
               "mass_anomaly_relative_drift"])

    def record():
        values = totals(sim, comm, depth, dx * dy, density)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Non-finite conservation integral")
        drift = (values - initial) / np.abs(initial)
        np.maximum(max_drift, np.abs(drift), out=max_drift)
        write_row([float(sim.sim.t), sim.sim.num_iterations,
                   minimum_dt if math.isfinite(minimum_dt) else 0.0,
                   maximum_dt, *values[:3], *drift])

    record()
    next_output = min(args.interval, args.duration)
    while float(sim.sim.t) < args.duration:
        sim.update_dt()
        limit = float(sim.sim.dt)
        if not math.isfinite(limit) or limit <= 0:
            raise RuntimeError(f"Invalid CFL timestep: {limit}")
        if not args.dynamic_dt and args.dt > limit:
            raise RuntimeError(f"Fixed timestep {args.dt} exceeds CFL limit {limit} at t={sim.sim.t}")
        requested = limit if args.dynamic_dt else args.dt
        step = float(np.float32(min(requested, args.duration - float(sim.sim.t))))
        if step <= 0:
            raise RuntimeError("Timestep rounded to zero before reaching the requested duration")
        sim.sim.dt = step
        sim.sim.step(step, write_now=False, split_step=False)
        minimum_dt = min(minimum_dt, step)
        maximum_dt = max(maximum_dt, step)
        if float(sim.sim.t) >= next_output:
            record()
            next_output = min(next_output + args.interval, args.duration)
    failed = bool(np.any(max_drift > args.tolerance))
    if rank == 0:
        print(f"{'FAIL' if failed else 'PASS'}: maximum relative drift: mass={max_drift[0]:.6g}, "
              f"momentum_x={max_drift[1]:.6g}, momentum_y={max_drift[2]:.6g}; "
              f"mass_anomaly={max_drift[3]:.6g}; "
              f"tolerance={args.tolerance:.6g}", file=sys.stderr)
        if args.output:
            output.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
