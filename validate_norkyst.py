"""Command-line interface to the shared 24-hour Norkyst twin experiment."""

import argparse
import logging

from gpuocean.utils.norkyst_twin import TwinConfig, TwinExperiment, plot_results, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="Local Norkyst Z-level file or OPeNDAP URL; repeat for continuation files")
    parser.add_argument("--output", help="New experiment output directory")
    parser.add_argument("--start", help="Exact source timestamp, ISO 8601 UTC; default first timestamp")
    parser.add_argument("--bounds", nargs=4, type=int, default=(1400, 1900, 450, 750),
                        metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--verification-bounds", nargs=4, type=int, metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--duration", type=float, default=86400)
    parser.add_argument("--output-interval", type=float, default=3600)
    parser.add_argument("--diagnostic-interval", type=float, default=600)
    parser.add_argument("--sponge-cells", type=int, default=20)
    parser.add_argument("--buffer-cells", type=int, default=10)
    parser.add_argument("--courant-number", type=float, default=0.8)
    parser.add_argument("--fixed-dt", type=float)
    parser.add_argument("--friction", type=float, default=0.003)
    parser.add_argument("--no-wind", action="store_true", help="Explicitly omit wind forcing; recorded as a model mismatch")
    parser.add_argument("--pressure-variable", help="Source atmospheric pressure variable, in Pa, hPa or mbar")
    parser.add_argument("--budget", action="store_true", help="Enable expensive RK2 discrete update accounting")
    parser.add_argument("--check-inputs", action="store_true", help="CPU-only preflight; creates no output")
    parser.add_argument("--analyse", metavar="DIRECTORY", help="CPU-only plots from a saved experiment")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.analyse:
        from matplotlib import pyplot as plt
        for figure in plot_results(args.analyse, save=True):
            plt.close(figure)
        return
    if not args.source or not args.output:
        parser.error("--source and --output are required unless --analyse is used")
    config = TwinConfig(
        sources=tuple(args.source), output=args.output, start=args.start, bounds=tuple(args.bounds),
        verification_bounds=tuple(args.verification_bounds) if args.verification_bounds else None,
        duration=args.duration, output_interval=args.output_interval, diagnostic_interval=args.diagnostic_interval,
        sponge_cells=args.sponge_cells, buffer_cells=args.buffer_cells, courant_number=args.courant_number,
        fixed_dt=args.fixed_dt, friction=args.friction, wind=not args.no_wind,
        pressure_variable=args.pressure_variable, budget=args.budget)
    if args.check_inputs:
        prepared = prepare(config)
        logging.info("Preflight passed: %d source timestamps; %g hours; dx=%g, dy=%g",
                     len(prepared.times), config.duration / 3600, prepared.dx, prepared.dy)
        return
    from mpi4py import MPI
    try:
        with TwinExperiment(config, comm=MPI.COMM_WORLD) as experiment:
            experiment.advance_to(config.duration)
        if MPI.COMM_WORLD.rank == 0:
            logging.info("Completed twin experiment: %s", config.output)
    except Exception:
        logging.exception("Twin experiment failed; output, if present, is incomplete")
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == "__main__":
    main()
