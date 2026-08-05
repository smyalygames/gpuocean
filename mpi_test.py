import argparse
import json
import logging
import os
from argparse import ArgumentParser, ArgumentTypeError
import time
from typing import Any
import socket
from tqdm import trange

from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import cupy as cp

from gpuocean.utils.gpu import KernelContext, gpu_device
from gpuocean.utils.mpi import MPIWrapper
from gpuocean.SWEsimulators import SimulatorType
from gpuocean.utils.netcdf.NetCDFInitialization import getInitialConditionsNorKystCases, removeMetadata, \
    rescaleInitialConditions

bump_elementwise = cp.ElementwiseKernel(
    'int32 sim_nx, int32 sim_ny, float64 sim_dx, float64 sim_dy, int32 h0, int32 h1, int32 h2, int32 h3, int32 n_cols',
    'T eta',
    '''
    // 'i' is automatically provided by CuPy as the 1D linear index.
    // We must reconstruct the 2D grid coordinates from 'i'.
    int row = i / n_cols;
    int col = i % n_cols;

    // Adjust for the halo to match the original Python loop bounds
    int grid_j = row - h2;
    int grid_i = col - h3;

    // Only process if we are within the target bounds
    if (grid_i >= -h3 && grid_i < sim_nx + h1 && grid_j >= -h2 && grid_j < sim_ny + h0) {

        double x_center = sim_dx * sim_nx / 2.0;
        double y_center = sim_dy * sim_ny / 2.0;
        double size = 500.0 * (sim_dx < sim_dy ? sim_dx : sim_dy);

        double x = sim_dx * grid_i - x_center;
        double y = sim_dy * grid_j - y_center;
        double r2 = x*x + y*y;

        if (r2 < size * size) {
            eta += exp(-(r2 / size)); // Adds directly to the current element
        }
    }
    ''',
    'add_bump_element'
)


def write_profiling():
    with open(profiling_file_name, "w") as file:
        json.dump(profiling_data, file, indent=4)


def addCentralBump(eta: npt.NDArray, sim_nx: int, sim_ny: int, sim_dx: float, sim_dy: float, halo: npt.NDArray):
    array = cp.asarray(eta)

    n_cols = sim_nx + halo[1] + halo[3]

    # Notice how much cleaner the Python call is! No block/grid math.
    bump_elementwise(
        sim_nx, sim_ny, sim_dx, sim_dy,
        int(halo[0]), int(halo[1]), int(halo[2]), int(halo[3]), n_cols, array
    )

    return cp.asnumpy(array)


parser = ArgumentParser(description="Testing Parallel GPU Ocean")
exchange_group = parser.add_argument_group('Exchange Method', 'Method for used for ghost cell exchanges')
exchange_method = exchange_group.add_mutually_exclusive_group(required=True)
init_group = parser.add_argument_group('Data Initialization', 'The type of data to use for initialization of the grid')
init_data = init_group.add_mutually_exclusive_group(required=False)
output = parser.add_argument_group('Output', 'What data should be outputted')


def min_run_times(value):
    ivalue = int(value)
    if ivalue < 1:
        raise ArgumentTypeError(f"{value} is an invalid positive integer value (minimum 1).")
    return ivalue


parser.add_argument('-nx', type=int, default=128)
parser.add_argument('-ny', type=int, default=128)
parser.add_argument('-dt', type=float, default=0.1, help='Time step size')
parser.add_argument('-t', type=float, default=1000, help='Total simulation time to run for')
parser.add_argument('--weak-scale', action='store_true', help='Do not do domain decomposition.')
parser.add_argument('--dynamic_dt', action='store_true', help='Dynamically calculate time step')
parser.add_argument('--rescale', action='store_true', help="Rescales the domain to match set nx")
output.add_argument('--netcdf', action='store_true', help='Output netCDF file')
output.add_argument('-o', '--output', type=str, default="mpi_test.nc", help='Output location for netCDF file')
output.add_argument('--include_ghostcells', action='store_false', help='Include ghost cells in the netCDF file')
output.add_argument('--nc-only-last', action='store_true', help="Only stores the first and last parts of the simulation in the netCDF file.")
output.add_argument('--profile', action='store_true')
output.add_argument('--log_debug', action='store_true', help='Writes debug logs to a log file')
output.add_argument('--progress-bar', action='store_true', help='Show progress bar.')

# Mostly for benchmarking
parser.add_argument('--warmup', action='store_true', help='Run a warmup simulation first before the main task')
parser.add_argument('--warmup_t', type=int, default=10, help='Time to run the simulation for during a warmup')
parser.add_argument('--run_times', type=min_run_times, default=1, help='Number of simulation runs to complete')

exchange_method.add_argument('--mpi', action='store_true', help='Uses persistent MPI')
exchange_method.add_argument('--mpi_np', action='store_true', help='Uses NON-persistent MPI')
exchange_method.add_argument('--nccl', action='store_true', help='Uses NCCL/RCCL')

init_data.add_argument('--norkyst_url', type=str, help='Use Norkyst data from a URL')
init_data.add_argument('--rank_data', action='store_true', help='Makes the data in each rank their rank value')

args = parser.parse_args()

# Simulator conditions
dt: float = args.dt
dynamic_dt: bool = args.dynamic_dt
rescale: bool = args.rescale
write_netcdf: bool = args.netcdf
netcdf_filename: str = args.output
ignore_ghostcells = args.include_ghostcells
nc_only_last: bool = args.nc_only_last
use_mpi: bool = args.mpi or args.mpi_np
use_mpi_persistent = not args.mpi_np
use_nccl: bool = args.nccl
log_debug: bool = args.log_debug
split_step = False
compile_opts = ['-O2', '-funroll-loops', '-ffast-math']
strong_scale = not args.weak_scale
disable_tqdm = not args.progress_bar

# Type of simulation data to use
norkyst_url: str | None = args.norkyst_url
create_rank_data: bool = args.rank_data

# Benchmarking arguments
warmup: bool = args.warmup
warmup_t: int = args.warmup_t
run_times: int = args.run_times

# MPI information
comm = MPI.COMM_WORLD
rank = comm.rank

device = gpu_device.get_device_for_rank(rank)

# Set up profiler
profiling = args.profile

job_id = ""

profiling_file_name = ""
total_devices = gpu_device.get_device_count()
if "SLURM_JOB_ID" in os.environ:
    job_id = int(os.environ["SLURM_JOB_ID"])
    allocated_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    profiling_file_name = f"MPI_jobid_{job_id}_{allocated_nodes}_nodes_"
else:
    profiling_file_name = f"MPI_{MPI.COMM_WORLD.size}_procs_"
profiling_file_name += f"{total_devices}_GPUs_profiling_proc_{rank}.json"

if profiling:
    profiling_data = {}
    profiling_data['job_id'] = job_id
    profiling_data['hostname'] = socket.gethostname()
    profiling_data['mpi_abi'] = str(MPI.Get_abi_info())
    profiling_data['mpi_abi_ver'] = str(MPI.Get_abi_version())
    profiling_data['mpi4py_ver'] = str(MPI.Get_library_version())

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.rank
    local_size = local_comm.size
    local_comm.Free()

    if local_size == comm.size:
        if local_rank == 0:
            send_gpu_count = total_devices
            send_unique_device = 1
        else:
            send_gpu_count = 0
            send_unique_device = 0

        global_gpu_count = comm.allreduce(send_gpu_count, op=MPI.SUM)
        nodes_count = comm.allreduce(send_unique_device, op=MPI.SUM)
    else:
        global_gpu_count = total_devices
        nodes_count = 1

    profiling_data['nodes_count'] = nodes_count
    profiling_data['gpu_device_global_count'] = global_gpu_count
    profiling_data['gpu_device_node_count'] = total_devices
    profiling_data['gpu_device'] = device
    profiling_data['n_processes'] = MPI.COMM_WORLD.size
    profiling_data['gpu_compile_args'] = compile_opts
    profiling_data['netcdf_only_last'] = nc_only_last
    profiling_data['log_debug'] = log_debug
    init_data_type = 'bump'
    if norkyst_url is not None:
        init_data_type = 'norkyst'
    elif create_rank_data:
        init_data_type = 'rank'
    profiling_data['init_data_type'] = init_data_type

    write_profiling()

    t_total_start = time.time()
    t_init_start = time.time()

current_path = os.getcwd()

# For PyCharm debugger
# if rank == 0:
#     import pydevd_pycharm
#     pydevd_pycharm.settrace('localhost', port=12345, stdout_to_server=True, stderr_to_server=True)

# Create logger
log_level_console = 20
log_level_file = 10

logger = logging.getLogger('gpuocean')
logger.setLevel(log_level_console)
ch = logging.StreamHandler()
ch.setLevel(log_level_console)
logger.addHandler(ch)
logger.info("Console logger using level %s", logging.getLevelName(log_level_console))

if log_debug:
    log_filename = f'mpi_{rank}'
    if job_id == "":
        log_filename += '.log'
    else:
        log_filename += f'_{job_id}.log'

    logger.setLevel(min(log_level_console, log_level_file))
    fh = logging.FileHandler(log_filename)
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(log_level_file)
    logger.addHandler(fh)

    logger.info("File logger using level %s to %s", logging.getLevelName(log_level_file), log_filename)

nccl_test = os.getenv('NCCL_DEBUG')
logger.info(f"Singularity env test for NCCL_DEBUG, got: {nccl_test}")
# Simulator variables
gpu_ctx = KernelContext(device=device)

waterHeight = 60

ghosts = (2, 2, 2, 2)  # north, east, south, west
validDomain = np.array([2, 2, 2, 2])
dataRange = [-2, -2, 2, 2]
refRange = dataRange

T = 1

nx = args.nx
ny = args.ny

if norkyst_url is not None:
    T_hours = 24
    timestep_indices = [list(range(0, T_hours))]
    case_name = 'complete_coast'

    if profiling:
        norkyst_info = {
            'url': norkyst_url,
            'case_name': case_name,
            't_hours': T_hours,
        }
        profiling_data['norkyst_info'] = norkyst_info
        write_profiling()

    if rank == 0:
        kwargs: dict[str, Any] = getInitialConditionsNorKystCases(norkyst_url, case_name,
                                                                  download_data=True, erode_land=1,
                                                                  timestep_indices=timestep_indices)
    else:
        kwargs = None

    kwargs: dict[str, Any] = comm.bcast(kwargs, root=0)

    if rescale:
        scale = nx / kwargs['nx']
        kwargs = rescaleInitialConditions(kwargs, scale)

    kwargs = removeMetadata(kwargs)

    nx = kwargs['nx']
    ny = kwargs['ny']
else:
    dx = 10.0
    dy = 10.0

    g = 9.81
    f = 0.0
    r = 0.0
    A = 1

    # Create testing grid
    shape = (ny + ghosts[0] + ghosts[2],
             nx + ghosts[1] + ghosts[3])

    grid = np.zeros(shape, dtype=np.float32)

    if create_rank_data:
        grid[:] = int(rank)
        # for j in range(shape[0]):
        #     for i in range(shape[1]):
        #         grid[j, i] = int(rank)  # i * (10 ** int(rank)) + j

    eta0 = grid
    u0 = grid
    v0 = grid
    # eta0 = np.zeros(shape, dtype=np.float32)
    # u0 = np.zeros(shape, dtype=np.float32)
    # v0 = np.zeros(shape, dtype=np.float32)
    Hi = np.ones((shape[0] + 1, shape[1] + 1), dtype=np.float32, order='C') * waterHeight

    if not create_rank_data:
        eta0 = addCentralBump(eta0, nx, ny, dx, dy, validDomain)

    kwargs: dict[str, Any] = {
        'eta0': eta0,
        'hu0': u0,
        'hv0': v0,
        'H': Hi,
        'dx': dx,
        'dy': dy,
        'dt': dt,
        'g': g,
        'f': f,
        'r': r,
    }

# Setup Simulator
sim_args = (
)
kwargs.update({
    'gpu_ctx': gpu_ctx,
    'write_netcdf': write_netcdf,
    'super_dir_name': current_path,
    'netcdf_filename': netcdf_filename,
    'ignore_ghostcells': ignore_ghostcells,
    'compile_opts': compile_opts
})

# Initialisation
if profiling:
    profiling_data['nx'] = nx
    profiling_data['ny'] = ny
    profiling_data['dt_start'] = dt

    write_profiling()

    t_sim_init_start = time.time()

sim = MPIWrapper(SimulatorType.CDKLM16, nx, ny, ghosts, strong_scale, comm, use_nccl, use_mpi_persistent, *sim_args, **kwargs)

if profiling:
    t_sim_init_end = time.time()
    t_init_end = time.time()

    t_sim_init = t_sim_init_end - t_sim_init_start
    t_init = t_init_end - t_init_start
    profiling_data['t_sim_init'] = t_sim_init
    profiling_data['t_init'] = t_init

    if sim.mpi_handler is None:
        profiling_data['exchange_method'] = None
        pass

    elif sim.mpi_handler.nccl is not None:
        profiling_data['exchange_method'] = 'nccl'
    elif sim.mpi_handler.nccl_comm is not None:
        profiling_data['exchange_method'] = 'nccl_comm'
    elif sim.mpi_handler.mpi_persistent:
        profiling_data['exchange_method'] = 'mpi4py'
    else:
        profiling_data['exchange_method'] = 'non-persistent mpi4py'

    write_profiling()

# Warmup run
if warmup:
    logger.info("Starting warmup.")
    if profiling:
        t_warmup_start = time.time()

    sim.step(t_end=warmup_t, update_dt=dynamic_dt, split_step=split_step, write_now=not nc_only_last,
             enable_progress_bar=not disable_tqdm)
    sim.sim.gpu_stream.synchronize()
    if sim.sim.write_netcdf:
        if nc_only_last:
            sim.sim.writeState()
        sim.sim.sim_writer.sync()

    if profiling:
        t_warmup_end = time.time()
        t_warmup = t_warmup_end - t_warmup_start
        profiling_data['t_warmup'] = t_warmup
        profiling_data['warmup_set_sim_t'] = warmup_t

        write_profiling()

    logger.info("Completed warmup, re-initialising simulator")

    sim.reinit(kwargs['eta0'], kwargs['hu0'], kwargs['hv0'], kwargs['dt'])

    logger.debug("Warmup re-initialisation complete")

if profiling:
    profiling_data['t_sim_run'] = []
    write_profiling()

logger.info(f"Running simulations for {run_times} runs.")
for i in trange(run_times, disable=disable_tqdm):
    run = i + 1
    logger.info(f"Starting run {run}.")
    if profiling:
        t_sim_run_start = time.time()

    # Run simulator
    t = sim.step(t_end=args.t, update_dt=dynamic_dt, split_step=split_step, write_now=not nc_only_last,
                 enable_progress_bar=not disable_tqdm)
    sim.sim.gpu_stream.synchronize()
    if sim.sim.write_netcdf:
        if nc_only_last:
            sim.sim.writeState()
        sim.sim.sim_writer.sync()

    if profiling:
        t_sim_run_end = time.time()
        t_sim_run = t_sim_run_end - t_sim_run_start
        profiling_data['t_sim_run'].append(t_sim_run)
        write_profiling()

    logger.info(f"Completed run {run}.")

    # Re-initialize the simulator for next runs
    if i < run_times - 1:
        logger.debug(f"Re-initialising simulator after run {run}.")
        sim.reinit(kwargs['eta0'], kwargs['hu0'], kwargs['hv0'], kwargs['dt'])
        logger.debug(f"Completed re-initialisation of the simulator after run {run}.")

# Output results
# eta1, u1, v1 = sim.download()

# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

print(f"Rank {rank} Completed. Results:")
# print(f"eta1 ({rank}): {eta1}\n")

# print(f"diff ({rank}): {eta1 - eta0}")
# print(f"u1: {u1}\n")
# print(f"v1: {v1}\n")

if profiling:
    t_total_end = time.time()
    t_total = t_total_end - t_total_start
    profiling_data['t_mean_sim_run'] = np.mean(profiling_data['t_sim_run'])
    profiling_data['t_total'] = t_total

    logger.info(f"Total run time on {rank} is {t_total} seconds.")

    # if rank != 0:
    #     exit(0)
    profiling_data['sim_nx'] = sim.sim.nx
    profiling_data['sim_ny'] = sim.sim.ny
    profiling_data['dt_end'] = sim.sim.dt
    profiling_data['n_iterations'] = sim.sim.num_iterations
    profiling_data['sim_time'] = sim.sim.t

    write_profiling()

exit(0)
