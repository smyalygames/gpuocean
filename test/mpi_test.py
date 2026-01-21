from mpi4py import MPI
import numpy as np

from gpuocean.SWEsimulators.CDKLM16 import CDKLM16
from gpuocean.utils.gpu import KernelContext
from gpuocean.utils.mpi import MPIWrapper

# MPI information
comm = MPI.COMM_WORLD
rank = comm.rank

# For PyCharm debugger
# if rank == 0:
#     import pydevd_pycharm
#     pydevd_pycharm.settrace('localhost', port=12345, stdout_to_server=True, stderr_to_server=True)

# Simulator variables
gpu_ctx = KernelContext()

dx = 200.0
dy = 200.0

dt = 0.9
g = 9.81
f = 0.0
r = 0.0
A = 1

waterHeight = 60

ghosts = [2, 2, 2, 2]  # north, east, south, west
validDomain = np.array([2, 2, 2, 2])
dataRange = [-2, -2, 2, 2]
refRange = dataRange

T = 50.0

ny, nx = (8, 8)

# Create testing grid
shape = (ny + ghosts[0] + ghosts[2],
         nx + ghosts[1] + ghosts[3])

grid = np.ones(shape, dtype=np.float32)
# half = shape[0] // 2
grid[:] = int(rank)

eta0 = grid
u0 = grid
v0 = grid
Hi = np.ones((shape[0] + 1, shape[1] + 1), dtype=np.float32, order='C') * waterHeight

# Setup Simulator
local_sim = CDKLM16(gpu_ctx,
                    eta0, u0, v0, Hi,
                    nx, ny,
                    dx, dy, dt,
                    g, f, r,
                    comm=comm)

sim = MPIWrapper(local_sim)

# Run simulator
# t = sim.step()

# Output results
eta1, u1, v1 = sim.download()

print(f"Rank {rank} Completed. Results:")
print(f"eta1: {eta1}\n")
# print(f"u1: {u1}\n")
# print(f"v1: {v1}\n")
