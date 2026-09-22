#!/bin/bash -l
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)

module load LUMI/25.09 partition/G
module load rocm/6.4.4
module load cray-mpich/9.0.1
module load cray-hdf5-parallel/1.14.3.7
module load cray-netcdf-hdf5parallel/4.9.2.1
module load lumi-CrayPath

export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_LANG=hip
export UV_LINK_MODE=copy
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=${ROCM_PATH:-${EBROOTROCM:-/opt/rocm}}
export HIPCC=${ROCM_HOME}/bin/hipcc
export HCC_AMDGPU_TARGET=gfx90a
export NETCDF4_MPI=enabled
export NETCDF4_DIR=${NETCDF_DIR}
export CC=cc
export CXX=CC
export MPICC=cc
export MPICXX=CC
export GPUOCEAN_LUMI_KERNEL=1

cd "${project_dir}"
exec "${project_dir}/.venv/bin/python" "$@"