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
export UV_NO_PROGRESS=1
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

cd "${project_dir}"

uv sync --locked --extra lumi --group jupyter
uv run python -m ipykernel install --user --name gpuocean-lumi --display-name "GPUOcean LUMI (uv)"

export GPUOCEAN_KERNEL_WRAPPER=${script_dir}/lumi_uv_kernel.sh
export GPUOCEAN_VENV_KERNEL=${project_dir}/.venv/share/jupyter/kernels/python3/kernel.json
export GPUOCEAN_USER_KERNEL=$(uv run python -m jupyter --data-dir)/kernels/gpuocean-lumi/kernel.json
uv run python - <<'PY'
import json
import os
from pathlib import Path

wrapper = os.environ["GPUOCEAN_KERNEL_WRAPPER"]
argv = [wrapper, "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"]

for kernel_path in (Path(os.environ["GPUOCEAN_VENV_KERNEL"]), Path(os.environ["GPUOCEAN_USER_KERNEL"])):
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    if kernel_path.exists():
        data = json.loads(kernel_path.read_text())
    else:
        data = {"language": "python", "metadata": {"debugger": True}}
    data["argv"] = argv
    data["display_name"] = "GPUOcean LUMI (uv)"
    data["language"] = "python"
    data.setdefault("metadata", {})["debugger"] = True
    kernel_path.write_text(json.dumps(data, indent=1) + "\n")

sitecustomize_path = Path(os.environ["VIRTUAL_ENV"]) / "lib" / "python3.13" / "site-packages" / "sitecustomize.py"
sitecustomize_path.write_text(f'''\
import os
import sys

try:
    with open("/proc/self/cmdline", "rb") as cmdline_file:
        cmdline_args = [arg.decode(errors="ignore") for arg in cmdline_file.read().split(b"\\0") if arg]
except OSError:
    cmdline_args = [sys.executable, *sys.argv]

is_ipykernel = any("ipykernel_launcher" in arg for arg in cmdline_args) or any("ipykernel_launcher" in arg for arg in sys.argv)

if os.environ.get("GPUOCEAN_LUMI_KERNEL") != "1" and is_ipykernel:
    wrapper = {wrapper!r}
    if os.path.exists(wrapper):
        os.execv(wrapper, [wrapper, *cmdline_args[1:]])
''')
PY

uv run python -c "import ipykernel, cupy, hip; from gpuocean.SWEsimulators import GPUOceanSim; print('ok'); print('devices:', cupy.cuda.runtime.getDeviceCount())"