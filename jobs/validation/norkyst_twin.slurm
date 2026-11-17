#!/bin/bash -l
#SBATCH --job-name=GPUOcean-Norkyst-Twin
#SBATCH --account=project_465002898
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=GPUOcean-Norkyst-Twin-%j.out

set -euo pipefail

module load LUMI/25.09 partition/G
module load rocm/6.4.4
module load cray-mpich-abi/9.0.1
module load lumi-CrayPath
module load aws-ofi-nccl/1.18.0-rocm

project_dir=${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
: "${CONTAINER:?Set CONTAINER to the existing LUMI GPU Ocean Singularity image}"
: "${INPUT_DIR:?Set INPUT_DIR to the directory containing staged Norkyst files}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a shared writable result directory}"
project_dir=$(realpath "${project_dir}")
input_dir=$(realpath "${INPUT_DIR}")
container=$(realpath "${CONTAINER}")
mkdir -p "${OUTPUT_ROOT}"
output_root=$(realpath "${OUTPUT_ROOT}")
output_dir="${output_root}/norkyst-twin-${SLURM_JOB_ID}"
[[ -f "${project_dir}/validate_norkyst.py" ]] || { echo "PROJECT_DIR is not the updated GPU Ocean checkout" >&2; exit 1; }
[[ ! -e "${output_dir}" ]] || { echo "Output already exists: ${output_dir}" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "Pass --source FILE [--source FILE ...] and optional experiment arguments after the script" >&2; exit 1; }
[[ -f "${container}" ]] || { echo "Container does not exist: ${container}" >&2; exit 1; }
if [[ "${SLURM_NTASKS}" -ne "${SLURM_GPUS:-${SLURM_NTASKS}}" ]]; then
    echo "Allocate one GPU per MPI rank" >&2
    exit 1
fi

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_IPC_THRESHOLD=1
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_DEFAULT_TX_SIZE=2048
if [[ "${SLURM_NNODES}" -gt 1 ]]; then
    export FI_CXI_RX_MATCH_MODE=hybrid
fi

export SINGULARITYENV_UV_PROJECT="${project_dir}"
export SINGULARITYENV_UV_PROJECT_ENVIRONMENT=/app/.venv
export SINGULARITYENV_UV_LINK_MODE=copy
export SINGULARITYENV_VIRTUAL_ENV=/app/.venv
export SINGULARITYENV_UV_PYTHON_INSTALL_DIR=/python
export SINGULARITYENV_GPU_LANG=hip
export SINGULARITYENV_MPLBACKEND=Agg
export SINGULARITYENV_PYTHONPATH="${project_dir}/src"
source "${project_dir}/jobs/lumi-singularity-bindings.sh"

cpu_bind=()
if [[ "${SLURM_NTASKS}" -eq $((8 * SLURM_NNODES)) ]]; then
    cpu_bind=(--cpu-bind=map_cpu:49,57,17,25,1,9,33,41)
fi

echo "Starting Norkyst twin job ${SLURM_JOB_ID} at $(date --iso-8601=seconds)"
cd "${input_dir}"
srun "${cpu_bind[@]}" --mpi=cray_shasta \
    singularity exec --writable-tmpfs \
    --bind="${project_dir}" --bind="${input_dir}" --bind="${output_root}" \
    "${container}" uv run --no-sync --project "${project_dir}" \
    python "${project_dir}/validate_norkyst.py" "$@" --output "${output_dir}"

# Analysis is deliberately a separate, single-process CPU operation.
srun --ntasks=1 --mpi=cray_shasta \
    singularity exec --writable-tmpfs \
    --bind="${project_dir}" --bind="${output_root}" \
    "${container}" uv run --no-sync --project "${project_dir}" \
    python "${project_dir}/validate_norkyst.py" --analyse "${output_dir}"
echo "Completed at $(date --iso-8601=seconds): ${output_dir}"
