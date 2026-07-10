#!/bin/bash -l
#SBATCH --job-name=MPI-GPUOcean-Strong-Scaling
#SBATCH --account=project_465002898


module load LUMI/25.09 partition/G
module load rocm/6.4.4
module load cray-mpich-abi/9.0.1
module load lumi-CrayPath
module load aws-ofi-nccl/1.18.0-rocm

##### Helper Functions #####
curr_time() {
  date "+%Y-%m-%d %H:%M:%S"
}

##### Variables ######
processes="${PROCESSES}"
benchmark_id="${BENCHMARK_ID}"

params="-nx=22000 -ny=22000 -dt=0.1 -t=100 --profile --mpi_np --warmup --run_times=3"
project_dir=/project/project_465002898/anthony/gpuocean
application=${project_dir}/mpi_test.py
container=${project_dir}/../gpuocean_0.0.13-lumi.sif
data_out_dir=$project_dir/bench/strong/mpi_${benchmark_id}

if [ "$processes" -ge 8 ]; then
  cpu_bind="map_cpu:49,57,17,25,1,9,33,41"
fi

##### Required modules #####

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_IPC_THRESHOLD=1
export HSA_ENABLE_SDMA=0
#export MPICH_GPU_IPC_ENABLED=0
#export MPICH_ENV_DISPLAY=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_DEFAULT_TX_SIZE=2048
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
## WARNING: Do not set NCCL_NET on single-node Slurm runs. Setting this variable
## forces NCCL to use the network transport even when all ranks share the same
## node, causing unnecessary VNI allocation and degraded performance.
if [ "$processes" -gt 8 ]; then
  export NCCL_NET="OFI"
  export FI_CXI_RX_MATCH_MODE=hybrid
fi

##### Running the job ######
start_time=$(curr_time)
echo "Starting job at: ${start_time}"

export SINGULARITYENV_UV_PROJECT=${project_dir}
export SINGULARITYENV_UV_PROJECT_ENVIRONMENT=/app/.venv
export SINGULARITYENV_UV_LINK_MODE=copy
export SINGULARITYENV_VIRTUAL_ENV=/app/.venv
export SINGULARITYENV_UV_PYTHON_INSTALL_DIR=/python

source $project_dir/jobs/lumi-singularity-bindings.sh

mkdir -p "${data_out_dir}/${processes}"

mkdir -p "$data_out_dir/$processes"
srun ${cpu_bind:+"--cpu-bind=${cpu_bind}"} --mpi=cray_shasta \
  singularity exec \
  --writable-tmpfs \
  --bind="${project_dir}" \
  ${container} \
  bash -c "uv run --no-cache ${application} ${params}" || exit 1

find ${project_dir} -maxdepth 1 -type f \( -iname "*${SLURM_JOB_ID}*.log" -o -iname "*${SLURM_JOB_ID}*.json" \) \
  -print0 | xargs -0r mv -t "${data_out_dir}/${processes}"

end_time=$(curr_time)
echo "Finished job at: ${end_time}"