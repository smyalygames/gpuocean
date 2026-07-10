#!/bin/bash -l
#SBATCH --job-name=netCDF-GPUOcean-Test
#SBATCH --account=project_465002898
#SBATCH --time=00:30:00
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=out/%x-%j.out
#SBATCH --exclusive


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
processes=8

project_dir=/project/project_465002898/anthony/gpuocean
application=${project_dir}/mpi_test.py
container=${project_dir}/../gpuocean_0.0.8-lumi.sif
data_out_dir=$project_dir/bench/test_netcdf_nccl_${SLURM_JOB_ID}
netcdf_file="${project_dir}/${SLURM_JOB_ID}.nc"
params="-nx=256 -ny=256 -dt=0 --dynamic_dt -t=100 --profile --warmup --mpi --run_times=3 --netcdf -o=$netcdf_file"

if [ "$processes" -ge 8 ]; then
  cpu_bind="map_cpu:49,57,17,25,1,9,33,41"
fi

##### Required modules #####

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
#export MPICH_GPU_IPC_ENABLED=0
#export MPICH_ENV_DISPLAY=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

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
if [ "$processes" -ge 8 ]; then
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

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export SINGULARITYENV_MASTER_ADDR="$master_addr"
export SINGULARITYENV_MASTER_PORT=6200
echo "MASTER_ADDR="$SINGULARITYENV_MASTER_ADDR "MASTER_PORT="$SINGULARITYENV_MASTER_PORT

source $project_dir/jobs/lumi-singularity-bindings.sh

mkdir -p "${data_out_dir}/${processes}"

mkdir -p "$data_out_dir/$processes"
srun ${cpu_bind:+"--cpu-bind=${cpu_bind}"} --mpi=cray_shasta \
  singularity exec \
  --writable-tmpfs \
  --bind="${project_dir}" \
  ${container} \
  bash -c "uv run --no-cache ${application} ${params}" || exit 1

find ${project_dir} -maxdepth 1 -type f \
  \( -iname "*${SLURM_JOB_ID}*.log" -o -iname "*${SLURM_JOB_ID}*.json" -o -iname "*${SLURM_JOB_ID}*.nc" \) \
  -print0 | xargs -0r mv -t "${data_out_dir}/${processes}"

end_time=$(curr_time)
echo "Finished job at: ${end_time}"