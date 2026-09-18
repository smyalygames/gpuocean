#!/bin/bash -l
#SBATCH --job-name=GPUOcean-Test
#SBATCH --account=project_465002898
#SBATCH --time=00:30:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=out/%x-%j.out
#SBATCH --exclusive


module load LUMI/25.09 partition/G
module load rocm/6.4.4
module load cray-mpich-abi/9.0.1
module load lumi-CrayPath

##### Helper Functions #####
curr_time() {
  date "+%Y-%m-%d %H:%M:%S"
}

##### Variables ######
#params="-nx=512 -ny=512 --dynamic_dt --profile"
params="-nx=4096 -ny=4096 -t=1 --mpi"
project_dir=/project/project_465002898/anthony/gpuocean
application=${project_dir}/mpi_test.py
container=${project_dir}/../gpuocean_0.0.13-lumi.sif

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

##### Required modules #####

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
#export MPICH_GPU_IPC_ENABLED=0
#export MPICH_ENV_DISPLAY=1

##### Running the job ######
start_time=$(curr_time)
echo "Starting job at: ${start_time}"

export SINGULARITYENV_UV_PROJECT=${project_dir}
export SINGULARITYENV_UV_PROJECT_ENVIRONMENT=/app/.venv
export SINGULARITYENV_UV_LINK_MODE=copy
export SINGULARITYENV_VIRTUAL_ENV=/app/.venv
export SINGULARITYENV_UV_PYTHON_INSTALL_DIR=/python

source $project_dir/jobs/lumi-singularity-bindings.sh

srun --cpu-bind=${CPU_BIND} --mpi=cray_shasta \
	  singularity exec \
	  --writable-tmpfs \
	  --bind="${project_dir}" \
	  ${container} \
	  bash -c "uv run --no-cache \
	  rocprofv3 --sys-trace --stats --summary --truncate-kernels --output-format pftrace --output-directory ${project_dir}/prof/${SLURM_JOB_ID} -- python ${application} ${params}"


end_time=$(curr_time)
echo "Finished job at: ${end_time}"