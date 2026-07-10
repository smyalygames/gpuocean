#!/bin/bash -l
#SBATCH --job-name=GPUOcean-Test
#SBATCH --account=project_465002898
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=%x-%j.out
#SBATCH --exclusive


module load LUMI partition/G
#module load cray-mpich-abi/8.1.32

##### Helper Functions #####
curr_time() {
  date "+%Y-%m-%d %H:%M:%S"
}

##### Variables ######
project_dir=/project/project_465002898/anthony/gpuocean
application=${project_dir}/mpi_test.py
container=${project_dir}/../gpuocean_container

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

##### Required modules #####

#export MPICH_GPU_SUPPORT_ENABLED=1

##### Running the job ######
start_time=$(curr_time)
echo "Starting job at: ${start_time}"

export PATH="${container}/bin:$PATH"

#export UV_PROJECT=$project_dir
#export UV_PROJECT_ENVIRONMENT=/app/.venv
#export UV_LINK_MODE=copy
#export VIRTUAL_ENV=/app/.venv
#export UV_PYTHON_INSTALL_DIR=/python

srun --cpu-bind=${CPU_BIND} --mpi=cray_shasta \
	  uv run --no-project ${application}


end_time=$(curr_time)
echo "Finished job at: ${end_time}"