#!/bin/bash -l
#SBATCH --job-name=GPUOcean-Strong-Scaling-Benchmark
#SBATCH --account=project_465002898
#SBATCH --time=01:30:00
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --output=out/%x-%j.out


module load LUMI partition/G
#module load craype-accel-amd-gfx90a
#module load cray-mpich-abi/8.1.32

##### Helper Functions #####
curr_time() {
  date "+%Y-%m-%d %H:%M:%S"
}

##### Variables ######
params="-nx=4100 -ny=4100 -dt=0.1 -t=1000 --profile --nccl --warmup --run_times=3"
project_dir=/project/project_465002898/anthony/gpuocean
application=${project_dir}/mpi_test.py
container=${project_dir}/../gpuocean_0.0.8-lumi.sif
DATA_OUT_DIR=$project_dir/$SLURM_JOB_ID

#CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

##### Required modules #####

export MPICH_GPU_SUPPORT_ENABLED=1
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
export SINGULARITYENV_MPICH_GPU_IPC_ENABLED=0

source $project_dir/jobs/lumi-singularity-bindings.sh

mkdir -p $DATA_OUT_DIR

for processes in {1..4}; do
  mkdir -p "$DATA_OUT_DIR"/"$processes"
  srun --mpi=cray_shasta --ntasks="${processes}" \
	  singularity exec \
	  --writable-tmpfs \
	  --bind="${project_dir}" \
	  ${container} \
	  bash -c "uv run --no-cache ${application} ${params}"

  mv "$project_dir"/*.log "$DATA_OUT_DIR"/"$processes"
  mv "$project_dir"/*.json "$DATA_OUT_DIR"/"$processes"
done


end_time=$(curr_time)
echo "Finished job at: ${end_time}"