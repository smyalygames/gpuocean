#!/bin/bash -l
#SBATCH --job-name=One-Node-Scaling-GPUOcean
#SBATCH --account=project_465002898
#SBATCH --time=01:00:00
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=out/%x-%j.out


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
params="-dt=0.1 -t=100 --profile --nccl --warmup --run_times=3"
project_dir=/project/project_465002898/anthony/gpuocean
application=${project_dir}/mpi_test.py
container=${project_dir}/../gpuocean_0.0.13-lumi.sif
data_out_dir=$project_dir/bench/one_node_scaling/$SLURM_JOB_ID

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

mkdir -p "$data_out_dir"

for power in {8..14}; do
  dimension=$((2 ** power))
  mkdir -p "$data_out_dir"/"$dimension"
  srun --mpi=cray_shasta --ntasks=1 \
	  singularity exec \
	  --writable-tmpfs \
	  --bind="${project_dir}" \
	  ${container} \
	  bash -c "uv run --no-cache ${application} ${params} -nx=${dimension} -ny=${dimension}"

  find ${project_dir} -maxdepth 1 -type f \( -iname "*${SLURM_JOB_ID}*.log" -o -iname "*${SLURM_JOB_ID}*.json" \) \
  -print0 | xargs -0r mv -t "${data_out_dir}/${dimension}"
done


end_time=$(curr_time)
echo "Finished job at: ${end_time}"