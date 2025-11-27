#!/bin/bash -l
#SBATCH --job-name=FVM-Test
#SBATCH --account=project_465001926
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=%x-%j.out
#SBATCH --exclusive


module load LUMI/24.03 partition/G
module load singularity-bindings

##### Helper Functions #####
curr_time() {
  date "+%Y-%m-%d %H:%M:%S"
}

##### Variables ######
project_dir=/project/project_465001926/anthony
application=${project_dir}/FiniteVolumeGPU/mpi_testing_hip.py
container=${project_dir}/FiniteVolumeGPU/fvm.sif

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

##### Required modules #####

#export MPICH_GPU_SUPPORT_ENABLED=1

##### Running the job ######
start_time=$(curr_time)
echo "Starting job at: ${start_time}"

srun --cpu-bind=${CPU_BIND} --mpi=cray_shasta \
	  singularity exec \
	  -B ${project_dir}/FiniteVolumeGPU \
	  ${container} \
	  micromamba run -n base python ${application} -nx 1028 -ny 1028 --profile


end_time=$(curr_time)
echo "Finished job at: ${end_time}"