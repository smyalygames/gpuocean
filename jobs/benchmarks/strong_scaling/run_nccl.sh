#!/bin/bash
set -e

opt_params=""

while getopts ":d" option; do
  case $option in
    d)
      opt_params+="--dynamic_dt";;
    \?)
      echo "Invalid option provided"
      exit;;
  esac
done

export OPTIONAL_PARAMS=$opt_params

benchmark_id=$(date +%s)
export BENCHMARK_ID=$benchmark_id
echo "Submitting nccl strong scaling test, with benchmark ID: ${benchmark_id}"

job=/project/project_465002898/anthony/gpuocean/jobs/benchmarks/strong_scaling/nccl.slurm

# Only for handling <1 node
for processes in {0..2}; do
  total_tasks=$((2**processes))

  export PROCESSES=$total_tasks

  # Default job settings
  nodes=1
  gpus_per_node=$total_tasks
  partition="small-g"

  echo "Submitting ${total_tasks}-process job to ${partition} (${nodes} nodes)..."

  sbatch \
    --partition=$partition \
    --nodes=$nodes \
    --ntasks-per-node=$gpus_per_node \
    --gpus-per-node=$gpus_per_node \
    --time=01:00:00 \
    --output="out/GPUOcean-nccl-strong-%j-${total_tasks}.out" \
    $job "$benchmark_id"
done

gpus=(1 2 4 8 16)

for nodes in "${gpus[@]}"; do
  gpus_per_node=8
  total_tasks=$((8*nodes))
  partition="standard-g"

  export PROCESSES=$total_tasks

  if [ "$nodes" -gt 4 ]; then
    partition="standard-g"
  fi

  echo "Submitting ${total_tasks}-process job to ${partition} (${nodes} nodes)..."

  sbatch \
    --partition=$partition \
    --nodes="$nodes" \
    --ntasks-per-node=$gpus_per_node \
    --gpus-per-node=$gpus_per_node \
    --exclusive \
    --time=00:20:00 \
    --output="out/GPUOcean-nccl-strong-%j-${total_tasks}.out" \
    $job "$benchmark_id"
done

echo "All jobs submitted!"