#!/bin/bash -l
module load LUMI/25.09 partition/G
module load rocm/6.4.4
module load cray-mpich-abi/9.0.1
module load lumi-CrayPath
module load aws-ofi-nccl/1.18.0-rocm

project_dir=/project/project_465002898/anthony/gpuocean


##### Required modules #####

export MPICH_GPU_SUPPORT_ENABLED=1
#export MPICH_OFI_NIC_POLICY=GPU
#export MPICH_GPU_IPC_ENABLED=0
#export MPICH_ENV_DISPLAY=1
export NCCL_DEBUG=TRACE
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_DEBUG=INFO
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
#export NCCL_NET="OFI"
#export FI_CXI_RX_MATCH_MODE=hybrid

export NCCL_DMABUF_ENABLE=1

##### Running the job ######

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

#srun -A project_465002898 -p dev-g --nodes=1 --gpus=2 --mem-per-gpu=64G --time=01:15:00 --pty bash -i
#salloc -A project_465002898 -p dev-g --nodes=1 --gpus=2 --ntasks-per-node=2 --mem-per-gpu=64G --time=01:15:00 --pty bash -i

#uv run --no-cache /project/project_465002898/anthony/gpuocean/mpi_test.py -nx=22000 -ny=22000 -dt=0.1 -t=10 --rank_data --warmup --nccl --run_times=1
# srun --ntasks-per-node=2 --gpus-per-node=2 --mpi=cray_shasta singularity exec --writable-tmpfs -B $PWD:$PWD ../gpuocean_0.0.13-lumi.sif bash -c "uv run --no-cache /project/project_465002898/anthony/gpuocean/mpi_test.py -nx=22000 -ny=22000 -dt=0.1 -t=10 --rank_data --warmup --nccl --run_times=1; EXIT_CODE=\$?; echo \"PROCESS \${SLURM_PROCID} ON \$(hostname) EXITED WITH CODE: \${EXIT_CODE}\"; exit \${EXIT_CODE}"