# Convert Docker image to Singularity:

There is a pre-built Docker image for LUMI, however, as LUMI does not support Docker containers, it needs to be
converted to a Singularity/Apptainer container.

## Preparation

To do so on LUMI, first prepare the cache directories as such:

```bash
$ mkdir -p /tmp/$USER
$ export SINGULARITY_TMPDIR=/tmp/$USER
$ export SINGULARITY_CACHEDIR=/tmp/$USER
```

## Creation

Then go to the directory where the Singularity container should be stored (preferably somewhere in the
`/project/project_XXXXXXXXXX` directory) and then it can be created with:

```
singularity pull docker://smyalygames/gpuocean:0.0.13-lumi
```

The latest versions can be found at https://hub.docker.com/r/smyalygames/gpuocean/tags, make sure that the tag ends with
`-lumi`.

# Building required EasyConfigs

`aws-ofi-nccl` works for both NCCL and RCCL, LUMI allows it to be installed through a
pre-configured [EasyConfig](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/a/aws-ofi-nccl/).

However, this is not required MPI will only be used (even though RCCL is usually more performant).

## Preparation

Ensure that the `EBU_USER_PREFIX` environment variable is set to the correct location either through `.profile`,
`.bashrc`, or running it in the terminal (which requires adding it additionally to the Slurm job scripts) as such:

```shell
export EBU_USER_PREFIX=/project/project_465000000/EasyBuild
```

Where `465000000` gets replaced with your project number.

## Building aws-ofi-nccl

Load the required modules:

```shell
$ module load LUMI/25.09 partition/G
$ module load rocm/6.4.4
$ module load lumi-CrayPath
$ module load EasyBuild-user
```

Then simply build it with the command:

```shell
eb aws-ofi-nccl-1.18.0-rocm.eb -r
```

Once completed, the new module has been created and it can be loaded through `module load aws-ofi-nccl/1.18.0-rocm`
(which is not necessary if using the job scripts provided in the repository).

# Running GPU Ocean

This GitHub repository can be cloned into the project directory into LUMI, and ensure that it is on the `lumi` branch.

## Running existing scripts

There are some pre-defined scripts in this repository which can be found in [`jobs/`](jobs). Ensure that the scripts
are updated to the current project, and that the directories are correctly set to where this repository is located on
LUMI.

Furthermore inside the script, the `params` variable can be changed to the params provided in [
`mpi_test.py`](mpi_test.py).

Some benchmarks have a bash script (e.g. `run_nccl.sh`) and a Slurm script (e.g. `nccl.slurm`). If this is the case, the
bash scripts should be used to run the tests (remember to `chmod +x ./jobs/*`) as they start multiple jobs with
different amount of processes (this can be modified in the bash script).

## Creating your own Slurm script

If you want to create a new script, there are a few important things that you need to consider including.

### Loading required modules

These are the modules that are used throughout the scripts created in this repository, but they may change in the future
as LUMI gets updated:

```sh
module load LUMI/25.09 partition/G
module load rocm/6.4.4
module load cray-mpich-abi/9.0.1
module load lumi-CrayPath
module load aws-ofi-nccl/1.18.0-rocm
```

Brief explanation on why these are used:

- `cray-mpich-abi/9.0.1` - allows the MPICH provided in the Singularity container as created above to be replaced by
  Cray MPICH on LUMI (requires setting up bindings)
- `lumi-CrayPath` - provides required `LD_LIBRARY_PATH` for certain Cray libraries which is required for `LUMI/25.09`
- `aws-ofi-nccl/1.18.0-rocm` - Speeds up RCCL communication

### MPI Optimisations

MPICH by default does not do direct GPU-GPU communication, and it has to be explicitly set. To enable it to do so, add
these environment variables to the Slurm script:

```sh
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
```

These can be included either way if MPI is not being used.

### RCCL optimisations

There are certain optimisations that can be implemented to improve the performance of RCCL, as found in this
paper: [arXiv:2408.14090v2](https://arxiv.org/abs/2408.14090v2).

The environment variables used were:

```sh
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
if [ "$processes" -gt 8 ]; then
  export NCCL_NET="OFI"
  export FI_CXI_RX_MATCH_MODE=hybrid
fi

export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NCHANNELS_PER_PEER=32
```

Take note of the line with `if [ "$processes" -gt 8 ]; then`, the variable `processes` has to be defined manually in the
script, as there is no automatic way (as far as I can tell) to detect the number of processes the Slurm job is going to
have before `srun`.

These can also be included if RCCL is not being used.

### Singularity environment variables

As the container uses uv as the Python package manager, these are required to ensure that uv functions correctly:

```sh
export SINGULARITYENV_UV_PROJECT=${project_dir} # ensure this variable is defined, or just hardcode the directory where GPU Ocean is located
export SINGULARITYENV_UV_PROJECT_ENVIRONMENT=/app/.venv
export SINGULARITYENV_UV_LINK_MODE=copy
export SINGULARITYENV_VIRTUAL_ENV=/app/.venv
export SINGULARITYENV_UV_PYTHON_INSTALL_DIR=/python
```

### Singularity bindings

To allow the use of various libraries on LUMI that are not available to install directly into the container, (i.e. Cray
libraries), the bindings have to be defined for the Singularity container.

There is already a specific script for provided in this repository located in [
`jobs/lumi-singularity-bindings.sh`](jobs/lumi-singularity-bindings.sh).

It is preferred to use that script by adding:

```sh
source $project_dir/jobs/lumi-singularity-bindings.sh
```

Ensure that `project_dir` is defined to where GPU Ocean is located on LUMI, or just replace `$project_dir` with the full
location.

#### Explaining the script

```sh
export SINGULARITYENV_LD_PRELOAD="/opt/cray/pe/mpich/9.0.1/gtl/lib/libmpi_gtl_hsa.so.0"
```

is required for the GPU interconnect to correctly function, otherwise the project will not run as it cannot detect any
GTL libraries.

The `SINGULARITYENV_LD_LIBRARY_PATH` part of the script ensures that the required Cray libraries are loaded for Cray
MPICH ABI to correctly function.

`SINGULARITY_BIND` binds the files for the required libraries and system configurations on the LUMI system into the
container, which is required for Cray and `aws-ofi-nccl`/RCCL to function.

### Running GPU Ocean

To run GPU Ocean with `srun` in the Slurm script, something along the lines of this is required:

```sh
srun ${cpu_bind:+"--cpu-bind=${cpu_bind}"} --mpi=cray_shasta \
  singularity exec \
  --writable-tmpfs \
  --bind="${project_dir}" \
  --bind="${scratch_dir}" \
  ${container} \
  bash -c "uv run --no-cache ${application} ${params}" || exit 1
```

- Ensure that the CPUs are bound correctly (if using one or more nodes), they can be defined with just
  `cpu_bind="map_cpu:49,57,17,25,1,9,33,41"` or exclude that if using less than a full node.
- `--mpi=cray_shasta` - allows for Cray MPICH to be used.
- `--bind="${project_dir}"` - is for where GPU Ocean is located on LUMI
- `--bind="${scratch_dir}"` - is optional if you specify where the netCDF files are output in the parameters for running
  GPU Ocean, as these files can get too large for the project directory
- `${container}` - location of the Singularity container created as was done earlier
- `bash -c "uv run --no-cache ${application} ${params}"` - executes GPU Ocean
    - `${application}` - location of the script to run with GPU Ocean, usually it is just the directory to where GPU
      Ocean is located on LUMI and then a script such as `mpi_test.py`
    - `${params}` - parameters for the Python script to take

### Profiling GPU Ocean inside the container

`rocprofv3` can be used to profile GPU Ocean and it just requires replacing the contents of `bash -c` with something
along the lines of:

```shell
bash -c "uv run --no-cache \
	  rocprofv3 --sys-trace --stats --summary --truncate-kernels --output-format pftrace --output-directory ${project_dir}/prof/${SLURM_JOB_ID} -- python ${application} ${params}"
```

Everything from `rocprofv3` to `-- python` is for the `rocprofv3` arguments. These current arguments should produce a
Perfetto trace file, that can be analysed through https://ui.perfetto.dev/.