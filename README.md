[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.11502400.svg)](http://dx.doi.org/10.5281/zenodo.11502400)

<img src="logo/gpuocean-logo.png" width=500>

 
**A GPU-accelerated simulation framework for running large ensembles of simplified ocean models for real-world domains.**

Operational ocean forecasting models are computationally expensive and are therefore often run as a single deterministic simulation at given intervals. This means that they often lack information about forecast uncertainties, which are significant given that there are relative few observations of the ocean compared to its sheer size. Information on uncertainty is, however, vital for various applications, including search-and-rescue operations at sea.

### What is GPU Ocean?
GPU Ocean is a simulation framework designed to explore the short-term uncertainty in forecasted ocean currents. It achieves this by running ensembles of simplified ocean models nested within operational ocean forecasts. These simplified models efficiently solve barotropic dynamics described by the shallow-water equations using GPUs.

### Key Features:

* **Efficient simulation:** GPU Ocean utilizes GPU acceleration and state-of-the-art finite-volume methods to solve shallow-water equations efficiently.
* **Model nesting:**  Easily import initialization data, terrain information, forcing, and boundary conditions from NetCDF files containing operational 3D ocean forecasts.
* **Drift trajectory prediction:** Conduct online drift trajectory simulations for drifting objects within the ocean models, directly assessing uncertainty in drift trajectories based on the uncertainty in the ocean currents.
* **Data assimilation**: GPU Ocean includes tailored data-assimilation methods for sparse in-situ observations.
* **Python with CUDA performance**: Rapid prototyping and easy orchestration, pre-, and post-processing using Python while getting the computational performance of CUDA.
* **Parallel processing:** MPI support for running even larger ensembles on multiple GPUs. 


GPU Ocean aims to be a powerful tool to complement ocean current forecasts through estimating and accounting for forecast uncertainties. By combining local observations and advanced data assimilation methods, users can make more informed decisions based on the latest oceanic data.

## Installation
See [here](https://github.com/gpuocean/gpuocean/wiki/Installation).

## Mass and momentum validation

`validate_conservation.py` runs a periodic, flat-bottom CDKLM16 experiment with
no external forcing, Coriolis or friction. It checks mass and both momentum
components for either fixed or dynamically calculated timesteps. The default
256 x 256 domain has 10 m cells and 60 m depth; its gravity-wave crossing time
is approximately 106 seconds. The default 1,000-second run covers about 9.5
crossings. Use `--duration 180` for a shorter pilot.

With the project's GPU dependencies installed, for example on ROCm 6.4:

```bash
uv sync --locked --extra lumi --inexact
uv run --no-sync python validate_conservation.py --dt 0.05 --output fixed.csv
uv run --no-sync python validate_conservation.py --dynamic-dt --output dynamic.csv
mpiexec -n 2 uv run --no-sync python validate_conservation.py --dt 0.05 --output mpi-fixed.csv
mpiexec -n 2 uv run --no-sync python validate_conservation.py --dynamic-dt --output mpi-dynamic.csv
```

Use the GPU extra appropriate for your system. Multiple ranks require a
GPU-aware MPI implementation; persistent MPI is the default, with
`--mpi-nonpersistent` available as an alternative. Diagnostics are reduced
across ranks and written by rank zero, so parallel NetCDF is not required.
Output files must not already exist. Without `--output`, CSV is written to
standard output. Multi-rank periodic boundaries currently require strong
scaling (a single global domain decomposed across ranks).

The CSV reports actual simulation times, iteration counts, cumulative minimum
and maximum timestep, integrals, and signed relative drifts. Mass includes
the mean water depth; the separate mass-anomaly drift is normalized by the
initial excess mass above that depth. Momentum drifts are normalized by the
nonzero initial momentum in each direction. Integrals use float64 accumulation
over interior cells, excluding halos. Diagnostics are sampled after the first
step at or beyond each `--interval` (default 10 seconds), and at the final
time. The last timestep may be shortened to reach the final time.

The command fails if a sampled absolute relative drift exceeds `--tolerance`
(default `1e-5`), a value becomes non-finite, or the fixed timestep exceeds the
calculated CFL limit at any step. The tolerance is a validation starting point,
not a universal error bound; calibrate it for the intended hardware and run.
This experiment does not validate forced, open-boundary, or variable-depth
simulations.

## 24-hour Norkyst twin experiment

The [interactive notebook](examples/NorkystTwinExperiment.ipynb) and
[`validate_norkyst.py`](validate_norkyst.py) use the same
[experiment implementation](src/gpuocean/utils/norkyst_twin.py). GPU Ocean is
initialized once from Norkyst and evolves without interior nudging, while
Norkyst supplies time-dependent flow-relaxation boundary values. Verification
uses sea level and **depth-averaged** velocity, not surface velocity. This is a
one-way parent-model fidelity test, not independent validation or a perfect-model
twin: unresolved three-dimensional dynamics and unmatched forcing can produce
legitimate differences.

### Input preparation and local execution

Stage compatible Norkyst Z-level files containing `h`, `zeta`, `ubar`, `vbar`,
`X`, `Y`, `angle`, and `lat`, and `Uwind`/`Vwind` when wind is enabled. The
default Lofoten bounds are `1400 1900 450 750`, at native resolution. Components
must be collocated and in the source grid axes, matching the existing Norkyst
initializer. Check that convention for the chosen product; no rotation is
guessed. Coordinates must be uniformly spaced metres, elevation/depth metres,
and velocity metres per second. Cell areas are projected `dx*dy`, not native
ROMS metric areas. Grid angle must be labelled in radians and latitude in degrees.
Multi-rank boundary/forcing fields must be globally sampled or spatially constant;
unsupported varying low-resolution forcing is rejected rather than incorrectly
replicated on each rank.

Require timestamps at both endpoints: 24 hours with hourly verification needs
25 hourly records. Use a compatible continuation if a daily file stops at hour
23. Matching duplicate endpoints are deduplicated; conflicting duplicates,
changing grids/masks, missing wet-cell values, and gaps over one hour fail
explicitly. Do not silently splice successive analysis/forecast cycles.

```bash
# CPU-only preflight; no output directory is created.
uv run --no-sync python validate_norkyst.py \
  --source /data/day-one.nc --source /data/continuation.nc \
  --start 2019-07-16T00:00:00Z --output /results/twin-001 --check-inputs

# Allocated GPU environment with the appropriate GPU extra installed:
uv run --no-sync python validate_norkyst.py \
  --source /data/day-one.nc --source /data/continuation.nc \
  --start 2019-07-16T00:00:00Z --output /results/twin-001

# CPU-only analysis of a notebook or batch run:
MPLBACKEND=Agg uv run --no-sync python validate_norkyst.py --analyse /results/twin-001
```

Winds are required by default, rather than silently replaced by zero. If source
winds are unavailable, explicitly use `--no-wind` and record that mismatch.
Use `--pressure-variable Pair` (or the correct variable name) for available
atmospheric pressure in Pa, hPa, or mbar. Pressure is otherwise not applied;
freshwater forcing is not implemented. `--friction` defaults to the initializer's
`0.003`, not a retrieved Norkyst bottom-friction configuration. All assumptions
are logged and saved. Do not interpret differences from omitted or different
physics as proof of a numerical defect.

Output defaults are hourly fields and ten-minute diagnostics, through exactly
86,400 simulation seconds within a one-microsecond time tolerance. Dynamic CFL
is used by default. `--fixed-dt` checks the CFL limit each step. A shorter,
explicit `--duration 3600` is useful for a pilot, but is not a 24-hour validation.
Output times must coincide with real reference records.

### Interactive notebook

Install the project's Jupyter dependencies in the intended environment, select
its kernel, and edit the notebook's paths. On LUMI, reuse
[`setup_lumi_uv_env.sh`](jobs/setup_lumi_uv_env.sh) and the existing
`GPUOcean LUMI (uv)` kernel inside an allocated GPU session. Set
`RUN_SIMULATION=True` for a new single-rank run, or leave it false to analyse a
saved run on a CPU without GPU/MPI imports. The notebook advances hourly using
the shared runner. Exiting/interruption closes files and marks partial runs
incomplete; restarting integration from saved output is not implemented.

### LUMI Slurm execution

[`jobs/validation/norkyst_twin.slurm`](jobs/validation/norkyst_twin.slurm)
follows the working MPI/container configuration in the strong-scaling jobs.
It uses the updated host checkout through `PYTHONPATH`, an existing compatible
container environment, and pre-staged local input files. It does not install
packages or download datasets on compute ranks. From the repository root:

```bash
export PROJECT_DIR="$PWD"
export CONTAINER=/path/to/gpuocean-lumi.sif
export INPUT_DIR=/scratch/project_465002898/norkyst
export OUTPUT_ROOT=/scratch/project_465002898/twin-results

sbatch jobs/validation/norkyst_twin.slurm \
  --source day-one.nc --source continuation.nc --start 2019-07-16T00:00:00Z

# Same global experiment on two GPUs, not weak scaling:
sbatch --ntasks-per-node=2 --gpus-per-node=2 \
  jobs/validation/norkyst_twin.slurm \
  --source day-one.nc --source continuation.nc --start 2019-07-16T00:00:00Z
```

Override the account/partition/resources as appropriate for the allocation.
Source paths are relative to `INPUT_DIR` (absolute paths inside that bound
directory also work). Outputs go directly to `OUTPUT_ROOT/norkyst-twin-JOBID`.
MPI is used, not NCCL. The default one-GPU allocation is intended for Lofoten;
choose walltime after a pilot. Batch plotting is a separate one-task step.

### Results, budgets and manual acceptance checks

Each new output directory contains:

* `fields.nc`: model/reference snapshots, actual UTC/elapsed times, coordinates,
  native and reconstructed bathymetry, latitude/angle, and fixed verification masks.
* `metrics.csv`: regional sea-level/component/vector errors, native/common-depth
  transport errors, and persistence skill. Undefined skill is `nan`, not zero.
* `diagnostics.csv`: volume and density-free momentum integrals, changes,
  timestep ranges, and sampled central-section transports.
* `metadata.json`: resolved settings, source records, revision/dirty state,
  forcing assumptions, MPI size, and complete/incomplete status.
* Comparison maps, skill curves and representative interior-site time series
  generated by analysis.

Primary scores exclude halos, the sponge and an additional buffer. Boundary,
buffer, farther-interior (where available), shallow (<50 m), and offshore
scores help diagnose sensitivity. Area weighting is uniform on the supported
grid. The initial comparison diagnoses mapping error; assess forecast skill
at hours 1--24 separately. Positive skill means improvement over persistence.
Boundary exclusion does not eliminate boundary influence propagated by waves.

`--budget` enables optional, expensive deterministic RK2 accounting in both
CUDA and HIP kernels. It records stage-weighted numerical flux-divergence,
combined physical-source, friction, clipping, and actual boundary-relaxation
increments. Sources combine topographic, Coriolis, wind and pressure effects,
rather than separately attributing each. Residuals are integral change minus
these increments, including float32 update roundoff. Relative residual scales use
the initial absolute sea-level anomaly/momenta, absolute integral change, and
absolute net applied terms (the largest scale); they do not use resting-water
volume. Zero scales produce undefined relative residuals. Halos are excluded and
signed internal-interface divergences cancel when fluxes match. Boundary
accounting synchronizes/downloads state each RK stage, so do not use this mode
for performance comparisons. Without this option, integral changes are **not**
conservation residuals. Source-free periodic conservation remains a prerequisite.
Central-section transports use cell-centred samples, not conservative face
fluxes. Volume is in m3; density-free momentum is in m4/s (multiply by 1025
kg/m3 for physical momentum). Normalize residuals by meaningful exchanged or
anomaly scales rather than only the large resting-water volume.

Local CPU checks:

```bash
uv run --no-sync pytest -q test/norkyst_twin_test.py
bash -n jobs/validation/norkyst_twin.slurm
```

GPU/MPI compilation and execution must be verified manually in an allocation:

```bash
# Existing periodic conservation tests, then new diagnostic/decomposition tests:
uv run --no-sync pytest -q test/mpi_periodic_test.py
GPUOCEAN_RUN_GPU_TESTS=1 uv run --no-sync pytest -q test/norkyst_twin_gpu_test.py
```

The latter starts workers with `mpiexec` for two-rank tests; where nested MPI
launch is unsupported, run that file's `--worker OUTPUT OPEN_BOUNDARY BUDGET`
mode directly through the site's `srun`/container setup, using 0/1 flags, and
compare outputs. Check that enabling diagnostics does not change fields and
that residuals/decomposition differences are small. GPU compilation, LUMI
execution and real-data accuracy are not established by the CPU tests.

Run a short pilot first, then the full 24 hours through both interfaces. Verify
25 snapshots, endpoint coverage, no non-finite wet cells, interface parity and
budget closure. Repeat with `--courant-number 0.4`, and with enlarged `--bounds`
while holding `--verification-bounds 1430 1870 480 720` fixed for the default
Lofoten comparison. Use separate output directories and a multi-GPU consistency
run. Accuracy thresholds should be agreed before judging results; successful
execution alone is not a scientific pass. Repeat across contrasting forcing
conditions before making general claims.

## Academic publications using GPU Ocean
* F. Beiser, H. Holm, K. Lye, J. Eidsvik (2025) **Multi-Level Data Assimilation for Ocean Forecasting using the Shallow-Water Equations**, *Journal of Computational Physics*, vol 524, 113722. DOI: [10.1016/j.jcp.2025.113722](https://doi.org/10.1016/j.jcp.2025.113722)
* F. Beiser, H. Holm, M. Sætra, N. Kristensen, K. Christensen (2024) **Combining Barotropic and Baroclinic Simplified Models for Drift Trajectory Predictions**, *Journal of Operational Oceanography*, 17(3), 187-206. DOI: [10.1080/1755876X.2024.2364975](https://doi.org/10.1080/1755876X.2024.2364975)
* F. Beiser, H. Holm, J. Eidsvik (2024) **Comparison of Ensemble-Based Data Assimilation Methods for Sparse Oceanographic Data**,  *Quarterly Journal of the Royal Meteorological Society*, 150(759), 1068–1095. DOI: [10.1002/qj.4637](https://doi.org/10.1002/qj.4637) [Preprint: [arXiv:2302.07197](https://arxiv.org/abs/2302.07197)]. 
* H. Holm, F. Beiser (2023) **Reducing Numerical Artifacts by Sacrificing Well-Balance for Rotating Shallow-Water Flow**. In: Franck, E., Fuhrmann, J., Michel-Dansac, V., Navoret, L. (eds) *Finite Volumes for Complex Applications X — Volume 2, Hyperbolic and Related Problems*. FVCA 2023. Springer Proceedings in Mathematics & Statistics, vol 433. Springer, Cham. DOI: [10.1007/978-3-031-40860-1_19](https://doi.org/10.1007/978-3-031-40860-1_19)
*	A. Brodtkorb, H. Holm (2021) **Coastal Ocean Forecasting on the GPU using a Two-Dimensional Finite Volume Scheme**. *Tellus A: Dynamic Meteorology and Oceanography*, 73:1, 1-22, DOI: [10.1080/16000870.2021.1876341](https://doi.org/10.1080/16000870.2021.1876341) [Preprint: [arXiv:1912.02457](https://arxiv.org/abs/1912.02457)]
*	H. Holm, A. Brodtkorb, M. Sætra (2020) **Data Assimilation for Ocean Drift Trajectories Using Massive Ensembles and GPUs**. In: Klöfkorn, R., Keilegavlen, E., Radu, F.A., Fuhrmann, J. (eds) *Finite Volumes for Complex Applications IX - Methods, Theoretical Aspects, Examples*. FVCA 2020. Springer Proceedings in Mathematics & Statistics, vol 323. Springer, Cham. DOI: [10.1007/978-3-030-43651-3_68](https://doi.org/10.1007/978-3-030-43651-3_68)
*	H. Holm, M. Sætra, P. van Leeuwen (2020) **Massively Parallel Implicit-Equal Weights Particle Filter for Ocean Drift Trajectory Forecasting**. *Journal of Computational Physics: X*, volume 6, 100053. DOI: [10.1016/j.jcpx.2020.100053](https://doi.org/10.1016/j.jcpx.2020.100053) [Preprint: [arXiv:1910.01031](https://arxiv.org/abs/1910.01031)]
* H. Holm, A. Brodtkorb, K. Christensen, G. Broström, M. Sætra (2020) **Evaluation of Selected Finite-Difference and Finite-Volume Approaches to Rotational Shallow-Water Flow**. *Communications in Computational Physics*, volume 27, pp. 1234-1274. DOI: [10.4208/cicp.OA-2019-0033](https://doi.org/10.4208/cicp.OA-2019-0033)
* H. Holm, A. Brodtkorb, M. Sætra (2020) **GPU Computing with Python: Performance, Energy Efficiency and Usability**. *Computation*, volume 8, number 1:4 (Special issue on Energy-Efficient Computing on Parallel Architectures). DOI: [10.3390/computation8010004](https://doi.org/10.3390/computation8010004). [Preprint: [arXiv:1912.02607](https://arxiv.org/abs/1912.02607)]
*	H. Holm, A. Brodtkorb, M. Sætra (2020) **Performance and Energy Efficiency of CUDA and OpenCL for GPU Computing using Python**. *Advances in Parallel Computing*, volume 36, pp. 593-604. DOI: [10.3233/APC200089](https://doi.org/10.3233/APC200089)



## Development and funding
GPU Ocean is developed through a collaboration between the Norwegian Meteorological Institute and the Applied Computational Science research group at SINTEF Digital. We are greatful for the support from the Norwegian Research Council under grant numbers 250935 (GPU Ocean) and 310515 (Havvarsel).

![image](https://github.com/gpuocean/gpuocean/assets/5363644/34bcbdfa-96d3-4c27-987d-36c9f2007ca3)  <html>&nbsp;&nbsp;&nbsp;</html>    ![image](https://github.com/gpuocean/gpuocean/assets/5363644/267d4675-5a87-4cf6-aa46-a1809b134fe2)



