"""CPU tests use small, explicitly synthetic NetCDF fixtures, never replacement experiment data."""

import csv
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from gpuocean.utils.Common import BoundaryConditionsData, SingleBoundaryConditionData
from gpuocean.utils.norkyst_twin import (
    TwinConfig, TwinExperiment, TwinOutput, integrals, load_results, plot_results,
    prepare, schedule, scores, velocities, verification_masks,
)


def source_file(path, hours, wind=True):
    with Dataset(path, "w") as nc:
        for name, size in (("time", len(hours)), ("Y", 30), ("X", 34)):
            nc.createDimension(name, size)
        nc.createVariable("time", "f8", ("time",))[:] = hours
        nc["time"].units = "hours since 2019-07-16 00:00:00"
        for name, length in (("X", 34), ("Y", 30)):
            variable = nc.createVariable(name, "f8", (name,))
            variable.units = "m"
            variable[:] = np.arange(length) * 800.0
        depth = nc.createVariable("h", "f4", ("Y", "X"), fill_value=-9999)
        depth.units = "m"
        depth[:] = 60
        depth[10, 12] = np.ma.masked
        for name, value in (("angle", 0.1), ("lat", 68.0)):
            variable = nc.createVariable(name, "f8", ("Y", "X"))
            variable[:] = value
            variable.units = "radians" if name == "angle" else "degrees_north"
        for name, unit, offset, rate in (
                ("zeta", "m", 0, 0.01), ("ubar", "m s-1", 0.2, 0.001), ("vbar", "m s-1", -0.1, 0)):
            variable = nc.createVariable(name, "f4", ("time", "Y", "X"), fill_value=-9999)
            variable.units = unit
            variable[:] = np.asarray([offset + rate * t for t in hours])[:, None, None]
            variable[:, 10, 12] = np.ma.masked
        if wind:
            for name in ("Uwind", "Vwind"):
                variable = nc.createVariable(name, "f4", ("time", "Y", "X"))
                variable.units = "m s-1"
                variable[:] = 2
    return str(path)


@pytest.fixture
def config(tmp_path):
    source = source_file(tmp_path / "source.nc", range(25))
    return TwinConfig((source,), str(tmp_path / "result"), bounds=(2, 30, 2, 26),
                      sponge_cells=3, buffer_cells=1)


def test_prepare_24_hours_and_exact_endpoint(config, tmp_path):
    prepared = prepare(config)
    assert len(prepared.times) == 25
    assert prepared.times[-1] - prepared.times[0] == 86400
    assert prepared.reference.shape == (25, 3, 20, 24)
    assert not prepared.wet[6, 8]
    np.testing.assert_allclose(prepared.reference_at(12 * 3600)[0, prepared.wet], 0.12)
    assert not (tmp_path / "result").exists()
    np.testing.assert_array_equal(schedule(11, 4), [0, 4, 8, 11])


def test_continuation_deduplicates_and_supports_start_offset(config, tmp_path):
    following = source_file(tmp_path / "continuation.nc", range(24, 49))
    with Dataset(following, "a") as nc:
        nc["time"].units = "hours since 2019-07-17 00:00:00"
        nc["time"][:] = np.arange(25)
    prepared = prepare(replace(config, sources=(*config.sources, following), start="2019-07-16T12:00:00Z"))
    assert len(prepared.records) == 25
    assert [r[1] for r in prepared.records][:2] == [12, 13]
    np.testing.assert_allclose(prepared.reference_at(86400)[0, prepared.wet], 0.36)
    with Dataset(following, "a") as nc:
        nc["zeta"][0, 5, 5] = 9
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare(replace(config, sources=(*config.sources, following)))


@pytest.mark.parametrize("kind,message", [
    ("endpoint", "exact start and final endpoint"), ("gap", "gaps longer than one hour"),
    ("missing", "Missing/non-finite ubar"), ("wind", "Missing time-varying Uwind"),
    ("units", "Unsupported ubar units"), ("grid", "Nonuniform X spacing"),
    ("angle", "Unsupported angle units"),
    ("boundary_mask", "Boundary bathymetry or land mask changed"),
])
def test_rejects_bad_inputs(config, tmp_path, kind, message):
    if kind == "endpoint":
        config = replace(config, sources=(source_file(tmp_path / "short.nc", range(24)),))
    elif kind == "gap":
        config = replace(config, sources=(source_file(tmp_path / "gap.nc", [*range(12), *range(13, 25)]),))
    else:
        with Dataset(config.sources[0], "a") as nc:
            if kind == "missing":
                nc["ubar"][12, 5, 5] = np.ma.masked
            elif kind == "wind":
                nc.renameVariable("Uwind", "unexpected_wind")
            elif kind == "units":
                nc["ubar"].units = "cm/s"
            elif kind == "grid":
                nc["X"][5] += 20
            elif kind == "angle":
                nc["angle"].units = "degrees"
            elif kind == "boundary_mask":
                nc["zeta"][12, 1, 5] = np.ma.masked
    with pytest.raises(ValueError, match=message):
        prepare(config)
    if kind == "wind":
        assert prepare(replace(config, wind=False)).forcing["wind"] == "explicitly disabled"


def test_initialization_and_metrics_use_depth_averaged_velocities(config):
    prepared = prepare(config)
    depth = np.full(prepared.wet.shape, 65.0)
    masks = verification_masks(prepared, depth)
    reference = prepared.reference_at(3600)
    state = np.stack((reference[0], (depth + reference[0]) * reference[1],
                      (depth + reference[0]) * reference[2]))
    model = velocities(state, depth, masks["wet"])
    result = scores(model, reference, prepared.reference[0], masks["interior"])
    assert result["velocity_rmse_m_s"] == pytest.approx(0, abs=1e-14)
    assert result["eta_skill"] == pytest.approx(1)
    undefined = scores(prepared.reference[0], prepared.reference[0], prepared.reference[0], masks["interior"])
    assert np.isnan(undefined["eta_skill"])
    model[1, masks["interior"]] += 0.03
    model[2, masks["interior"]] += 0.04
    assert scores(model, reference, prepared.reference[0], masks["interior"])["velocity_rmse_m_s"] == pytest.approx(0.05)
    state[1, 5, 5] = np.nan
    with pytest.raises(ValueError, match="Non-finite model"):
        velocities(state, depth, masks["wet"])
    assert not np.any(masks["interior"] & masks["boundary_band"])


@pytest.mark.parametrize("complete", [False, True])
def test_output_reload_and_plotting_are_cpu_only(config, complete):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    prepared = prepare(replace(config, duration=3600, budget=True))
    depth = np.full(prepared.wet.shape, 60.0)
    masks = verification_masks(prepared, depth)
    writer = TwinOutput(prepared, depth, masks, ranks=1)
    states = []
    for elapsed in (0, 3600):
        ref = prepared.reference_at(elapsed)
        state = np.stack((ref[0], (depth + ref[0]) * ref[1], (depth + ref[0]) * ref[2]))
        states.append(state)
        budget = np.zeros((5, 3))
        budget[0] = (integrals(state, depth, masks["wet"], 1) -
                     integrals(states[0], depth, masks["wet"], 1))
        with pytest.raises(RuntimeError, match="Non-finite discrete budget"):
            writer.record(elapsed, state, elapsed // 10, 10, 10, True, np.full((5, 3), np.nan))
        writer.record(elapsed, state, elapsed // 10, 10, 10, True, budget)
    writer.close(complete=complete)
    metadata, arrays = load_results(config.output)
    assert metadata["status"] == ("complete" if complete else "incomplete")
    np.testing.assert_allclose(arrays["elapsed_seconds"], [0, 3600])
    assert np.isnan(arrays["model_u"][:, ~masks["wet"]]).all()
    with (writer.path / "diagnostics.csv").open() as file:
        diagnostics = list(csv.DictReader(file))
    assert abs(float(diagnostics[-1]["volume_budget_residual"])) <= 4 * np.spacing(
        float(diagnostics[-1]["volume_m3"]))
    with pytest.raises(FileExistsError):
        TwinOutput(prepared, depth, masks, ranks=1)
    figures = plot_results(config.output, save=True)
    assert len(figures) == 3
    for figure in figures:
        plt.close(figure)
    assert (writer.path / "interior_timeseries.png").exists()


def test_spatial_boundary_data_is_sliced_without_mutating_global_series():
    def boundary(length):
        field = np.arange(2 * length, dtype=np.float32).reshape(2, length)
        return SingleBoundaryConditionData(field, field + 100, field + 200)

    data = BoundaryConditionsData([0, 3600], north=boundary(36), south=boundary(36),
                                  east=boundary(28), west=boundary(28))
    local = data.subdomain(12, 30, 8, 24, (28, 36))
    np.testing.assert_array_equal(local.north.h, data.north.h[:, 12:30])
    np.testing.assert_array_equal(local.east.hu, data.east.hu[:, 8:24])
    assert local.north.h.shape == (2, 18)
    assert data.north.h.shape == (2, 36)
    uniform = BoundaryConditionsData()
    assert uniform.subdomain(12, 30, 8, 24, (28, 36)).north is uniform.north
    with pytest.raises(ValueError, match="within the global grid"):
        data.subdomain(12, 40, 8, 24, (28, 36))


def test_spatial_forcing_is_sliced_without_double_shifting_pressure():
    from gpuocean.utils.WindStress import WindStress
    from gpuocean.utils.AtmosphericPressure import AtmosphericPressure

    field = np.arange(28 * 36, dtype=np.float32).reshape(28, 36)
    wind = WindStress(t=[0, 3600], wind_u=[field, field + 1], wind_v=[field + 2, field + 3])
    wind.compute_wind_stress_from_wind()
    local = wind.subdomain(12, 30, 8, 24, (28, 36))
    np.testing.assert_array_equal(local.wind_u[0], field[8:24, 12:30])
    np.testing.assert_array_equal(local.stress_v[1], wind.stress_v[1][8:24, 12:30])
    pressure = AtmosphericPressure(t=[0, 3600], P=[100000 + field, 100000 + field + 10])
    original = pressure.getOriginalP()
    local = pressure.subdomain(12, 30, 8, 24, (28, 36))
    np.testing.assert_array_equal(local.getOriginalP()[1], original[1][8:24, 12:30])
    np.testing.assert_array_equal(pressure.getOriginalP()[0], original[0])
    assert WindStress().subdomain(12, 30, 8, 24, (28, 36)).wind_u[0].shape == (1, 1)


def test_incremental_scheduler_does_not_overshoot_or_skip_outputs(config):
    experiment = TwinExperiment(replace(config, duration=7200))
    checkpoints, steps = [], []

    class Solver:
        t = 0.0
        dt = 1300.7

        def step(self, elapsed, **kwargs):
            assert elapsed <= self.dt
            steps.append(elapsed)
            self.t += elapsed

    solver = Solver()
    experiment.sim = SimpleNamespace(sim=solver, update_dt=lambda: setattr(solver, "dt", 1300.7))

    def record(elapsed):
        checkpoints.append(elapsed)
        experiment.last_record = elapsed

    experiment._record = record
    experiment.advance_to(3600)
    experiment.advance_to(7200)
    np.testing.assert_allclose(checkpoints, np.arange(600, 7201, 600), atol=1e-6)
    assert experiment.complete
    assert sum(steps) == pytest.approx(7200, abs=1e-6)
    experiment.config = replace(experiment.config, fixed_dt=2000)
    solver.t = 0
    with pytest.raises(RuntimeError, match="exceeds CFL"):
        experiment.advance_to(600)


def test_cpu_cli_preflight(config):
    from pathlib import Path
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "validate_norkyst.py"),
               "--source", config.sources[0], "--output", config.output, "--check-inputs",
               "--bounds", *map(str, config.bounds), "--sponge-cells", "3", "--buffer-cells", "1"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Preflight passed: 25 source timestamps" in result.stderr
    assert not Path(config.output).exists()
