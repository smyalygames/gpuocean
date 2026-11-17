"""Shared preparation, execution and analysis for a one-way Norkyst twin experiment.

Importing this module and analysing saved results do not require GPU or MPI libraries.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num, num2date

LOG = logging.getLogger(__name__)
TIME_UNITS = "seconds since 1970-01-01 00:00:00 UTC"
FIELDS = ("eta", "u", "v")
BUDGET_TERMS = ("flux", "source", "friction", "clipping", "relaxation")


@dataclass(frozen=True)
class TwinConfig:
    sources: tuple[str, ...]
    output: str
    start: str | None = None
    bounds: tuple[int, int, int, int] = (1400, 1900, 450, 750)
    duration: float = 86400.0
    output_interval: float = 3600.0
    diagnostic_interval: float = 600.0
    sponge_cells: int = 20
    buffer_cells: int = 10
    verification_bounds: tuple[int, int, int, int] | None = None
    courant_number: float = 0.8
    fixed_dt: float | None = None
    friction: float = 0.003
    wind: bool = True
    pressure_variable: str | None = None
    budget: bool = False

    def __post_init__(self):
        if not self.sources:
            raise ValueError("At least one Norkyst source is required")
        for name in ("duration", "output_interval", "diagnostic_interval", "courant_number"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.courant_number > 0.8:
            raise ValueError("Use a courant_number no greater than the solver default, 0.8")
        if self.fixed_dt is not None and (not math.isfinite(self.fixed_dt) or self.fixed_dt <= 0):
            raise ValueError("fixed_dt must be finite and positive")
        if not math.isfinite(self.friction) or self.friction < 0:
            raise ValueError("friction must be finite and nonnegative")
        if self.sponge_cells < 3 or self.buffer_cells < 0:
            raise ValueError("Require sponge_cells >= 3 and buffer_cells >= 0")
        x0, x1, y0, y1 = self.bounds
        if min(x0, y0) < 1 or min(x1 - x0, y1 - y0) < 8:
            raise ValueError("bounds must leave an outer source row and at least eight cells per axis")
        if not str(self.output).strip():
            raise ValueError("An output directory is required")


def schedule(duration, interval):
    """Include both endpoints, even when the interval does not divide the duration."""
    return np.unique(np.append(np.arange(0.0, duration, interval), duration))


def seconds(time_variable):
    calendar = getattr(time_variable, "calendar", "standard")
    if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"Unsupported time calendar: {calendar}")
    values = np.ma.asarray(time_variable[:])
    if np.any(np.ma.getmaskarray(values)) or not np.all(np.isfinite(values)):
        raise ValueError("Missing or non-finite source timestamps")
    dates = num2date(values, time_variable.units, calendar=calendar)
    return np.asarray(date2num(dates, TIME_UNITS), dtype=np.float64)


def _read(variable, key, wet=None):
    values = np.ma.asarray(variable[key])
    valid = ~np.ma.getmaskarray(values) & np.isfinite(values.filled(np.nan))
    if wet is not None and np.any(wet & ~valid):
        raise ValueError(f"Missing/non-finite {variable.name} on wet cells")
    return np.asarray(values.filled(np.nan), dtype=np.float64), valid


def _spacing(coordinates, name):
    differences = np.diff(coordinates)
    if not np.all(np.isfinite(coordinates)) or not np.all(differences > 0):
        raise ValueError(f"{name} coordinates must be finite and increasing")
    if not np.allclose(differences, differences[0], rtol=1e-4, atol=1e-6):
        raise ValueError(f"Nonuniform {name} spacing is not supported by this experiment")
    return float(differences[0])


@dataclass
class PreparedTwin:
    config: TwinConfig
    times: np.ndarray
    records: list[tuple[str, int, float]]
    reference: np.ndarray
    source_depth: np.ndarray
    wet: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    forcing: dict
    grid_fields: dict[str, np.ndarray]

    def reference_at(self, elapsed):
        matches = np.flatnonzero(np.isclose(self.times - self.times[0], elapsed, rtol=0, atol=1e-6))
        if len(matches) != 1:
            raise ValueError(f"No unique Norkyst reference at elapsed time {elapsed}")
        return self.reference[matches[0]]


def prepare(config: TwinConfig) -> PreparedTwin:
    """Read the real source fields and reject incomplete or incompatible inputs.

    Velocity components must already be in the model's grid axes, as required by
    getInitialConditions(norkyst_data=True). No rotation is guessed from names.
    """
    x0, x1, y0, y1 = config.bounds
    region = (slice(y0 + 2, y1 - 2), slice(x0 + 2, x1 - 2))
    candidates = []
    for source in config.sources:
        with Dataset(source) as nc:
            times = seconds(nc["time"])
            if not np.all(np.diff(times) > 0):
                raise ValueError(f"Source timestamps are not strictly increasing: {source}")
            candidates.extend((source, i, float(t)) for i, t in enumerate(times))
    candidates.sort(key=lambda item: item[2])
    if not candidates:
        raise ValueError("Norkyst sources contain no records")
    start = candidates[0][2]
    if config.start is not None:
        date = datetime.fromisoformat(config.start.replace("Z", "+00:00"))
        start = date.replace(tzinfo=timezone.utc).timestamp() if date.tzinfo is None else date.timestamp()
    end = start + config.duration
    candidates = [r for r in candidates if start - 1e-6 <= r[2] <= end + 1e-6]
    if not candidates or abs(candidates[0][2] - start) > 1e-6 or abs(candidates[-1][2] - end) > 1e-6:
        raise ValueError("Sources must include the exact start and final endpoint (25 records for 24 hourly hours)")

    records, reference = [], []
    previous_signature = None
    grid_fields = None
    outer_geometry = None
    depth = wet = x = y = None
    forcing = {"wind": "source Uwind/Vwind" if config.wind else "explicitly disabled",
               "pressure": config.pressure_variable or "not applied",
               "freshwater": "not applied", "velocity_frame": "source grid axes",
               "area_convention": "constant projected dx*dy; no ROMS metric correction"}
    for source, index, timestamp in candidates:
        with Dataset(source) as nc:
            if nc["h"].shape[0] <= y1 or nc["h"].shape[1] <= x1:
                raise ValueError(f"bounds need an additional source row/column outside the domain: {source}")
            if getattr(nc["h"], "units", "").lower() not in ("m", "meter", "meters", "metre", "metres"):
                raise ValueError("Bathymetry units must be metres")
            current_depth, depth_valid = _read(nc["h"], region)
            _, sea_valid = _read(nc["zeta"], (index, *region))
            current_wet = depth_valid & (current_depth > 0) & sea_valid
            current_x = np.asarray(nc["X"][x0 + 2:x1 - 2], dtype=np.float64)
            current_y = np.asarray(nc["Y"][y0 + 2:y1 - 2], dtype=np.float64)
            for name in ("X", "Y"):
                if getattr(nc[name], "units", "").lower() not in ("m", "meter", "meters", "metre", "metres"):
                    raise ValueError(f"{name} units must be metres")
            if wet is None:
                depth, wet, x, y = current_depth, current_wet, current_x, current_y
                if not np.any(wet):
                    raise ValueError("No wet cells in selected source domain")
            elif (not np.array_equal(wet, current_wet) or
                  not np.allclose(depth[wet], current_depth[wet], rtol=0, atol=1e-6) or
                  not np.array_equal(x, current_x) or not np.array_equal(y, current_y)):
                raise ValueError("Grid, bathymetry or land mask changed between source records")
            current_grid = {}
            for name in ("angle", "lat"):
                if nc[name].shape != nc["h"].shape:
                    raise ValueError(f"{name} must be on the source cell-centre grid")
                current_grid[name], _ = _read(nc[name], region, wet)
                units = getattr(nc[name], "units", "").lower()
                accepted = ("radian", "radians", "rad") if name == "angle" else (
                    "degrees_north", "degree_north", "degrees", "degree")
                if units not in accepted:
                    raise ValueError(f"Unsupported {name} units: {units}")
            if grid_fields is None:
                grid_fields = current_grid
            elif any(not np.allclose(current_grid[name][wet], grid_fields[name][wet], rtol=0, atol=1e-8)
                     for name in grid_fields):
                raise ValueError("Grid angle or latitude changed between source records")
            reference_values = []
            for name in ("zeta", "ubar", "vbar"):
                if nc[name].ndim != 3 or nc[name].shape[1:] != nc["h"].shape:
                    raise ValueError(f"{name} must be a collocated (time,Y,X) Norkyst Z-level variable")
                units = getattr(nc[name], "units", "").lower().replace(" ", "")
                accepted = ("m", "meter", "meters", "metre", "metres") if name == "zeta" else (
                    "m/s", "ms-1", "m.s-1", "metersecond-1", "meterssecond-1")
                if units not in accepted:
                    raise ValueError(f"Unsupported {name} units: {nc[name].units}")
                array, _ = _read(nc[name], (index, *region), wet)
                reference_values.append(array)
            if config.wind:
                for name in ("Uwind", "Vwind"):
                    if name not in nc.variables or nc[name].ndim != 3:
                        raise ValueError(f"Missing time-varying {name}; provide forcing or explicitly use --no-wind")
                    if nc[name].shape[1:] != nc["h"].shape:
                        raise ValueError(f"{name} must be on the source cell-centre grid")
                    units = getattr(nc[name], "units", "").lower().replace(" ", "")
                    if units not in ("m/s", "ms-1", "m.s-1", "metersecond-1", "meterssecond-1"):
                        raise ValueError(f"Unsupported {name} units: {units}")
                    _read(nc[name], (index, *region), wet)
            if config.pressure_variable is not None:
                variable = nc[config.pressure_variable]
                if variable.ndim != 3 or variable.shape[1:] != nc["h"].shape:
                    raise ValueError("Atmospheric pressure must be on the source cell-centre grid")
                if variable.units.lower() not in ("pa", "hpa", "mbar"):
                    raise ValueError(f"Unsupported atmospheric pressure units: {variable.units}")
                _read(variable, (index, slice(y0, y1), slice(x0, x1)),
                      np.ones((y1 - y0, x1 - x0), dtype=bool))
            # The initializer also consumes the source ring outside the stored halos.
            outer = (slice(y0 - 1, y1 + 1), slice(x0 - 1, x1 + 1))
            outer_depth, outer_valid = _read(nc["h"], outer)
            _, outer_sea_valid = _read(nc["zeta"], (index, *outer))
            outer_wet = outer_valid & (outer_depth > 0) & outer_sea_valid
            if outer_geometry is None:
                outer_geometry = (outer_depth, outer_wet)
            elif (not np.array_equal(outer_wet, outer_geometry[1]) or
                  not np.allclose(outer_depth, outer_geometry[0], rtol=0, atol=1e-6, equal_nan=True)):
                raise ValueError("Boundary bathymetry or land mask changed between source records")
            signature_fields = [outer_depth]
            for name in ("zeta", "ubar", "vbar", *(("Uwind", "Vwind") if config.wind else ()),
                         *((config.pressure_variable,) if config.pressure_variable is not None else ())):
                values, _ = _read(nc[name], (index, *outer), outer_wet)
                signature_fields.append(values)
            signature = np.stack(signature_fields)
            frame = np.stack(reference_values)
            if np.any((current_depth + frame[0])[wet] <= 0):
                raise ValueError("Nonpositive source water depth on wet cells")
            if records and abs(timestamp - records[-1][2]) < 1e-6:
                if not np.allclose(signature, previous_signature, rtol=0, atol=1e-8, equal_nan=True):
                    raise ValueError("Conflicting duplicate timestamp; do not splice different analysis cycles")
                continue
            records.append((source, index, timestamp))
            reference.append(frame)
            previous_signature = signature
    times = np.array([r[2] for r in records])
    if np.any(np.diff(times) > 3600.0 + 1e-6):
        raise ValueError("Norkyst initial/boundary data must have no gaps longer than one hour")
    prepared = PreparedTwin(config, times, records, np.stack(reference), depth, wet, x, y,
                            _spacing(x, "X"), _spacing(y, "Y"), forcing, grid_fields)
    for elapsed in schedule(config.duration, config.output_interval):
        prepared.reference_at(elapsed)
    verification_masks(prepared, depth)
    LOG.warning("One-way parent-model comparison, not independent validation. Forcing assumptions: %s", forcing)
    return prepared


def verification_masks(prepared, model_depth):
    config = prepared.config
    ny, nx = prepared.wet.shape
    yy, xx = np.mgrid[2:ny + 2, 2:nx + 2]
    distance = np.minimum.reduce((yy, xx, ny + 3 - yy, nx + 3 - xx))
    wet = prepared.wet & np.isfinite(model_depth) & (model_depth < 1e10) & (model_depth > 0)
    core = wet & (distance >= config.sponge_cells + config.buffer_cells)
    if config.verification_bounds is not None:
        x0, _, y0, _ = config.bounds
        a, b, c, d = config.verification_bounds
        if a < x0 + 2 or b > x0 + nx + 2 or c < y0 + 2 or d > y0 + ny + 2:
            raise ValueError("verification_bounds must lie inside the computational interior")
        core &= (xx + x0 >= a) & (xx + x0 < b) & (yy + y0 >= c) & (yy + y0 < d)
    if not np.any(core):
        raise ValueError("No verification cells remain outside sponge/buffer and optional verification bounds")
    return {"wet": wet, "interior": core,
            "boundary_band": wet & (distance < config.sponge_cells + config.buffer_cells),
            "buffer_band": wet & (distance >= config.sponge_cells) &
                           (distance < config.sponge_cells + config.buffer_cells),
            "interior_far": core & (distance >= config.sponge_cells + config.buffer_cells + 20),
            "shallow": core & (prepared.source_depth < 50),
            "offshore": core & (prepared.source_depth >= 50)}


def velocities(state, depth, wet):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (3, *depth.shape):
        raise ValueError("State must have shape (3, ny, nx)")
    if not np.all(np.isfinite(state[:, wet])):
        raise ValueError("Non-finite model values on wet cells")
    h = depth + state[0]
    if np.any(h[wet] <= 0):
        raise ValueError("Nonpositive water depth on verification wet cells")
    result = np.full_like(state, np.nan)
    result[0, wet] = state[0, wet]
    np.divide(state[1:], h, out=result[1:], where=wet[None, :, :])
    if not np.all(np.isfinite(result[:, wet])):
        raise ValueError("Non-finite depth-averaged model velocity")
    return result


def scores(model, reference, initial_reference, mask):
    if not np.any(mask):
        return {}
    error = model[:, mask] - reference[:, mask]
    baseline = initial_reference[:, mask] - reference[:, mask]
    eta_mse = np.mean(error[0] ** 2)
    velocity_mse = np.mean(error[1] ** 2 + error[2] ** 2)
    eta_baseline = np.mean(baseline[0] ** 2)
    velocity_baseline = np.mean(baseline[1] ** 2 + baseline[2] ** 2)
    return {"cells": int(np.count_nonzero(mask)), "eta_bias_m": float(np.mean(error[0])),
            "model_mean_eta_m": float(np.mean(model[0, mask])),
            "reference_mean_eta_m": float(np.mean(reference[0, mask])),
            "eta_rmse_m": float(np.sqrt(eta_mse)),
            "eta_pattern_rmse_m": float(np.std(error[0])),
            "u_bias_m_s": float(np.mean(error[1])), "v_bias_m_s": float(np.mean(error[2])),
            "u_rmse_m_s": float(np.sqrt(np.mean(error[1] ** 2))),
            "v_rmse_m_s": float(np.sqrt(np.mean(error[2] ** 2))),
            "velocity_rmse_m_s": float(np.sqrt(velocity_mse)),
            "persistence_eta_rmse_m": float(np.sqrt(eta_baseline)),
            "persistence_velocity_rmse_m_s": float(np.sqrt(velocity_baseline)),
            "eta_skill": float(1 - eta_mse / eta_baseline) if eta_baseline > 0 else math.nan,
            "velocity_skill": float(1 - velocity_mse / velocity_baseline) if velocity_baseline > 0 else math.nan}


def integrals(state, depth, wet, area):
    return area * np.array([np.sum((depth + state[0])[wet], dtype=np.float64),
                            np.sum(state[1, wet], dtype=np.float64),
                            np.sum(state[2, wet], dtype=np.float64)])


def _initial_arguments(prepared):
    from gpuocean.utils.Common import SpongeCells
    from gpuocean.utils.netcdf.NetCDFInitialization import getInitialConditions, removeMetadata

    sources, indices, time_values = [], [], []
    for source, index, timestamp in prepared.records:
        if not sources or sources[-1] != source:
            sources.append(source)
            indices.append([])
            time_values.append([])
        indices[-1].append(index)
        time_values[-1].append(timestamp)
    config = prepared.config
    arguments = getInitialConditions(
        sources.copy(), *config.bounds, timestep_indices=indices,
        sponge_cells=SpongeCells(*([config.sponge_cells] * 4)), wind_forcing=config.wind,
        time_values=time_values)
    elapsed = (prepared.times - prepared.times[0]).tolist()
    arguments["boundary_conditions_data"].t = elapsed
    arguments["wind"].t = elapsed if config.wind else [0]
    if not config.wind:
        from gpuocean.utils.WindStress import WindStress
        arguments["wind"] = WindStress()
    if config.pressure_variable is not None:
        from gpuocean.utils.AtmosphericPressure import AtmosphericPressure
        x0, x1, y0, y1 = config.bounds
        pressures = []
        for source, index, _ in prepared.records:
            with Dataset(source) as nc:
                variable = nc[config.pressure_variable]
                units = variable.units.lower()
                if units not in ("pa", "hpa", "mbar"):
                    raise ValueError(f"Unsupported atmospheric pressure units: {variable.units}")
                values = np.ma.asarray(variable[index, y0:y1, x0:x1])
                if np.any(np.ma.getmaskarray(values)) or not np.all(np.isfinite(values)):
                    raise ValueError("Atmospheric pressure must be finite over the full domain")
                pressures.append(np.asarray(values, dtype=np.float32) * (1 if units == "pa" else 100))
        arguments["atmospheric_pressure"] = AtmosphericPressure(t=elapsed, P=pressures)
    arguments.update(r=config.friction, courant_number=config.courant_number, write_netcdf=False,
                     budget_diagnostics=config.budget, dt=0.0)
    if config.budget:
        arguments["budget_wet_mask"] = np.pad(prepared.wet, 2, constant_values=False)
    return removeMetadata(arguments)


class TwinOutput:
    def __init__(self, prepared, depth, masks, ranks):
        self.path = Path(prepared.config.output)
        self.path.mkdir(parents=True, exist_ok=False)
        self.prepared, self.depth, self.masks = prepared, depth, masks
        revision = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        self.metadata = {"status": "incomplete", "config": asdict(prepared.config), "git_revision": revision,
                         "forcing": prepared.forcing, "mpi_ranks": ranks,
                         "records": prepared.records, "time_units": TIME_UNITS,
                         "initial_time": float(prepared.times[0]),
                         "budget_terms": BUDGET_TERMS if prepared.config.budget else [],
                         "budget_normalization": "max(initial sum(abs(eta,hu,hv))*area, "
                                                 "abs(integral change), sum(abs(net applied terms))); "
                                                 "zero scale yields undefined relative residual",
                         "momentum_convention": "density-free; multiply by 1025 kg/m3 for physical momentum"}
        self.metadata["git_dirty"] = bool(subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip())
        self._metadata()
        self.nc = Dataset(self.path / "fields.nc", "w")
        ny, nx = depth.shape
        for name, length in (("time", None), ("y", ny), ("x", nx)):
            self.nc.createDimension(name, length)
        self.nc.setncatts({"title": "GPU Ocean one-way Norkyst twin experiment",
                          "git_revision": revision, "status": "incomplete",
                          "velocity_frame": "source grid axes", "area_convention": prepared.forcing["area_convention"]})
        for name, values, units in (("x", prepared.x, "m"), ("y", prepared.y, "m")):
            variable = self.nc.createVariable(name, "f8", (name,))
            variable.units = units
            variable[:] = values
        time = self.nc.createVariable("time", "f8", ("time",))
        time.units, time.calendar = TIME_UNITS, "standard"
        self.nc.createVariable("elapsed_seconds", "f8", ("time",)).units = "s"
        for name, values in (("model_depth", depth), ("source_depth", prepared.source_depth)):
            variable = self.nc.createVariable(name, "f8", ("y", "x"), fill_value=np.nan)
            variable.units = "m"
            variable[:] = np.where(masks["wet"], values, np.nan)
        for name, values in prepared.grid_fields.items():
            variable = self.nc.createVariable(name, "f8", ("y", "x"), fill_value=np.nan)
            variable.units = "radians" if name == "angle" else "degrees_north"
            variable[:] = np.where(masks["wet"], values, np.nan)
        for name, values in masks.items():
            self.nc.createVariable(f"mask_{name}", "i1", ("y", "x"))[:] = values
        for prefix, names in (("model", ("eta", "hu", "hv", "u", "v")), ("reference", FIELDS)):
            for name in names:
                variable = self.nc.createVariable(f"{prefix}_{name}", "f4", ("time", "y", "x"),
                                                  zlib=True, fill_value=np.nan)
                variable.units = "m" if name == "eta" else ("m2 s-1" if name in ("hu", "hv") else "m s-1")
        self.metrics_file = (self.path / "metrics.csv").open("x", newline="")
        self.diagnostics_file = (self.path / "diagnostics.csv").open("x", newline="")
        self.metrics_writer = self.diagnostics_writer = None
        self.initial_totals = None
        self.initial_scales = None

    def _metadata(self):
        temporary = self.path / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path / "metadata.json")

    def record(self, elapsed, state, iterations, min_dt, max_dt, snapshot, budget):
        wet = self.masks["wet"]
        model = velocities(state, self.depth, wet)
        area = self.prepared.dx * self.prepared.dy
        totals = integrals(state, self.depth, wet, area)
        if not np.all(np.isfinite(totals)):
            raise RuntimeError("Non-finite volume or momentum integral")
        if budget is not None and not np.all(np.isfinite(budget)):
            raise RuntimeError("Non-finite discrete budget increment")
        if self.initial_totals is None:
            self.initial_totals = totals.copy()
            self.initial_scales = area * np.sum(np.abs(state[:, wet]), axis=1, dtype=np.float64)
        row = dict(time_s=elapsed, iterations=iterations, min_dt_s=min_dt, max_dt_s=max_dt,
                   volume_m3=totals[0], momentum_x_m4_s=totals[1], momentum_y_m4_s=totals[2])
        delta = totals - self.initial_totals
        for i, name in enumerate(("volume", "momentum_x", "momentum_y")):
            row[f"{name}_change"] = delta[i]
            if budget is not None:
                for term, values in zip(BUDGET_TERMS, budget):
                    row[f"{name}_{term}_increment"] = area * values[i]
                residual = delta[i] - area * budget[:, i].sum()
                scale = max(self.initial_scales[i], abs(delta[i]), area * np.sum(np.abs(budget[:, i])))
                row[f"{name}_budget_residual"] = residual
                row[f"{name}_budget_scale"] = scale
                row[f"{name}_budget_relative_residual"] = residual / scale if scale > 0 else math.nan
        ny, nx = wet.shape
        row["sampled_central_x_transport_m3_s"] = self.prepared.dy * np.sum(
            state[1, :, nx // 2][wet[:, nx // 2]], dtype=np.float64)
        row["sampled_central_y_transport_m3_s"] = self.prepared.dx * np.sum(
            state[2, ny // 2, :][wet[ny // 2, :]], dtype=np.float64)
        if self.diagnostics_writer is None:
            self.diagnostics_writer = csv.DictWriter(self.diagnostics_file, fieldnames=list(row))
            self.diagnostics_writer.writeheader()
        self.diagnostics_writer.writerow(row)
        self.diagnostics_file.flush()
        if not snapshot:
            return
        reference = self.prepared.reference_at(elapsed)
        # Native-depth transport and common-depth transport are deliberately distinguished.
        native_h = self.prepared.source_depth + reference[0]
        common_h = self.depth + reference[0]
        rows = []
        for region, mask in self.masks.items():
            if region == "wet" or not np.any(mask):
                continue
            result = scores(model, reference, self.prepared.reference[0], mask)
            native_error = state[1:] - reference[1:] * native_h
            common_error = state[1:] - reference[1:] * common_h
            result["native_transport_rmse_m2_s"] = float(np.sqrt(np.mean(np.sum(native_error[:, mask] ** 2, axis=0))))
            result["common_depth_transport_rmse_m2_s"] = float(np.sqrt(np.mean(np.sum(common_error[:, mask] ** 2, axis=0))))
            rows.append(dict(time_s=elapsed, region=region, **result))
        if self.metrics_writer is None:
            self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=list(rows[0]))
            self.metrics_writer.writeheader()
        self.metrics_writer.writerows(rows)
        self.metrics_file.flush()
        index = len(self.nc["time"])
        self.nc["time"][index] = self.prepared.times[0] + elapsed
        self.nc["elapsed_seconds"][index] = elapsed
        for name, values in zip(("eta", "hu", "hv", "u", "v"), (*state, *model[1:])):
            self.nc[f"model_{name}"][index] = np.where(wet, values, np.nan)
        for name, values in zip(FIELDS, reference):
            self.nc[f"reference_{name}"][index] = np.where(wet, values, np.nan)
        self.nc.sync()

    def close(self, complete=False):
        self.nc.status = "complete" if complete else "incomplete"
        self.nc.close()
        self.metrics_file.close()
        self.diagnostics_file.close()
        self.metadata["status"] = "complete" if complete else "incomplete"
        self._metadata()


class TwinExperiment:
    """Collective MPI lifecycle; in a notebook use one rank and a context manager."""

    def __init__(self, config: TwinConfig, prepared: PreparedTwin | None = None, comm=None):
        self.config, self.prepared, self.comm = config, prepared, comm
        self.sim = self.writer = self.gpu_ctx = None
        self.complete = False
        self.min_dt, self.max_dt = math.inf, 0.0
        self.last_record = 0.0
        self.targets = np.union1d(schedule(config.duration, config.output_interval),
                                 schedule(config.duration, config.diagnostic_interval))
        self.snapshots = schedule(config.duration, config.output_interval)

    def _root(self, function):
        result, error = None, None
        if self.comm.rank == 0:
            try:
                result = function()
            except Exception as exc:
                LOG.exception("Twin experiment rank-zero operation failed")
                error = f"{type(exc).__name__}: {exc}"
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(error)
        return result

    def __enter__(self):
        try:
            return self._start()
        except BaseException:
            if self.writer is not None:
                self.writer.close(False)
            if self.sim is not None:
                self.sim.cleanUp()
            raise

    def _start(self):
        from mpi4py import MPI
        from gpuocean.SWEsimulators import SimulatorType
        from gpuocean.utils.gpu import KernelContext, gpu_device
        from gpuocean.utils.mpi import MPIWrapper

        if self.comm is None:
            self.comm = MPI.COMM_WORLD
        if self.prepared is not None and self.prepared.config != self.config:
            raise ValueError("Prepared data and experiment configuration differ")
        self.prepared = self._root(lambda: self.prepared if self.prepared is not None else prepare(self.config))
        self._root(lambda: self._check_output())
        arguments = self._root(lambda: _initial_arguments(self.prepared))
        arguments = self.comm.bcast(arguments, root=0)
        self.gpu_ctx = KernelContext(device=gpu_device.get_device_for_rank(self.comm.rank))
        nx, ny = arguments.pop("nx"), arguments.pop("ny")
        self.sim = MPIWrapper(SimulatorType.CDKLM16, nx, ny, (2, 2, 2, 2), comm=self.comm,
                              use_nccl=False, gpu_ctx=self.gpu_ctx, **arguments)
        self.sim.sim.t = np.float64(0)
        # Record the initialized state before the solver's first boundary application.
        if self.config.budget:
            self.sim.sim.budget.reset()
        local_depth = self.sim.sim.downloadBathymetry(interior_domain_only=True)[1]
        pieces = self.comm.gather((self.sim.grid.x0, self.sim.grid.x1, self.sim.grid.y0, self.sim.grid.y1,
                                   np.ma.filled(local_depth, np.nan)), root=0)

        def create_writer():
            depth = np.empty((ny, nx))
            for x0, x1, y0, y1, values in pieces:
                depth[y0:y1, x0:x1] = values
            masks = verification_masks(self.prepared, depth)
            return TwinOutput(self.prepared, depth, masks, self.comm.size)

        self.writer = self._root(create_writer)
        if self.config.budget:
            # Budget and integral masks must describe exactly the same control volume.
            masks = self.comm.bcast(self.writer.masks["wet"] if self.comm.rank == 0 else None, root=0)
            grid = self.sim.grid
            self.sim.sim.budget.wet = masks[grid.y0:grid.y1, grid.x0:grid.x1]
        self._record(0.0)
        return self

    def _check_output(self):
        if Path(self.config.output).exists():
            raise FileExistsError(f"Refusing to overwrite output directory: {self.config.output}")

    def _record(self, elapsed):
        state = self.sim.download(interior_domain_only=True)
        budget = None
        if self.config.budget:
            local = self.sim.sim.budget.totals()
            budget = np.empty_like(local)
            self.comm.Allreduce(local, budget)
        self._root(lambda: self.writer.record(
            elapsed, np.asarray(state), self.sim.sim.num_iterations,
            self.min_dt if math.isfinite(self.min_dt) else 0.0, self.max_dt,
            np.any(np.isclose(self.snapshots, elapsed, rtol=0, atol=1e-6)), budget))
        self.last_record = elapsed

    @property
    def time(self):
        return float(self.sim.sim.t)

    def advance_to(self, target):
        if not math.isfinite(target) or target < self.time or target > self.config.duration:
            raise ValueError("target must be finite, nondecreasing and within the configured duration")
        checkpoints = self.targets[(self.targets > self.last_record + 1e-6) & (self.targets <= target + 1e-6)]
        for endpoint in np.union1d(checkpoints, [target]):
            while endpoint - self.time > 1e-7:
                self.sim.update_dt()
                limit = float(self.sim.sim.dt)
                if not math.isfinite(limit) or limit <= 0:
                    raise RuntimeError(f"Invalid CFL timestep {limit} at {self.time}")
                requested = limit if self.config.fixed_dt is None else self.config.fixed_dt
                if requested > limit:
                    raise RuntimeError(f"Fixed timestep {requested} exceeds CFL limit {limit}")
                remaining = endpoint - self.time
                bound = min(requested, remaining)
                step = np.float32(bound)
                if float(step) > bound:
                    step = np.nextafter(step, np.float32(0))
                if step <= 0:
                    raise RuntimeError("Timestep rounded to zero before endpoint")
                before = self.time
                self.sim.sim.dt = float(step)
                self.sim.sim.step(float(step), write_now=False, apply_stochastic_term=False, split_step=False)
                if self.time <= before or self.time > endpoint + 1e-6:
                    raise RuntimeError(f"Invalid simulation-time advance from {before} to {self.time}")
                self.min_dt, self.max_dt = min(self.min_dt, float(step)), max(self.max_dt, float(step))
            if abs(self.time - endpoint) > 1e-6:
                raise RuntimeError(f"Failed to reach scheduled time {endpoint}: {self.time}")
            if np.any(np.isclose(self.targets, endpoint, rtol=0, atol=1e-6)) and endpoint > self.last_record + 1e-6:
                self._record(self.time)
        self.complete = abs(self.time - self.config.duration) <= 1e-6

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._root(lambda: self.writer.close(self.complete and exc_type is None) if self.writer is not None else None)
        finally:
            if self.sim is not None:
                self.sim.cleanUp()
        return False


def load_results(directory):
    """Load saved results on a CPU; incomplete runs remain explicitly identified."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    with Dataset(directory / "fields.nc") as nc:
        arrays = {name: np.ma.filled(variable[:], np.nan) if variable.dtype.kind == "f"
                  else np.asarray(variable[:]) for name, variable in nc.variables.items()}
    if metadata["status"] != "complete":
        LOG.warning("Loading incomplete experiment: %s", directory)
    return metadata, arrays


def plot_results(directory, elapsed=None, save=False):
    """Use the same comparison plots in batch analysis and in the notebook."""
    from matplotlib import pyplot as plt

    metadata, arrays = load_results(directory)
    if not len(arrays["time"]):
        raise ValueError("No snapshots available")
    elapsed = arrays["elapsed_seconds"][-1] if elapsed is None else elapsed
    candidates = np.flatnonzero(np.isclose(arrays["elapsed_seconds"], elapsed, atol=1e-6, rtol=0))
    if len(candidates) != 1:
        raise ValueError(f"No unique saved snapshot at {elapsed} seconds")
    index = candidates[0]
    figures = []
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for row, name in enumerate(FIELDS):
        model, reference = arrays[f"model_{name}"][index], arrays[f"reference_{name}"][index]
        both = np.concatenate((model.ravel(), reference.ravel()))
        low, high = np.nanmin(both), np.nanmax(both)
        difference = model - reference
        scale = max(float(np.nanmax(np.abs(difference))), np.finfo(float).eps)
        for col, (values, title) in enumerate(((model, "GPU Ocean"), (reference, "Norkyst"),
                                               (difference, "GPU Ocean - Norkyst"))):
            image = axes[row, col].imshow(values, origin="lower", cmap="RdBu_r" if col == 2 else "viridis",
                                          vmin=-scale if col == 2 else low, vmax=scale if col == 2 else high)
            axes[row, col].set_title(f"{title}: {name} ({'m' if name == 'eta' else 'm/s'})")
            fig.colorbar(image, ax=axes[row, col], shrink=0.8)
    fig.suptitle(f"{elapsed / 3600:g} hours; run {metadata['status']}")
    figures.append(fig)
    with (Path(directory) / "metrics.csv").open() as file:
        rows = [r for r in csv.DictReader(file) if r["region"] == "interior"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, name, label in zip(axes, ("eta", "velocity"), ("Sea-level RMSE (m)", "Vector velocity RMSE (m/s)")):
        suffix = "m" if name == "eta" else "m_s"
        hours = [float(r["time_s"]) / 3600 for r in rows]
        ax.plot(hours, [float(r[f"{name}_rmse_{suffix}"]) for r in rows], color="tab:blue", label="GPU Ocean")
        ax.plot(hours, [float(r[f"persistence_{name}_rmse_{suffix}"]) for r in rows],
                color="tab:orange", label="Persistence")
        ax.set(xlabel="Elapsed hours", ylabel=label)
        ax.legend()
    figures.append(fig)
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
    wet = arrays["mask_interior"].astype(bool)
    iy, ix = np.nonzero(wet)
    if not len(iy):
        raise ValueError("Saved experiment has no interior verification cells")
    ny, nx = wet.shape
    for column, fraction in enumerate((0.25, 0.5, 0.75)):
        nearest = np.argmin(((ix - fraction * nx) / nx) ** 2 + ((iy - 0.5 * ny) / ny) ** 2)
        y, x = iy[nearest], ix[nearest]
        for row, name in enumerate(FIELDS):
            ax = axes[row, column]
            for prefix, color, label in (("model", "tab:blue", "GPU Ocean"), ("reference", "tab:orange", "Norkyst")):
                ax.plot(arrays["elapsed_seconds"] / 3600, arrays[f"{prefix}_{name}"][:, y, x],
                        color=color, label=label)
            ax.set(xlabel="Elapsed hours", ylabel=f"{name} ({'m' if name == 'eta' else 'm/s'})")
            ax.set_title(f"x={arrays['x'][x] / 1000:g}, y={arrays['y'][y] / 1000:g} km")
            ax.legend()
    figures.append(fig)
    if save:
        for name, fig in zip((f"fields_{elapsed:g}s.png", "skill_timeseries.png", "interior_timeseries.png"), figures):
            fig.savefig(Path(directory) / name, dpi=150)
    return figures
