"""Consolidated on-disk representation of the processed data.

One netCDF file per (year, pipeline):

    data/processed/processed_2019_causal.nc

holding the nine satellite features, the five CARRA drivers, and the per-lake
event table on a shared ``lake`` dimension. Everything a downstream stage needs
arrives aligned, so nothing has to join arrays by index and nothing can drift.

netCDF rather than ``.npz`` because the inputs are already netCDF, because the
dimensions and units travel with the data, and because the file can carry its
own provenance in the global attributes -- the fill constants, the earth radius,
the pipeline flag. A tensor should not be separable from how it was made.

Long-format result tables (the sweep) stay in parquet: they are tables, not
grids, and forcing them onto a lake x day x variable mesh would be the wrong
shape for the format.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

from . import __version__
from .carra import CARRA_VARS, EARTH_RADIUS_M
from .preprocess import DERIVED, WINDOW, build_events, build_tensor

# Written into string variables where the quantity does not apply to that lake.
NOT_APPLICABLE = "N/A"

UNITS = {
    "HV_lake": ("dB", "Sentinel-1 HV backscatter over the lake"),
    "HV_out": ("dB", "Sentinel-1 HV backscatter over surrounding ice"),
    "HV_anom": ("dB", "lake-minus-surroundings backscatter anomaly"),
    "S2_water": ("1", "Sentinel-2 water fraction"),
    "LS_water": ("1", "Landsat water fraction"),
    "pwater": ("1", "combined optical water fraction"),
    "t2m": ("K", "2 m air temperature"),
    "S2_zenith": ("degree", "Sentinel-2 solar zenith angle"),
    "LS_zenith": ("degree", "Landsat solar zenith angle"),
    "runoff": ("kg m-2", "CARRA surface runoff, 24 h accumulation"),
    "albedo": ("1", "CARRA forecast albedo (rescaled from percent)"),
    "rh2m": ("%", "CARRA 2 m relative humidity"),
    "sde_swe": ("kg m-2", "CARRA snow depth water equivalent"),
    "sp": ("Pa", "CARRA surface pressure"),
}

ALL_VARS = DERIVED + CARRA_VARS


def build_dataset(cfg, data, pipeline: str, carra: np.ndarray, events: pd.DataFrame) -> xr.Dataset:
    """Assemble one (year, pipeline) dataset from its three parts."""
    features = build_tensor(data, pipeline=pipeline, fill_values=cfg.get("fill_values"))

    coords = {"lake": np.asarray(data.ids, dtype=str), "doy": data.doy}
    ds = xr.Dataset(coords=coords)

    for k, name in enumerate(DERIVED):
        ds[name] = (("lake", "doy"), features[:, :, k])
    for k, name in enumerate(CARRA_VARS):
        ds[name] = (("lake", "doy"), carra[:, :, k])
    for name, (units, long_name) in UNITS.items():
        ds[name].attrs.update(units=units, long_name=long_name)

    # Per-lake fields: the event table plus the metadata every stage needs.
    ev = events.reindex(ds.lake.values)
    meta = data.meta.reindex(ds.lake.values)
    ds["event_doy"] = ("lake", ev.event_doy.to_numpy(dtype="float32"))
    ds["event_doy"].attrs.update(
        units="1",
        long_name="day of year on which the lake starts losing water",
        note="ground truth, deliberately retrospective; a lead-time coordinate, never a feature",
        missing_meaning="NaN: undatable -- too few optical observations to establish a seasonal peak",
    )
    ds["event_source"] = ("lake", ev.event_source.fillna(NOT_APPLICABLE).to_numpy(dtype=str))
    ds["event_source"].attrs["long_name"] = (
        "'published' if upstream supplied the date, 'derived' if we computed it"
    )
    ds["radar_recovers"] = ("lake", ev.radar_recovers.to_numpy(dtype="int8"))
    ds["machine_label"] = ("lake", meta.machine_label.fillna(NOT_APPLICABLE).to_numpy(dtype=str))
    ds["expert_label"] = ("lake", meta.expert_label.fillna(NOT_APPLICABLE).to_numpy(dtype=str))
    ds["expert_label"].attrs["missing_meaning"] = (
        f"{NOT_APPLICABLE}: lake is not in the 1000-lake expert set (2019 only)"
    )
    ds["region"] = ("lake", meta.region.fillna(NOT_APPLICABLE).to_numpy(dtype=str))
    ds["fold"] = ("lake", meta.fold.fillna(-1).to_numpy(dtype="int8"))
    ds["fold"].attrs["missing_meaning"] = "-1: not applicable, lake is not in the expert set"
    ds["area"] = ("lake", meta["area"].to_numpy(dtype="float32"))
    ds["elevation"] = ("lake", meta.elevation.to_numpy(dtype="float32"))
    ds["drain_date"] = ("lake", meta.drain_date.to_numpy(dtype="float32"))
    # A numeric variable cannot hold the string "N/A", so the meaning of NaN is
    # recorded as an attribute instead. It is not missing data: upstream defines
    # a drain date only for rapid drainers, so for the other three classes the
    # quantity does not exist. Use event_doy, which is defined for all classes.
    ds["drain_date"].attrs.update(
        long_name="upstream drain date (Dunmire et al. 2025), rapid drainers only",
        missing_meaning="NaN: not applicable -- upstream defines this only for the 'drain' class",
        see_also="event_doy",
    )

    ds.attrs.update(
        title=f"sgl-leadtime processed features, {data.year}, {pipeline} pipeline",
        pipeline=pipeline,
        year=data.year,
        sgl_version=__version__,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        smoothing_window=WINDOW,
        carra_earth_radius_m=EARTH_RADIUS_M,
        fill_values=str(cfg.get("fill_values")),
        carra_fill_values=str(cfg.get("carra_fill_values")),
        causality=(
            "causal: every value at day t uses days <= t only. "
            "published: verbatim port of the upstream pipeline, which does not."
        ),
        source="Dunmire et al. (2025) deposit, doi:10.1029/2024EA003793; CARRA reanalysis (CDS)",
    )
    return ds


def path_for(cfg, year: int, pipeline: str):
    return cfg.processed / f"processed_{year}_{pipeline}.nc"


def save(cfg, ds: xr.Dataset, year: int, pipeline: str):
    cfg.processed.mkdir(parents=True, exist_ok=True)
    path = path_for(cfg, year, pipeline)
    encoding = {
        v: {"zlib": True, "complevel": 4}
        for v in ds.data_vars
        if np.issubdtype(ds[v].dtype, np.number)
    }
    ds.to_netcdf(path, encoding=encoding)
    return path


def load(cfg, year: int, pipeline: str = "causal") -> xr.Dataset:
    return xr.open_dataset(path_for(cfg, year, pipeline))


def to_tensor(ds: xr.Dataset, variables: list[str] | None = None) -> np.ndarray:
    """``(n_lakes, n_days, n_vars)`` in the requested variable order."""
    variables = variables or ALL_VARS
    return (
        ds[variables]
        .to_array(dim="variable")
        .transpose("lake", "doy", "variable")
        .to_numpy()
        .astype(np.float32)
    )
