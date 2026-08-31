"""Loading and joining the lake dataset.

Three sources, joined on ``new_id``:

* ``data/raw/all_lakes_{year}.nc``           daily satellite series, 365 days
* ``data/raw/labels_{year}_volumes.geojson``  machine labels, basin, drain date
* ``data/raw/all_training.geojson``          1000 expert labels and CV folds

Nothing here derives features; that is ``preprocess.py``. This module's only job
is to produce an aligned (series, metadata) pair that every downstream stage
shares, and to assert the join is what we think it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from .config import Config, load

# Raw numeric series present in the netCDFs. ``S1_image_id`` and ``S1_code`` are
# string provenance fields, carried in the file but not features.
RAW_VARS = [
    "HV_lake",
    "HV_out",
    "S2_water",
    "S2_zenith",
    "LS_water",
    "LS_zenith",
    "t2m",
]

# Expected join sizes. These are the loader's contract: a mismatch means the
# code is wrong, not the data.
N_LAKES = {2018: 3846, 2019: 6146}
N_EXPERT = 1000
N_AGREE = 994  # expert vs. machine label agreement on the expert subset


@dataclass
class LakeData:
    """Series and metadata for one melt season."""

    year: int
    ds: xr.Dataset
    meta: pd.DataFrame

    @cached_property
    def ids(self) -> np.ndarray:
        """Lake ids, in the order used by every array this class returns."""
        return self.ds.ids.values

    @cached_property
    def doy(self) -> np.ndarray:
        """Day of year, 1..365, aligned to the time axis."""
        return self.ds.time.dt.dayofyear.values

    def array(self, variables: list[str] | None = None) -> np.ndarray:
        """Raw series as ``(n_lakes, 365, n_vars)``, in ``variables`` order."""
        variables = variables or RAW_VARS
        return np.stack([self.ds[v].values for v in variables], axis=-1)

    def series(self, lake_id: str) -> pd.DataFrame:
        """One lake's raw series as a 365-row frame indexed by day of year."""
        sel = self.ds.sel(ids=lake_id)
        return pd.DataFrame(
            {v: sel[v].values for v in RAW_VARS},
            index=pd.Index(self.doy, name="doy"),
        )

    @property
    def expert(self) -> pd.DataFrame:
        """The labelled subset, in fold order. Empty for 2018."""
        return self.meta[self.meta.expert_label.notna()]

    def fold_mask(self, fold: int) -> np.ndarray:
        """Boolean mask over ``self.ids`` selecting one CV fold."""
        return (self.meta.fold == fold).to_numpy()


def _load_meta(cfg: Config, year: int, ids: np.ndarray) -> pd.DataFrame:
    """Machine labels joined to expert labels, reindexed onto the netCDF order."""
    labels = gpd.read_file(cfg.raw / f"labels_{year}_volumes.geojson")
    meta = pd.DataFrame(
        {
            "new_id": labels.new_id,
            "region": labels["region"],
            # Bracket access is required: GeoDataFrame.area is the geometric
            # area of the polygon, which silently shadows the column.
            "area": labels["area"],
            "elevation": labels["elevation"],
            "machine_code": labels["final_label"],
            "drain_date": labels["drain_date"],
        }
    ).set_index("new_id")
    meta["machine_label"] = meta.machine_code.map(cfg.label_map)

    # Expert labels exist for 2019 only, so 2018 simply gets empty columns.
    expert = gpd.read_file(cfg.raw / "all_training.geojson")
    expert = expert[expert.year == year]
    meta["expert_label"] = pd.Series(expert.label.values, index=expert.new_id)
    meta["fold"] = pd.Series(expert.fold.values, index=expert.new_id).astype("Int64")

    # Reindex onto the netCDF id order so row i of any array is meta row i.
    meta = meta.reindex(ids)
    meta["year"] = year
    return meta


def load_year(year: int, cfg: Config | None = None) -> LakeData:
    cfg = cfg or load()
    ds = xr.open_dataset(cfg.raw / f"all_lakes_{year}.nc")
    return LakeData(year=year, ds=ds, meta=_load_meta(cfg, year, ds.ids.values))


def validate(data: dict[int, LakeData]) -> None:
    """Assert the joins match what was measured by hand.

    Raises AssertionError with the observed value, so a failure says what broke
    rather than merely that something did.
    """
    for year, d in data.items():
        assert len(d.ids) == N_LAKES[year], f"{year}: {len(d.ids)} lakes, want {N_LAKES[year]}"
        assert d.meta.index.equals(pd.Index(d.ids)), f"{year}: meta not aligned to netCDF order"
        assert d.meta.machine_label.notna().all(), f"{year}: {d.meta.machine_label.isna().sum()} unlabelled"
        assert len(d.doy) == 365 and d.doy[0] == 1 and d.doy[-1] == 365, f"{year}: bad time axis"

    e = data[2019].expert
    assert len(e) == N_EXPERT, f"expert set is {len(e)}, want {N_EXPERT}"
    counts = e.expert_label.value_counts()
    assert set(counts) == {250}, f"expert classes unbalanced: {counts.to_dict()}"
    assert sorted(e.fold.unique()) == [0, 1, 2, 3, 4], f"folds are {sorted(e.fold.unique())}"

    agree = int((e.expert_label == e.machine_label).sum())
    assert agree == N_AGREE, f"expert/machine agreement is {agree}, want {N_AGREE}"

    # drain_date is supplied for the rapid-drain class only; the other three
    # classes must be derived in preprocess.py. Assert that, so a future change
    # upstream cannot silently remove the need for the derivation.
    supplied = e.groupby("expert_label").drain_date.apply(lambda s: s.notna().sum())
    assert supplied["drain"] == 250, f"drain dates supplied: {supplied['drain']}"
    assert supplied.drop("drain").sum() == 0, f"unexpected drain dates: {supplied.to_dict()}"


def load_all(cfg: Config | None = None) -> dict[int, LakeData]:
    cfg = cfg or load()
    data = {y: load_year(y, cfg) for y in cfg["years"]}
    validate(data)
    return data
