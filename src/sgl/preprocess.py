"""Feature construction, in two pipelines.

``causal``
    Every value at day *t* uses days <= *t* only. This is what the paper's
    lead-time claim requires: if a feature at DOY 180 can see DOY 186, then
    "accuracy at DOY 180" is not a statement about what was knowable on DOY 180.

``published``
    A faithful port of the upstream code in
    ``data/external/zenodo/python_function_files/data_prep_fns.py``, bugs and
    all. It is not a strawman and must not be "improved": the gap between the
    two pipelines is the paper's leakage measurement, so any deviation here is
    counted as leakage that upstream did not actually commit. The paper calls
    this chain "conventional".

Where they differ, concretely, for the radar anomaly:

    causal     linear-fill forbidden -> LOCF        -> trailing 12-day mean
    published  interpolate_na linear -> centred 12-day mean

Nothing else about the radar chain differs -- same variable, same window width.
On lake CW2019_1541 at DOY 180 that produces -9.12 against -8.03, because the
centred window reaches DOY 186, where backscatter has already recovered from a
drainage that occurs on DOY 184.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# The nine satellite features every downstream stage consumes, in tensor order.
DERIVED = [
    "HV_lake",
    "HV_out",
    "HV_anom",
    "S2_water",
    "LS_water",
    "pwater",
    "t2m",
    "S2_zenith",
    "LS_zenith",
]

WINDOW = 12          # smoothing width, shared by both pipelines
SHADOW_ZENITH = 75   # degrees; upstream threshold
SHADOW_WATER = 0.05
HAMPEL_SIGMA = 3.0
MAD_TO_SIGMA = 1.4826

MELT_START_DOY = 121  # 1 May
MELT_END_DOY = 288    # 15 Oct


# --------------------------------------------------------------------------
# shared, and causal by construction
# --------------------------------------------------------------------------


def mask_shadows(water: pd.Series, zenith: pd.Series) -> pd.Series:
    """Blank optical water where a low sun makes shadow look like water.

    Pointwise, so it is causal in both pipelines and identical in both.
    """
    bad = (zenith > SHADOW_ZENITH) & (water > SHADOW_WATER)
    return water.mask(bad)


def combine_optical(s2: pd.Series, ls: pd.Series) -> pd.Series:
    """Mean of whichever optical sensors saw the lake that day."""
    stacked = np.vstack([ls.to_numpy(dtype=float), s2.to_numpy(dtype=float)])
    # Days when neither sensor saw the lake are all-NaN by design, so numpy's
    # "mean of empty slice" warning is expected. Suppressed narrowly here rather
    # than globally, so a genuine warning elsewhere still surfaces.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        combined = np.nanmean(stacked, axis=0)
    return pd.Series(combined, index=s2.index)


# --------------------------------------------------------------------------
# causal pipeline
# --------------------------------------------------------------------------


def hampel_causal(x: pd.Series, window: int = WINDOW, sigma: float = HAMPEL_SIGMA) -> pd.Series:
    """One-sided Hampel filter: flag day *t* using days [t-window, t] only.

    The upstream outlier rules (``get_low_outliers``, ``get_high_outliers``)
    inspect ``water[i+1]`` and take season-wide maxima, so neither can be used
    here. This is the causal replacement: a point is dropped when it sits more
    than ``sigma`` robust standard deviations from the median of its own past.

    A trailing window cannot judge the first few observations of the season, so
    they are kept rather than dropped -- being permissive early is the safe
    error, since dropping them would silently shorten every early horizon.
    """
    med = x.rolling(window, min_periods=3).median()
    mad = (x - med).abs().rolling(window, min_periods=3).median() * MAD_TO_SIGMA
    deviant = (x - med).abs() > sigma * mad
    return x.mask(deviant.fillna(False))


def _causal_fill_smooth(x: pd.Series, window: int = WINDOW) -> pd.Series:
    """LOCF, then a trailing mean.

    LOCF is the only strictly causal fill that needs no model of the signal.
    It does produce staircases, which can manufacture autocorrelation at the
    satellite revisit cadence (~6 d radar, ~3.2 d optical), so any apparent
    periodicity at those lags should be read as a fill artifact, not signal.
    """
    return x.ffill().rolling(window, min_periods=1).mean()


def _build_causal(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    s2 = mask_shadows(raw.S2_water, raw.S2_zenith)
    ls = mask_shadows(raw.LS_water, raw.LS_zenith)
    water = hampel_causal(combine_optical(s2, ls))

    # Which variables get smoothed is dictated by the published pipeline, not by
    # taste: upstream applies rolling(12) to the radar chain only, and merely
    # interpolates everything else. Mirroring that exactly is what makes the two
    # pipelines differ in causality alone -- smooth something here that upstream
    # leaves raw, and the ablation would be measuring our filter, not their leak.
    out["HV_lake"] = _causal_fill_smooth(raw.HV_lake)
    out["HV_out"] = _causal_fill_smooth(raw.HV_out)
    out["HV_anom"] = _causal_fill_smooth(raw.HV_lake - raw.HV_out)
    out["S2_water"] = s2.ffill()
    out["LS_water"] = ls.ffill()
    out["pwater"] = water.ffill()
    out["t2m"] = raw.t2m.ffill()
    out["S2_zenith"] = raw.S2_zenith.ffill()
    out["LS_zenith"] = raw.LS_zenith.ffill()
    return out[DERIVED]


# --------------------------------------------------------------------------
# published pipeline -- ported verbatim, deliberately not fixed
# --------------------------------------------------------------------------


def _get_water_periods(water: np.ndarray):
    """Port of ``get_water_periods``."""
    lake = False
    lakes, lengths, temp = [], [], []
    for i in range(len(water) - 1):
        if water[i] > 0.1 and not lake:
            temp = [i]
            lake = True
        elif water[i] > 0.1:
            temp.append(i)
        if lake and water[i] < 0.1 and water[i + 1] < 0.1:
            lake = False
            lakes.append(temp)
            lengths.append(len(temp))
    if lake:
        lakes.append(temp)
        lengths.append(len(temp))
    return lakes, lengths


def _get_low_outliers(combined: pd.Series) -> pd.Series:
    """Port of ``get_low_outliers``. Reads ``water[i+1]`` -- non-causal."""
    valid = combined.dropna()
    w = valid.to_numpy(dtype=float)
    drop = [
        valid.index[i]
        for i in range(1, len(w) - 1)
        if w[i - 1] - w[i] > 0.2 and w[i + 1] - w[i] > 0.2
    ]
    return combined.mask(combined.index.isin(drop))


def _get_high_outliers(no_low: pd.Series) -> pd.Series:
    """Port of ``get_high_outliers``.

    Non-causal twice over: it reads ``water[i+1]``, and it compares every period
    against ``max(lake_lengths)`` computed over the entire season.
    """
    valid = no_low.dropna()
    w = valid.to_numpy(dtype=float)
    periods, lengths = _get_water_periods(w)

    outliers: list[int] = []
    actual: list[int] = []
    if lengths:
        max_length = max(lengths)
        for period, length in zip(periods, lengths):
            if len(period) > 1 and np.max(w[period]) > 0.4:
                actual.extend(period)
            elif length > 0.2 * max_length and (length > 3 or np.max(w[period]) > 0.8):
                actual.extend(period)

        actual_set = set(actual)
        outliers.extend(i for i in range(len(w)) if i not in actual_set and w[i] > 0.1)

        for i in range(1, len(w) - 1):
            had_water = bool(np.any(w[:i] > 0.8))
            if w[i] > 0.4 and w[i - 1] < w[i] * 0.5 and w[i + 1] < w[i] * 0.5 and had_water:
                outliers.append(i)

    out = no_low.mask(no_low.index.isin(valid.index[list(set(outliers))]))

    # Upstream pins two calendar days to zero by positional index. Indices 121
    # and 288 are 0-based, so they land on DOY 122 and 289 rather than the 1 May
    # and 15 Oct named in the comments. Ported as written.
    arr = out.to_numpy(dtype=float)
    arr[121] = 0.0
    arr[288] = 0.0
    return pd.Series(arr, index=out.index)


def _build_published(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    # prep_S1_data: linear interpolation, then a *centred* rolling mean.
    hv = raw[["HV_lake", "HV_out"]].interpolate(method="linear", limit_direction="both")
    hv_s = hv.rolling(WINDOW, min_periods=1, center=True).mean()

    # prep_S2_data: shadow fix -> combine -> two look-ahead outlier passes ->
    # linear interpolation -> zero-fill inside the melt window.
    s2 = mask_shadows(raw.S2_water, raw.S2_zenith)
    ls = mask_shadows(raw.LS_water, raw.LS_zenith)
    water = _get_high_outliers(_get_low_outliers(combine_optical(s2, ls)))
    water = water.interpolate(method="linear", limit_direction="both")
    in_melt = (water.index >= MELT_START_DOY) & (water.index <= MELT_END_DOY)
    water = water.where(~in_melt | water.notna(), 0.0)

    out["HV_lake"] = hv_s.HV_lake
    out["HV_out"] = hv_s.HV_out
    out["HV_anom"] = hv_s.HV_lake - hv_s.HV_out
    out["S2_water"] = s2.interpolate(method="linear", limit_direction="both")
    out["LS_water"] = ls.interpolate(method="linear", limit_direction="both")
    out["pwater"] = water
    out["t2m"] = raw.t2m.interpolate(method="linear", limit_direction="both")
    out["S2_zenith"] = raw.S2_zenith.interpolate(method="linear", limit_direction="both")
    out["LS_zenith"] = raw.LS_zenith.interpolate(method="linear", limit_direction="both")
    return out[DERIVED]


# --------------------------------------------------------------------------


def fill_leading(features: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    """Replace the pre-first-observation stretch with data-independent constants.

    Every remaining NaN after LOCF is strictly leading -- verified across both
    seasons, no interior gaps -- because forward fill covers everything once a
    lake has been seen once. Those leading days cannot be filled causally from
    the data: back-filling reads the future, and a training-set mean or a
    climatology still summarises days after *t*.

    So they are filled with fixed constants declared in ``config.yaml``, which
    depend on no observation whatsoever and are therefore causal by definition.
    The same constants are applied to both pipelines, so the causal-vs-published
    comparison measures leakage and not this choice.

    It matters mostly in early May: at DOY 121 up to 80% of lakes have no
    Landsat observation yet, falling to ~2.6% by DOY 152, the first horizon.
    """
    out = features.copy()
    for column, value in fill_values.items():
        if column in out.columns:
            out[column] = out[column].fillna(value)
    return out


def build_features(
    raw: pd.DataFrame, pipeline: str = "causal", fill_values: dict | None = None
) -> pd.DataFrame:
    """Derive the nine satellite features for one lake.

    ``raw`` is a 365-row frame indexed by day of year, as returned by
    ``LakeData.series``. Passing ``fill_values`` additionally replaces the
    leading, unobservable stretch with constants; leave it None to inspect where
    the data genuinely starts.
    """
    if pipeline == "causal":
        out = _build_causal(raw)
    elif pipeline == "published":
        out = _build_published(raw)
    else:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected 'causal' or 'published'")
    return fill_leading(out, fill_values) if fill_values else out


# --------------------------------------------------------------------------
# tensor construction
# --------------------------------------------------------------------------


def build_tensor(data, pipeline: str = "causal", fill_values: dict | None = None,
                 progress_every: int = 1000) -> np.ndarray:
    """Derive features for every lake in a season.

    Returns ``(n_lakes, 365, len(DERIVED))`` as float32, with row *i* aligned to
    ``data.ids[i]`` and ``data.meta`` row *i*.

    Because every causal feature obeys "day t uses only days <= t", horizon T is
    afterwards just ``tensor[:, :T]``. That is the whole reason this runs once
    rather than once per horizon: ten horizons for the price of one pass.
    """
    ids = data.ids
    out = np.full((len(ids), 365, len(DERIVED)), np.nan, dtype=np.float32)
    for i, lake_id in enumerate(ids):
        out[i] = build_features(
            data.series(lake_id), pipeline=pipeline, fill_values=fill_values
        ).to_numpy(dtype=np.float32)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {pipeline} {data.year}: {i + 1}/{len(ids)}", flush=True)
    return out


# --------------------------------------------------------------------------
# event dates
# --------------------------------------------------------------------------

# Upstream dates events inside 1 May - 1 Sep. That window truncates the late
# classes: dated refreeze and burial events pile up against it (90th percentile
# 230 and 238, max 243), and 406 of 6146 lakes in 2019 get no date at all.
# Extending to 15 Oct rescues 81% of them.
UPSTREAM_EVENT_WINDOW = (MELT_START_DOY, 244)
EVENT_WINDOW = (MELT_START_DOY, MELT_END_DOY)  # 1 May - 15 Oct


def water_final(raw: pd.DataFrame) -> pd.Series:
    """The upstream cleaned water series, used for event dating only."""
    s2 = mask_shadows(raw.S2_water, raw.S2_zenith)
    ls = mask_shadows(raw.LS_water, raw.LS_zenith)
    return _get_high_outliers(_get_low_outliers(combine_optical(s2, ls)))


def event_doy(raw: pd.DataFrame, supplied: float | None = None, window=None) -> float:
    """Day of year on which the lake starts losing water.

    Ported from upstream ``get_drain_start_end``: find the seasonal maximum,
    then walk forward to the first day that has fallen below 90% of it (and
    below 0.8 absolute); the event is dated to the observation *before* that.

    Retrospective by construction -- it uses the season maximum -- and that is
    correct. An event date is ground truth about what happened, not a feature.
    It is the coordinate lead time is measured against, never an input to a
    model, so it carries no leakage risk. Features remain strictly causal.

    Applied identically to all four classes, so refreeze and burial get dates on
    the same footing as drainage.

    When upstream supplies a ``drain_date`` we take it verbatim rather than
    re-deriving. Our own re-derivation reproduces all 2127 supplied dates
    exactly under the upstream window, so this changes no value -- but it makes
    the longer window used for the late classes incapable of perturbing a
    published date, which keeps the comparison anchored to upstream's own
    event coordinates.
    """
    if supplied is not None and np.isfinite(supplied):
        return float(supplied)

    lo, hi = window or EVENT_WINDOW
    water = water_final(raw).loc[lo:hi].dropna()
    if len(water) < 2:
        return np.nan

    values = water.to_numpy(dtype=float)
    days = water.index.to_numpy()
    peak = values.max()
    if not np.isfinite(peak) or peak <= 0:
        return np.nan

    start = int(np.argmax(values == peak))
    for i in range(start, len(values)):
        if values[i] < peak * 0.90 and values[i] <= 0.8:
            return float(days[i - 1]) if i > 0 else np.nan
    return np.nan


def radar_recovers(raw: pd.DataFrame, event: float, window: int = 30) -> bool:
    """Whether backscatter climbs back after the water goes.

    Distinguishes refreeze (water leaves, surface returns to a bright frozen
    signature) from burial (water leaves, surface does not recover). Used to
    label the mechanism, not to date it.
    """
    if not np.isfinite(event):
        return False
    anom = (raw.HV_lake - raw.HV_out).dropna()
    before = anom.loc[: int(event)]
    after = anom.loc[int(event) : int(event) + window]
    if before.empty or after.empty:
        return False
    return bool(after.max() > before.min() + 3.0)


def build_events(data) -> pd.DataFrame:
    """Event date and radar-recovery flag for every lake in a season."""
    records = []
    supplied_dates = data.meta.drain_date
    for lake_id in data.ids:
        raw = data.series(lake_id)
        doy = event_doy(raw, supplied=supplied_dates.get(lake_id))
        records.append(
            {
                "new_id": lake_id,
                "event_doy": doy,
                "event_source": "published" if np.isfinite(supplied_dates.get(lake_id, np.nan)) else "derived",
                "radar_recovers": radar_recovers(raw, doy),
            }
        )
    return pd.DataFrame(records).set_index("new_id")
