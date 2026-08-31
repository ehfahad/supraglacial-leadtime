"""The paper's central claim, as an executable test.

A feature computed for day *t* may depend only on days <= *t*. We check this the
only way that actually proves it: take a lake, destroy every observation after
day T, rebuild the features, and require that nothing at or before T moved.

Two assertions matter equally.

* ``causal`` must **pass**: our pipeline never reads the future.
* ``published`` must **fail**: the upstream pipeline does.

The second is not decoration. A test that only confirmed the good case would
still pass if a refactor silently pointed both pipelines at the same code, and
the leakage ablation would quietly become a comparison of nothing.

It needs only ``data/raw`` and is the one command a reader can run to check
the central result:  ``pytest tests/test_causality.py``
"""

from __future__ import annotations

import numpy as np
import pytest

from sgl.config import load
from sgl.data import load_year
from sgl.preprocess import DERIVED, build_features

# Cutoffs spanning the melt season: before, during and after the drainage of the
# reference lake (DOY 184), so the test covers the case where the future being
# hidden actually contains the event.
CUTOFFS = [152, 166, 180, 194, 208]

REFERENCE_LAKE = "CW2019_1541"

# Same variable, same 12-day window; the two pipelines differ only in fill
# method and window alignment, which is precisely the leak.
KNOWN = {
    ("causal", "HV_anom", 180): -9.12,     # LOCF + trailing mean
    ("published", "HV_anom", 180): -8.03,  # linear interpolation + centred mean
    ("causal", "pwater", 180): 0.947,
}


@pytest.fixture(scope="module")
def cfg():
    return load()


@pytest.fixture(scope="module")
def year2019(cfg):
    return load_year(2019, cfg)


@pytest.fixture(scope="module")
def sample_lakes(year2019):
    """Twenty lakes, five per expert class, so no result rests on one lake."""
    expert = year2019.expert
    rng = np.random.default_rng(0)
    ids = []
    for label in sorted(expert.expert_label.unique()):
        pool = expert.index[expert.expert_label == label].to_numpy()
        ids.extend(rng.choice(pool, size=5, replace=False))
    return ids


def scramble_after(series, cutoff, seed=0):
    """Destroy every observation after ``cutoff``, preserving nothing about it.

    Rows are permuted rather than blanked: blanking could be survived by a
    pipeline that merely propagates NaN, whereas permutation changes the values
    a leaky pipeline would actually read.
    """
    out = series.copy()
    future = out.index > cutoff
    block = out.loc[future].to_numpy()
    rng = np.random.default_rng(seed)
    out.loc[future] = block[rng.permutation(len(block))]
    return out


def _past_is_stable(raw, cutoff, pipeline):
    """True if rebuilding after scrambling leaves days <= cutoff untouched."""
    base = build_features(raw, pipeline=pipeline)
    rebuilt = build_features(scramble_after(raw, cutoff), pipeline=pipeline)
    past = base.index <= cutoff
    return np.allclose(
        base.loc[past, DERIVED].to_numpy(dtype=float),
        rebuilt.loc[past, DERIVED].to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_causal_pipeline_never_reads_the_future(year2019, sample_lakes, cutoff):
    for lake_id in sample_lakes:
        raw = year2019.series(lake_id)
        assert _past_is_stable(raw, cutoff, "causal"), (
            f"causal pipeline leaked: lake {lake_id} changed at or before "
            f"DOY {cutoff} when only later days were altered"
        )


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_published_pipeline_does_read_the_future(year2019, sample_lakes, cutoff):
    """The leakage ablation is only meaningful if this leak is real."""
    leaked = [
        lake_id
        for lake_id in sample_lakes
        if not _past_is_stable(year2019.series(lake_id), cutoff, "published")
    ]
    assert leaked, (
        f"published pipeline showed no leakage at DOY {cutoff} across "
        f"{len(sample_lakes)} lakes; the two pipelines are probably identical, "
        "which would make the leakage ablation vacuous"
    )


def test_reference_lake_reproduces_known_values(year2019):
    """Guards against a pipeline that is causal but silently wrong."""
    raw = year2019.series(REFERENCE_LAKE)
    for (pipeline, column, doy), expected in KNOWN.items():
        got = build_features(raw, pipeline=pipeline).loc[doy, column]
        assert got == pytest.approx(expected, abs=5e-3), (
            f"{pipeline} {column} at DOY {doy}: got {got:.4f}, want {expected}"
        )


def test_leak_is_visible_at_the_reference_lake(year2019):
    """The concrete instance behind the abstract claim.

    CW2019_1541 drains on DOY 184. A centred 12-day window at DOY 180 reaches
    DOY 186, where radar backscatter has already recovered, so the published
    feature at DOY 180 encodes a drainage that has not happened yet.
    """
    raw = year2019.series(REFERENCE_LAKE)
    causal = build_features(raw, pipeline="causal").loc[180, "HV_anom"]
    published = build_features(raw, pipeline="published").loc[180, "HV_anom"]
    assert published > causal + 1.0, (
        "expected the published feature to be pulled toward the post-drainage "
        f"value, but got causal {causal:.3f} vs published {published:.3f}"
    )
