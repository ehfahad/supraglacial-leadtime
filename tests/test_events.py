"""Event dating must reproduce upstream exactly where upstream supplies a date.

Lead time is measured against these dates, so if ours drifted from the
published ones the truncation comparison would run on a different coordinate
system than the model it audits.
"""

import numpy as np
import pytest

from sgl.config import load
from sgl.data import load_year
from sgl.preprocess import UPSTREAM_EVENT_WINDOW, event_doy


@pytest.fixture(scope="module")
def cfg():
    return load()


@pytest.mark.parametrize("year", [2018, 2019])
def test_rederivation_matches_every_supplied_drain_date(cfg, year):
    """Derive from scratch -- ignoring the supplied value -- and compare."""
    data = load_year(year, cfg)
    supplied = data.meta.drain_date.dropna()
    assert len(supplied) > 700, f"expected many supplied dates in {year}, got {len(supplied)}"

    mismatches = []
    for lake_id, want in supplied.items():
        got = event_doy(data.series(lake_id), supplied=None, window=UPSTREAM_EVENT_WINDOW)
        if not (np.isfinite(got) and got == want):
            mismatches.append((lake_id, want, got))
    assert not mismatches, f"{len(mismatches)}/{len(supplied)} disagree, e.g. {mismatches[:5]}"


def test_reference_lake_drains_on_doy_184(cfg):
    data = load_year(2019, cfg)
    assert event_doy(data.series("CW2019_1541")) == 184.0
