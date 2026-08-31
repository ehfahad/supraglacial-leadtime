# Data

Nothing in this directory is committed. Both sources are public and are fetched
by DOI; neither is re-hosted here.

## 1. Supraglacial lake record and expert labels

Dunmire et al., <https://doi.org/10.5281/zenodo.14587026> (CC-BY 4.0).

From `INPUT.zip` and `OUTPUT.zip`, place these in `data/raw/`:

| File | Size | Used for |
|---|---|---|
| `all_lakes_2019.nc` | 294 MB | daily satellite series, 2019 |
| `all_lakes_2018.nc` | 184 MB | daily satellite series, 2018 |
| `all_training.geojson` | 1.7 MB | the 1,000 expert labels and the 5 folds |
| `labels_2019_volumes.geojson` | 12 MB | machine labels, basin, drainage date |
| `labels_2018_volumes.geojson` | 7.3 MB | machine labels, basin, drainage date |

The released full-season models (`OUTPUT/models/*.pkl`) are needed only for the
truncation comparison in `scripts/run_c5.py`; put the unpacked deposit in
`data/external/zenodo/`.

The tests need only `all_lakes_2019.nc`, `all_training.geojson` and
`labels_2019_volumes.geojson`, about 307 MB.

## 2. CARRA reanalysis

Copernicus Arctic Regional Reanalysis, single levels,
<https://doi.org/10.24381/cds.713858f6> (CC-BY 4.0), from the Climate Data
Store.

Five surface variables over 2018-2019, Greenland domain, placed in `data/raw/`
as GRIB2:

| File | Variables |
|---|---|
| `carra_runoff_2018_2019.grib` | surface meltwater runoff |
| `carra_albedo_2018_2019.grib` | broadband albedo (12 and 15 UTC) |
| `carra_analysis_2018_2019.grib` | 2 m relative humidity, snow water equivalent, surface pressure |

These total 6.2 GB and are needed only to rebuild the processed tensors. The
CARRA channels are already inside the published tensors, so most users can skip
this download entirely.

Copernicus attribution, required when redistributing anything derived from
these files: *Contains modified Copernicus Climate Change Service information
2026. Neither the European Commission nor ECMWF is responsible for any use of
the Copernicus information or data it contains.*

## 3. Processed tensors

Either build them with `make preprocess`, which needs both sources above, or
download `processed.zip` from <https://doi.org/10.5281/zenodo.22085303> and
unpack it into `data/processed/`. The two are equivalent; the download is
122 MB against a 6.5 GB build.
