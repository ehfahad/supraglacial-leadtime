"""CARRA extraction: GRIB2 decoding and per-lake sampling, in pure numpy.

No GRIB library is installed and none is needed. The three CARRA files use data
representation template 5.0 (simple packing), which unpacks with arithmetic:

    value = (R + X * 2**E) / 10**D

where R, E, D come from the data representation section and X is the packed
integer field.

The grid is read from the file rather than assumed. In particular the shape-of-
earth octet is checked before any coordinate is computed: a wrong earth radius
shifts every lake by roughly a grid cell, which would silently misattribute the
melt forcing of one catchment to its neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

# GRIB2 code table 3.2 entry 6: spherical earth, radius fixed by the standard.
SHAPE_SPHERICAL_6371229 = 6
EARTH_RADIUS_M = 6371229.0

# Files carry (discipline, category, number) triples rather than names.
PARAMETERS = {
    (2, 0, 34): "runoff",     # surface runoff, kg m-2, 24 h accumulation
    (0, 19, 1): "albedo",     # forecast albedo, PERCENT -- see ALBEDO_SCALE
    (0, 1, 1): "rh2m",        # 2 m relative humidity, %
    (0, 1, 60): "sde_swe",    # snow depth WATER EQUIVALENT, kg m-2 (not metres)
    (0, 3, 0): "sp",          # surface pressure, Pa
}

# CARRA reports albedo in percent (observed range 0-87), so it needs dividing.
ALBEDO_SCALE = {"albedo": 0.01}


def _u(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _signed(value: int, bits: int = 32) -> int:
    """GRIB uses sign-and-magnitude, not two's complement."""
    high = 1 << (bits - 1)
    return -(value & (high - 1)) if value & high else value


def _ieee(b: bytes) -> float:
    return float(np.frombuffer(b, dtype=">f4")[0])


@dataclass(frozen=True)
class Grid:
    """Lambert conformal grid, read from section 3."""

    nx: int
    ny: int
    la1: float
    lo1: float
    lad: float
    lov: float
    dx: float
    dy: float
    latin1: float
    latin2: float
    scan_mode: int

    @property
    def proj4(self) -> str:
        return (
            f"+proj=lcc +lat_1={self.latin1} +lat_2={self.latin2} "
            f"+lat_0={self.lad} +lon_0={self.lov} +x_0=0 +y_0=0 "
            f"+R={EARTH_RADIUS_M} +units=m +no_defs"
        )


def parse_gds(section3: bytes) -> Grid:
    """Read the grid definition, refusing anything whose earth we cannot trust."""
    g = lambda a, b: section3[a - 1 : b]  # noqa: E731 -- 1-based GRIB octets

    template = _u(g(13, 14))
    if template != 30:
        raise ValueError(f"expected Lambert conformal (3.30), got template 3.{template}")

    shape = g(15, 15)[0]
    if shape != SHAPE_SPHERICAL_6371229:
        raise ValueError(
            f"shape-of-earth octet is {shape}, not {SHAPE_SPHERICAL_6371229}. "
            "The radius is no longer the one this module assumes; re-read the "
            "scaled radius before trusting any coordinate."
        )

    return Grid(
        nx=_u(g(31, 34)),
        ny=_u(g(35, 38)),
        la1=_signed(_u(g(39, 42))) / 1e6,
        lo1=_signed(_u(g(43, 46))) / 1e6,
        lad=_signed(_u(g(48, 51))) / 1e6,
        lov=_signed(_u(g(52, 55))) / 1e6,
        dx=_u(g(56, 59)) / 1e3,
        dy=_u(g(60, 63)) / 1e3,
        latin1=_signed(_u(g(66, 69))) / 1e6,
        latin2=_signed(_u(g(70, 73))) / 1e6,
        scan_mode=g(65, 65)[0],
    )


def iter_messages(path: Path):
    """Yield ``(offset, length, sections)`` for every GRIB2 message in a file.

    Sections are read as raw bytes; section 7 (the payload) is skipped here and
    read on demand, so scanning a 2.9 GB file costs kilobytes.
    """
    size = path.stat().st_size
    with open(path, "rb") as fh:
        offset = 0
        while offset < size:
            fh.seek(offset)
            head = fh.read(16)
            if len(head) < 16 or head[:4] != b"GRIB":
                break
            total = _u(head[8:16])
            sections = {0: head}
            pos = offset + 16
            while pos < offset + total - 4:
                fh.seek(pos)
                hdr = fh.read(5)
                if hdr[:4] == b"7777":
                    break
                length, number = _u(hdr[:4]), hdr[4]
                if number == 7:  # payload: record where it is, do not read it
                    sections[7] = (pos, length)
                else:
                    fh.seek(pos)
                    sections[number] = fh.read(length)
                pos += length
            yield offset, total, sections
            offset += total


def message_key(sections: dict) -> tuple:
    """``(parameter, forecast_time, reference_date)`` for one message."""
    s1, s4 = sections[1], sections[4]
    discipline = sections[0][6]
    category, number = s4[9], s4[10]
    year, month, day = _u(s1[12:14]), s1[14], s1[15]
    hour, minute = s1[16], s1[17]
    lead = _u(s4[18:22])
    name = PARAMETERS.get((discipline, category, number))
    return name, (year, month, day, hour, minute), lead


def decode_values(path: Path, sections: dict, grid: Grid) -> np.ndarray:
    """Unpack one message's field as ``(ny, nx)`` with missing values as NaN."""
    s5 = sections[5]
    template = _u(s5[9:11])
    if template != 0:
        raise ValueError(f"expected simple packing (5.0), got 5.{template}")

    reference = _ieee(s5[11:15])
    binary_scale = _signed(_u(s5[15:17]), bits=16)
    decimal_scale = _signed(_u(s5[17:19]), bits=16)
    bits = s5[19]
    n_values = _u(s5[5:9])

    payload_offset, payload_length = sections[7]
    with open(path, "rb") as fh:
        fh.seek(payload_offset + 5)
        raw = fh.read(payload_length - 5)

    packed = _unpack_bits(raw, bits, n_values)
    values = (reference + packed * (2.0**binary_scale)) / (10.0**decimal_scale)

    # Section 6: a bitmap marks which grid points carry data.
    s6 = sections.get(6)
    if s6 is not None and s6[5] == 0:
        mask = np.unpackbits(np.frombuffer(s6[6:], dtype=np.uint8))[: grid.ny * grid.nx]
        field = np.full(grid.ny * grid.nx, np.nan)
        field[mask.astype(bool)] = values
    else:
        field = values.astype(float)

    field = field.reshape(grid.ny, grid.nx)
    # Scan mode bit 2 set means rows run south to north; flip to a north-up array.
    if grid.scan_mode & 0x40:
        field = field[::-1]
    return field


def _unpack_bits(raw: bytes, bits: int, count: int) -> np.ndarray:
    """Unpack ``count`` big-endian unsigned integers of ``bits`` width."""
    if bits == 0:
        return np.zeros(count)
    if bits % 8 == 0:
        nbytes = bits // 8
        arr = np.frombuffer(raw[: count * nbytes], dtype=np.uint8).reshape(count, nbytes)
        out = np.zeros(count, dtype=np.int64)
        for i in range(nbytes):
            out = (out << 8) | arr[:, i].astype(np.int64)
        return out.astype(float)
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[: count * bits]
    return unpacked.reshape(count, bits).dot(1 << np.arange(bits - 1, -1, -1)).astype(float)


def lake_grid_indices(lats: np.ndarray, lons: np.ndarray, grid: Grid) -> tuple:
    """Map lake centroids to (row, col) on the CARRA grid.

    The grid origin comes from the file's own first-point latitude/longitude, so
    no offset is assumed. Returns row and column arrays plus a validity mask.
    """
    from pyproj import CRS, Transformer

    crs = CRS.from_proj4(grid.proj4)
    to_grid = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

    x0, y0 = to_grid.transform(grid.lo1, grid.la1)
    x, y = to_grid.transform(lons, lats)

    col = np.rint((x - x0) / grid.dx).astype(int)
    row_from_south = np.rint((y - y0) / grid.dy).astype(int)
    row = (grid.ny - 1 - row_from_south) if grid.scan_mode & 0x40 else row_from_south

    valid = (col >= 0) & (col < grid.nx) & (row >= 0) & (row < grid.ny)
    return row, col, valid


# --------------------------------------------------------------------------
# per-lake extraction
# --------------------------------------------------------------------------

CARRA_VARS = ["runoff", "albedo", "rh2m", "sde_swe", "sp"]


def lake_centroids(cfg, year: int, ids):
    """Centroid latitude/longitude for each lake, in netCDF row order."""
    import geopandas as gpd
    import pandas as pd

    lb = gpd.read_file(cfg.raw / f"labels_{year}_volumes.geojson")
    cen = lb.geometry.centroid
    frame = pd.DataFrame({"lat": cen.y.values, "lon": cen.x.values}, index=lb["new_id"])
    frame = frame.reindex(ids)
    return frame.lat.to_numpy(), frame.lon.to_numpy()


def extract(cfg, lakes: dict, verbose: bool = True) -> dict:
    """Sample all five CARRA drivers at every lake centroid, for every day.

    ``lakes`` maps year -> LakeData. Returns year -> ``(n_lakes, 365, 5)``.

    Each file is streamed once. Fields are decoded, sampled at the lake cells,
    then discarded, so peak memory is one 1069x1269 field rather than the 6.2 GB
    of GRIB on disk. Albedo has two messages per day (12 and 15 UTC, bracketing
    solar noon across east and west Greenland) and is averaged.
    """
    index = {}
    grid = None
    out = {}
    counts = {}
    for year, data in lakes.items():
        out[year] = np.full((len(data.ids), 365, len(CARRA_VARS)), np.nan, dtype=np.float32)
        counts[year] = np.zeros((365, len(CARRA_VARS)), dtype=int)

    for name in (
        "carra_runoff_2018_2019.grib",
        "carra_albedo_2018_2019.grib",
        "carra_analysis_2018_2019.grib",
    ):
        path = cfg.raw / name
        for n, (offset, total, sections) in enumerate(iter_messages(path)):
            variable, reference, lead = message_key(sections)
            if variable is None:
                raise ValueError(f"unmapped parameter in {name} at offset {offset}")
            year, month, day = reference[0], reference[1], reference[2]
            if year not in out:
                continue

            if grid is None:
                grid = parse_gds(sections[3])
                for y, d in lakes.items():
                    lat, lon = lake_centroids(cfg, y, d.ids)
                    row, col, valid = lake_grid_indices(lat, lon, grid)
                    if not valid.all():
                        raise ValueError(f"{(~valid).sum()} lakes fall outside the CARRA grid in {y}")
                    index[y] = (row, col)

            field = decode_values(path, sections, grid)
            row, col = index[year]
            values = field[row, col] * ALBEDO_SCALE.get(variable, 1.0)

            doy = date(year, month, day).timetuple().tm_yday
            k = CARRA_VARS.index(variable)
            target = out[year][:, doy - 1, k]
            # Average repeated messages for the same day (albedo has two).
            if counts[year][doy - 1, k] == 0:
                target[:] = values
            else:
                target[:] = (target * counts[year][doy - 1, k] + values) / (
                    counts[year][doy - 1, k] + 1
                )
            counts[year][doy - 1, k] += 1

            if verbose and (n + 1) % 200 == 0:
                print(f"    {name}: {n + 1} messages", flush=True)
    return out


def fill_causal(tensor: np.ndarray, fill_values: list) -> np.ndarray:
    """LOCF along the day axis, then constants for the leading stretch.

    CARRA covers 1 May to 31 Oct only, so every lake starts and ends the
    calendar year unobserved. Forward fill carries the last known value across
    the winter tail; the leading days get the same data-independent constants
    the satellite features use.
    """
    out = tensor.copy()
    n_lakes, n_days, n_vars = out.shape
    for t in range(1, n_days):
        gap = np.isnan(out[:, t, :])
        out[:, t, :] = np.where(gap, out[:, t - 1, :], out[:, t, :])
    for k, value in enumerate(fill_values):
        out[:, :, k] = np.where(np.isnan(out[:, :, k]), value, out[:, :, k])
    return out
