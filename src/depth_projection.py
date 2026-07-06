"""
depth_projection.py
-------------------
Surface-projected earthquake-to-slab assignment for the
slab-geometry-seismicity pipeline.

Motivation
~~~~~~~~~~
The baseline pipeline assigns earthquakes to a subduction zone using a
rectangular longitude/latitude bounding box.  That is crude: a box can
capture events that do not lie on the zone's slab surface (e.g. shallow
crustal events, or events belonging to an adjacent slab), which adds noise
to the per-zone seismicity counts and weakens the geometry-seismicity
correlations.

Chau, Bendick, Choi & Mahadevan (2026, arXiv:2606.02520) report stronger
correlations than the bounding-box reproduction recovers.  This module tests
the hypothesis that the gap is an assignment artifact by re-assigning each
earthquake using its distance to the Slab2 slab surface itself:

    For an earthquake at (lon, lat, depth_eq), interpolate the Slab2 slab
    depth  depth_slab(lon, lat).  The event is assigned to that slab iff

        | depth_eq  -  depth_slab(lon, lat) |  <=  tolerance_km

Because the true tolerance used by the paper is not stated, the driver runs
a sweep over several tolerances (default 20, 30, 50 km) and reports how the
correlations respond.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


# ── slab-depth interpolator ─────────────────────────────────────────────────────

def build_slab_interpolator(grid: dict):
    """
    Build a bilinear interpolator for slab depth over (lat, lon).

    Parameters
    ----------
    grid : dict from slab_loader.load_zone_grid(), with keys
           'lons', 'lats', 'dep2d'  (dep2d is positive-down km, NaN off-slab)

    Returns
    -------
    A callable f(lon, lat) -> slab depth (km, positive down), returning NaN
    where the query is outside the slab footprint or over a NaN cell.
    """
    lons = np.asarray(grid["lons"], dtype=float)
    lats = np.asarray(grid["lats"], dtype=float)
    dep2d = np.asarray(grid["dep2d"], dtype=float)  # shape (ny, nx) = (lat, lon)

    # RegularGridInterpolator needs strictly ascending axes.
    lon_order = np.argsort(lons)
    lat_order = np.argsort(lats)
    lons_s = lons[lon_order]
    lats_s = lats[lat_order]
    dep_s = dep2d[np.ix_(lat_order, lon_order)]

    interp = RegularGridInterpolator(
        (lats_s, lons_s), dep_s,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    def _f(lon, lon_is_array_lat=None):
        raise RuntimeError("call via query_depth")

    def query_depth(lon_arr, lat_arr):
        lon_arr = np.asarray(lon_arr, dtype=float)
        lat_arr = np.asarray(lat_arr, dtype=float)
        # Slab2 longitudes are stored 0..360 in places; normalise the query
        lon_q = np.where(lon_arr < lons_s.min() - 1e-6, lon_arr + 360.0, lon_arr)
        lon_q = np.where(lon_q > lons_s.max() + 1e-6, lon_q - 360.0, lon_q)
        pts = np.column_stack([lat_arr, lon_q])
        return interp(pts)

    return query_depth


# ── assignment ──────────────────────────────────────────────────────────────────

def _catalog_columns(catalog: pd.DataFrame):
    """Resolve the lon/lat/depth/mag column names from a ComCat CSV."""
    def pick(cands):
        for c in catalog.columns:
            if c.lower() in cands:
                return c
        return None
    lon = pick({"longitude", "lon"})
    lat = pick({"latitude", "lat"})
    dep = pick({"depth", "dep"})
    mag = pick({"mag", "magnitude"})
    return lon, lat, dep, mag


def assign_by_projection(catalog: pd.DataFrame, grid: dict,
                         tolerance_km: float) -> pd.DataFrame:
    """
    Keep only the earthquakes whose depth is within *tolerance_km* of the
    slab surface at their (lon, lat).

    Returns the filtered catalog (a copy).
    """
    if catalog is None or len(catalog) == 0:
        return catalog.copy() if catalog is not None else pd.DataFrame()

    lon_c, lat_c, dep_c, _ = _catalog_columns(catalog)
    if lon_c is None or lat_c is None or dep_c is None:
        # Cannot project without coordinates; return unfiltered (fail-open)
        return catalog.copy()

    lon = pd.to_numeric(catalog[lon_c], errors="coerce").to_numpy()
    lat = pd.to_numeric(catalog[lat_c], errors="coerce").to_numpy()
    dep = pd.to_numeric(catalog[dep_c], errors="coerce").to_numpy()

    query_depth = build_slab_interpolator(grid)
    slab_dep = query_depth(lon, lat)

    residual = np.abs(dep - slab_dep)
    keep = np.isfinite(residual) & (residual <= tolerance_km)

    return catalog.loc[keep].copy()


def productivity_by_projection(catalog: pd.DataFrame, grid: dict,
                               tolerance_km: float) -> dict:
    """
    Assign by projection, then compute seismicity productivity metrics.
    Mirrors earthquake_catalog.compute_productivity on the filtered catalog.
    """
    from .earthquake_catalog import compute_productivity
    filtered = assign_by_projection(catalog, grid, tolerance_km)
    stats = compute_productivity(filtered)
    stats["n_kept"] = len(filtered)
    stats["n_original"] = len(catalog) if catalog is not None else 0
    return stats
