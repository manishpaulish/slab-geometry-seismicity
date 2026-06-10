"""
slab_loader.py
--------------
Downloads and loads Slab2 depth grids for the 13 subduction zones
used in Chau, Bendick, Choi, Mahadevan (2026), arXiv:2606.02520.

The published Slab2 v2 grids (Hayes et al. 2018, Science) are served
directly from the usgs/slab2 GitHub repository (0418database branch).
Each file is a GMT/NetCDF grid with columns: lon, lat, depth (km, negative down).

Zone codes follow the Slab2 convention:
    alu  Aleutians          cam  Central America
    cas  Cascadia           izu  Izu-Bonin
    ker  Kermadec           kur  Kuril
    phi  Philippines        ryu  Ryukyu
    sam  South America      sco  Scotia
    sol  Solomon Islands    sum  Sumatra/Java
    van  Vanuatu
"""

import os
import requests
import numpy as np
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

ZONES = ["alu", "cam", "cas", "izu", "ker", "kur",
         "phi", "ryu", "sam", "sco", "sol", "sum", "van"]

# Hayes et al. 2018 (April release) — the version used in Chau et al. 2026
_SLAB2_RELEASE_TAG = "02.24.18"

# Direct raw download base from usgs/slab2 GitHub (0418database folder).
# Each depth grid: [zone]_slab2_dep_02.24.18.csv  (lon, lat, dep columns)
_GITHUB_BASE = (
    "https://raw.githubusercontent.com/usgs/slab2/master/0418database"
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _local_cache(data_dir: Path, zone: str) -> Path:
    return data_dir / f"{zone}_slab2_dep_{_SLAB2_RELEASE_TAG}.csv"


def download_zone(zone: str, data_dir: Path, force: bool = False) -> Path:
    """
    Download the Slab2 depth CSV for *zone* into *data_dir*.
    Returns the local path.  Skips download if file already exists
    (unless force=True).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = _local_cache(data_dir, zone)

    if dest.exists() and not force:
        return dest

    url = f"{_GITHUB_BASE}/{zone}_slab2_dep_{_SLAB2_RELEASE_TAG}.csv"
    print(f"  Downloading {zone} depth grid … ", end="", flush=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    size_kb = dest.stat().st_size / 1024
    print(f"done ({size_kb:.0f} KB)")
    return dest


def download_all_zones(data_dir: str = "data/slab2", force: bool = False) -> dict:
    """
    Download depth grids for all 13 zones.
    Returns {zone_code: local_path}.
    """
    data_dir = Path(data_dir)
    paths = {}
    print("Downloading Slab2 depth grids …")
    for zone in ZONES:
        try:
            paths[zone] = download_zone(zone, data_dir, force=force)
        except requests.HTTPError as exc:
            print(f"  WARNING: could not fetch {zone}: {exc}")
    print(f"Done. {len(paths)}/{len(ZONES)} zones available.")
    return paths


def load_zone_grid(path: Path) -> dict:
    """
    Load a Slab2 depth CSV into a regular lon/lat grid.

    Returns a dict with keys:
        lon2d  : (ny, nx) float array, longitude (°E)
        lat2d  : (ny, nx) float array, latitude (°N)
        dep2d  : (ny, nx) float array, depth (km, positive down)
        lons   : 1-D unique longitudes
        lats   : 1-D unique latitudes
        spacing: float, grid spacing (°)
    """
    # Slab2 CSV columns: lon, lat, dep  (dep is negative = km below surface)
    data = np.genfromtxt(path, delimiter=",", skip_header=1,
                         usecols=(0, 1, 2), filling_values=np.nan)
    lon_flat, lat_flat, dep_flat = data[:, 0], data[:, 1], data[:, 2]

    # Remove NaN rows
    valid = np.isfinite(dep_flat)
    lon_flat, lat_flat, dep_flat = lon_flat[valid], lat_flat[valid], dep_flat[valid]

    # Convert depth to positive-down convention
    dep_flat = np.abs(dep_flat)

    lons = np.sort(np.unique(np.round(lon_flat, 4)))
    lats = np.sort(np.unique(np.round(lat_flat, 4)))
    spacing_lon = float(np.median(np.diff(lons))) if len(lons) > 1 else 0.05
    spacing_lat = float(np.median(np.diff(lats))) if len(lats) > 1 else 0.05
    spacing = round((spacing_lon + spacing_lat) / 2, 4)

    nx, ny = len(lons), len(lats)
    lon2d = np.full((ny, nx), np.nan)
    lat2d = np.full((ny, nx), np.nan)
    dep2d = np.full((ny, nx), np.nan)

    # Map each point into the grid
    lon_idx = np.searchsorted(lons, np.round(lon_flat, 4))
    lat_idx = np.searchsorted(lats, np.round(lat_flat, 4))

    # Guard against out-of-range indices from floating-point rounding
    mask = (lon_idx < nx) & (lat_idx < ny)
    lon2d[lat_idx[mask], lon_idx[mask]] = lon_flat[mask]
    lat2d[lat_idx[mask], lon_idx[mask]] = lat_flat[mask]
    dep2d[lat_idx[mask], lon_idx[mask]] = dep_flat[mask]

    return dict(lon2d=lon2d, lat2d=lat2d, dep2d=dep2d,
                lons=lons, lats=lats, spacing=spacing)


def load_all_zones(data_dir: str = "data/slab2") -> dict:
    """
    Load all downloaded Slab2 grids.
    Returns {zone_code: grid_dict}.
    """
    data_dir = Path(data_dir)
    grids = {}
    for zone in ZONES:
        path = _local_cache(data_dir, zone)
        if path.exists():
            grids[zone] = load_zone_grid(path)
        else:
            print(f"  WARNING: {zone} not found — run download_all_zones() first.")
    return grids
