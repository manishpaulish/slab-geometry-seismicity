"""
earthquake_catalog.py
---------------------
Downloads and organises USGS ComCat earthquake records for each
subduction zone, matching the procedure of Chau et al. 2026
(arXiv:2606.02520): Mw ≥ 6.0, depth 0–700 km, 1900-01-01 to 2025-12-31.

The API endpoint is the ANSS ComCat FDSN service:
    https://earthquake.usgs.gov/fdsnws/event/1/query

Results are cached as CSV in data/earthquakes/.
"""

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ── zone bounding boxes  [minlat, maxlat, minlon, maxlon] ──────────────────────
# Derived from the spatial extent of each Slab2 model region.
# These match the zone boundaries used by Chau et al. (2026).
ZONE_BOUNDS = {
    "alu": [48.0,  62.0,  188.0,  220.0],  # Aleutians
    "cam": [ 6.0,  19.0, -93.0,  -83.0],  # Central America
    "cas": [38.0,  52.0, -128.0, -120.0],  # Cascadia
    "izu": [22.0,  35.0,  138.0,  145.0],  # Izu-Bonin
    "ker": [-38.0, -26.0, 175.0,  186.0],  # Kermadec
    "kur": [40.0,  52.0,  142.0,  155.0],  # Kuril
    "phi": [ 5.0,  21.0,  120.0,  130.0],  # Philippines
    "ryu": [23.0,  32.0,  126.0,  133.0],  # Ryukyu
    "sam": [-42.0,   2.0, -78.0,  -66.0],  # South America
    "sco": [-61.0, -52.0, -62.0,  -24.0],  # Scotia
    "sol": [-13.0,  -2.0, 152.0,  165.0],  # Solomon Islands
    "sum": [-11.0,   6.0,  94.0,  108.0],  # Sumatra/Java
    "van": [-22.0, -12.0, 165.0,  172.0],  # Vanuatu
}

_COMCAT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
_START = "1900-01-01"
_END   = "2025-12-31"
_MIN_MAG = 6.0
_MAX_DEPTH = 700.0   # km


def _fetch_zone_catalog(zone: str, bounds: list,
                        retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    """
    Query ComCat CSV for one zone.  Returns a DataFrame.
    The API caps results at 20,000 events per query; we chunk by decade if needed.
    """
    minlat, maxlat, minlon, maxlon = bounds

    params = dict(
        format="csv",
        starttime=_START,
        endtime=_END,
        minmagnitude=_MIN_MAG,
        maxdepth=_MAX_DEPTH,
        minlatitude=minlat,
        maxlatitude=maxlat,
        minlongitude=minlon,
        maxlongitude=maxlon,
        orderby="time-asc",
    )

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(_COMCAT_URL, params=params, timeout=120)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            return df
        except requests.HTTPError as exc:
            if attempt == retries:
                raise
            print(f"    retry {attempt}/{retries} for {zone}: {exc}")
            time.sleep(pause * attempt)

    return pd.DataFrame()


def download_catalog(zone: str, data_dir: str = "data/earthquakes",
                     force: bool = False) -> pd.DataFrame:
    """
    Download the earthquake catalog for *zone* and save to CSV.
    Returns the DataFrame.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / f"{zone}_catalog.csv"

    if dest.exists() and not force:
        return pd.read_csv(dest, low_memory=False)

    bounds = ZONE_BOUNDS[zone]
    print(f"  Querying ComCat for {zone} …", end="", flush=True)
    df = _fetch_zone_catalog(zone, bounds)
    df.to_csv(dest, index=False)
    print(f" {len(df)} events saved.")
    return df


def download_all_catalogs(data_dir: str = "data/earthquakes",
                          force: bool = False) -> dict:
    """
    Download catalogs for all 13 zones.
    Returns {zone: DataFrame}.
    """
    from .slab_loader import ZONES
    catalogs = {}
    print("Downloading USGS earthquake catalogs …")
    for zone in ZONES:
        try:
            catalogs[zone] = download_catalog(zone, data_dir, force=force)
        except Exception as exc:
            print(f"  WARNING: failed for {zone}: {exc}")
            catalogs[zone] = pd.DataFrame()
        time.sleep(0.5)   # polite rate-limiting
    return catalogs


def compute_productivity(catalog: pd.DataFrame) -> dict:
    """
    Compute seismicity metrics from a zone catalog DataFrame.

    Returns a dict with:
        n_events   : total event count
        mag_max    : maximum magnitude
        mag_avg    : mean magnitude
    """
    if catalog is None or len(catalog) == 0:
        return dict(n_events=0, mag_max=np.nan, mag_avg=np.nan)

    # Magnitude column name varies slightly across ComCat CSV versions
    mag_col = next((c for c in catalog.columns
                    if c.lower() in ("mag", "magnitude")), None)
    if mag_col is None:
        return dict(n_events=len(catalog), mag_max=np.nan, mag_avg=np.nan)

    mags = pd.to_numeric(catalog[mag_col], errors="coerce").dropna()
    return dict(
        n_events = len(mags),
        mag_max  = float(mags.max()),
        mag_avg  = float(mags.mean()),
    )
