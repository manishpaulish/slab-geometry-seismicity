"""
correlation.py
--------------
Reproduces and extends the Pearson correlation analysis of
Chau, Bendick, Choi & Mahadevan (2026), arXiv:2606.02520.

Table 1 of that paper reports correlations between six curvature metrics
(Mean|H|, Mean|K|, SD|H|, SD|K|, CV|H|, CV|K|) and three seismicity
measures (M_max, M_avg, n_events) for 13 subduction zones.

This module:
  1. Reproduces that exact table.
  2. Runs a multi-scale sensitivity analysis (5 space-separation values).
  3. Produces a tidy DataFrame ready for plotting or ML input.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def pearson_table(feature_dict: dict, target_dict: dict,
                  alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute Pearson r between every (feature, target) pair.

    Parameters
    ----------
    feature_dict : {col_name: array-like of length N}
    target_dict  : {col_name: array-like of length N}
    alpha        : significance level for p-value flagging

    Returns
    -------
    DataFrame with rows = features, columns = targets.
    Each cell is a dict {'r': float, 'p': float, 'sig': bool}.
    For display, call .applymap(lambda d: f"{d['r']:.4f}{'*' if d['sig'] else ''}").
    """
    features = list(feature_dict.keys())
    targets  = list(target_dict.keys())
    rows = []
    for feat in features:
        row = {}
        x = np.asarray(feature_dict[feat], dtype=float)
        for tgt in targets:
            y = np.asarray(target_dict[tgt], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 3:
                row[tgt] = dict(r=np.nan, p=np.nan, sig=False)
            else:
                r, p = stats.pearsonr(x[valid], y[valid])
                row[tgt] = dict(r=float(r), p=float(p), sig=(p < alpha))
        rows.append(row)
    return pd.DataFrame(rows, index=features)


def build_zone_dataframe(curvature_stats: dict,
                         productivity_stats: dict) -> pd.DataFrame:
    """
    Merge curvature and seismicity statistics into a tidy DataFrame.

    Parameters
    ----------
    curvature_stats   : {zone: dict from curvature.zone_curvature_stats()}
    productivity_stats: {zone: dict from earthquake_catalog.compute_productivity()}

    Returns
    -------
    DataFrame with one row per zone and columns for all metrics.
    """
    records = []
    for zone in curvature_stats:
        if zone not in productivity_stats:
            continue
        row = {"zone": zone}
        row.update(curvature_stats[zone])
        row.update(productivity_stats[zone])
        records.append(row)
    return pd.DataFrame(records).set_index("zone")


def reproduce_table1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Table 1 of Chau et al. (2026): Pearson correlation
    coefficients between curvature metrics and earthquake productivity.

    Input df must have columns:
        mean_absH, mean_absK, sd_absH, sd_absK, cv_absH, cv_absK
        mag_max, mag_avg, n_events

    Returns a formatted DataFrame with bolded (|r|≥0.5) entries marked with *.
    """
    feature_cols = ["mean_absH", "mean_absK", "sd_absH",
                    "sd_absK", "cv_absH",  "cv_absK"]
    target_cols  = ["mag_max", "mag_avg", "n_events"]

    feat_dict = {col: df[col].values for col in feature_cols if col in df.columns}
    tgt_dict  = {col: df[col].values for col in target_cols  if col in df.columns}

    raw = pearson_table(feat_dict, tgt_dict)

    # Format: show r value; bold (mark with **) if |r| >= 0.5
    display_rows = []
    for feat in raw.index:
        row = {}
        for tgt in raw.columns:
            cell = raw.loc[feat, tgt]
            r = cell["r"]
            row[tgt] = f"{r:+.4f}{'**' if abs(r) >= 0.5 else ''}"
        display_rows.append(row)

    result = pd.DataFrame(display_rows, index=raw.index,
                          columns=raw.columns)
    result.index.name = "Curvature metric (X)"
    result.columns.name = "Seismicity metric (Y)"
    return result


def multiscale_correlation(grids: dict, catalogs: dict,
                            space_seps: Optional[list] = None) -> pd.DataFrame:
    """
    Run correlation analysis at multiple spatial scales (Chau et al. §S4).

    Parameters
    ----------
    grids       : {zone: grid_dict} from slab_loader.load_all_zones()
    catalogs    : {zone: DataFrame} from earthquake_catalog.download_all_catalogs()
    space_seps  : list of space-separation values in degrees (default 5)

    Returns
    -------
    DataFrame with columns: space_sep, metric_pair, pearson_r
    """
    from .curvature import compute_curvature, zone_curvature_stats
    from .earthquake_catalog import compute_productivity

    if space_seps is None:
        space_seps = [None]   # native spacing only

    records = []
    for sep in space_seps:
        # Compute curvature stats at this scale
        curv_stats = {}
        for zone, grid in grids.items():
            try:
                curv = compute_curvature(grid, space_sep=sep)
                curv_stats[zone] = zone_curvature_stats(curv)
            except Exception:
                continue

        prod_stats = {z: compute_productivity(catalogs.get(z)) for z in grids}
        df = build_zone_dataframe(curv_stats, prod_stats)
        if df.empty:
            continue

        # Compute correlations for key pairs
        for feat in ["mean_absH", "mean_absK", "sd_absK", "cv_absK"]:
            for tgt in ["mag_max", "mag_avg", "n_events"]:
                if feat not in df.columns or tgt not in df.columns:
                    continue
                x = df[feat].values.astype(float)
                y = df[tgt].values.astype(float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 3:
                    continue
                r, p = stats.pearsonr(x[valid], y[valid])
                records.append(dict(
                    space_sep = sep if sep is not None else grids[list(grids.keys())[0]]["spacing"],
                    feature   = feat,
                    target    = tgt,
                    pearson_r = float(r),
                    p_value   = float(p),
                ))

    return pd.DataFrame(records)
