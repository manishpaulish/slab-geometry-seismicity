"""
diagnose_projection.py
----------------------
(1) Verify the identical proj_20km / proj_30km correlations are real physics,
    not a filtering artifact, by printing per-zone retained-event counts and
    M_max at each tolerance.
(2) Extend the tolerance sweep to 75 and 100 km to locate where the
    correlation peaks.

Usage:
    python diagnose_projection.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src import (load_all_zones, download_all_zones,
                 download_all_catalogs,
                 compute_curvature, zone_curvature_stats,
                 compute_productivity, build_zone_dataframe)
from src.depth_projection import assign_by_projection, productivity_by_projection

TOLERANCES = [20.0, 30.0, 50.0, 75.0, 100.0]
PAPER = {"mean_absH": -0.59, "mean_absK": -0.60}
FEATURES = ["mean_absH", "mean_absK", "sd_absH", "sd_absK"]


def _corr(df, feat, target="mag_max"):
    x = df[feat].to_numpy(float); y = df[target].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    return pearsonr(x[m], y[m])[0] if m.sum() >= 3 else np.nan


def main():
    download_all_zones()
    grids = load_all_zones()
    catalogs = download_all_catalogs()
    curv = {z: zone_curvature_stats(compute_curvature(g)) for z, g in grids.items()}

    # ---- PART 1: per-zone verification of the 20 vs 30 km identity ----------
    print("=" * 78)
    print("PART 1  —  per-zone events retained and M_max at each tolerance")
    print("=" * 78)
    zones = list(grids.keys())

    # counts
    print(f"\n{'zone':<6}" + "".join(f"{'n@'+str(int(t)):>9}" for t in TOLERANCES))
    print("-" * (6 + 9 * len(TOLERANCES)))
    for z in zones:
        row = f"{z:<6}"
        for t in TOLERANCES:
            kept = assign_by_projection(catalogs[z], grids[z], t)
            row += f"{len(kept):>9}"
        print(row)

    # M_max per zone per tolerance (this drives the correlation)
    print(f"\n{'zone':<6}" + "".join(f"{'Mmax@'+str(int(t)):>10}" for t in TOLERANCES))
    print("-" * (6 + 10 * len(TOLERANCES)))
    mmax_by_tol = {t: {} for t in TOLERANCES}
    for z in zones:
        row = f"{z:<6}"
        for t in TOLERANCES:
            stats = productivity_by_projection(catalogs[z], grids[z], t)
            mm = stats.get("mag_max", np.nan)
            mmax_by_tol[t][z] = mm
            row += f"{mm:>10.2f}" if np.isfinite(mm) else f"{'--':>10}"
        print(row)

    # explicit check: does M_max differ anywhere between 20 and 30 km?
    diffs = [z for z in zones
             if not np.isclose(mmax_by_tol[20.0][z], mmax_by_tol[30.0][z], equal_nan=True)]
    print("\nZones where M_max changes between 20 km and 30 km:",
          diffs if diffs else "NONE")
    if not diffs:
        print("  -> identical 20/30 km correlations are REAL: the largest event in")
        print("     every zone already lies within 20 km of the slab, so widening to")
        print("     30 km adds only smaller events and cannot move M_max.")
    else:
        print("  -> M_max DOES change; identical correlations would indicate a bug.")

    # ---- PART 2: extended tolerance sweep ----------------------------------
    print("\n" + "=" * 78)
    print("PART 2  —  extended tolerance sweep (Pearson r vs M_max)")
    print("=" * 78 + "\n")

    prod_box = {z: compute_productivity(catalogs[z]) for z in zones}
    df_box = build_zone_dataframe(curv, prod_box)

    cols = ["box"] + [f"{int(t)}km" for t in TOLERANCES]
    print(f"{'metric':<12}" + "".join(f"{c:>10}" for c in cols) + f"{'paper':>9}")
    print("-" * (12 + 10 * len(cols) + 9))

    corr_table = {"box": {f: _corr(df_box, f) for f in FEATURES}}
    for t in TOLERANCES:
        prod = {z: productivity_by_projection(catalogs[z], grids[z], t) for z in zones}
        dfp = build_zone_dataframe(curv, prod)
        corr_table[f"{int(t)}km"] = {f: _corr(dfp, f) for f in FEATURES}

    for f in FEATURES:
        row = f"{f:<12}"
        for c in cols:
            row += f"{corr_table[c][f]:>10.3f}"
        row += f"{PAPER.get(f, float('nan')):>9.2f}" if f in PAPER else f"{'--':>9}"
        print(row)

    # find peak tolerance for the headline metric
    print("\nPeak search (mean_absH):")
    best_c = max(cols, key=lambda c: abs(corr_table[c]["mean_absH"]))
    print(f"  strongest at: {best_c}  (r = {corr_table[best_c]['mean_absH']:.3f})")
    if best_c == cols[-1]:
        print("  NOTE: still climbing at the widest tolerance — consider going wider.")
    else:
        print("  Correlation peaks at a finite tolerance, then declines — a physical scale.")


if __name__ == "__main__":
    main()
