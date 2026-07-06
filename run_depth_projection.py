"""
run_depth_projection.py
-----------------------
Internship deliverable #1: does surface-projected earthquake assignment
close the correlation gap versus Chau et al. (2026)?

Runs end to end:
  1. Download Slab2 depth grids + USGS ComCat catalogs (13 zones).
  2. Compute per-zone curvature statistics (unchanged from baseline).
  3. Compute per-zone seismicity two ways:
        (a) bounding-box assignment      -> baseline
        (b) depth-projection assignment  -> at tolerances 20, 30, 50 km
  4. Report Pearson r for the headline metrics under each method,
     alongside the published values.

Usage:
    python run_depth_projection.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src import (load_all_zones, download_all_zones,
                 download_all_catalogs,
                 compute_curvature, zone_curvature_stats,
                 compute_productivity, build_zone_dataframe)
from src.depth_projection import productivity_by_projection

# Published headline correlations (Chau et al. 2026, Table 1), vs M_max
PAPER = {
    "mean_absH": -0.59,
    "mean_absK": -0.60,
}

TOLERANCES_KM = [20.0, 30.0, 50.0]

HEADLINE_FEATURES = ["mean_absH", "mean_absK", "sd_absH", "sd_absK"]


def _corr(df, feat, target="mag_max"):
    x = df[feat].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return pearsonr(x[m], y[m])[0]


def main():
    print("=" * 70)
    print("DEPTH-PROJECTION vs BOUNDING-BOX  —  earthquake assignment study")
    print("=" * 70)

    # 1. data ---------------------------------------------------------------
    print("\n[1/4] Downloading Slab2 grids …")
    download_all_zones()
    grids = load_all_zones()

    print("[1/4] Downloading USGS catalogs …")
    catalogs = download_all_catalogs()

    # 2. curvature (method-independent) -------------------------------------
    print("[2/4] Computing curvature statistics …")
    curv_stats = {z: zone_curvature_stats(compute_curvature(g))
                  for z, g in grids.items()}

    # 3a. baseline: bounding-box seismicity ---------------------------------
    print("[3/4] Baseline (bounding-box) correlations …")
    prod_box = {z: compute_productivity(catalogs[z]) for z in grids}
    df_box = build_zone_dataframe(curv_stats, prod_box)

    results = {"paper": PAPER, "bounding_box": {}}
    for feat in HEADLINE_FEATURES:
        results["bounding_box"][feat] = _corr(df_box, feat)

    # 3b. depth-projection at each tolerance --------------------------------
    print("[3/4] Depth-projection correlations …")
    for tol in TOLERANCES_KM:
        prod_proj = {}
        kept_total = 0
        for z in grids:
            stats = productivity_by_projection(catalogs[z], grids[z], tol)
            prod_proj[z] = stats
            kept_total += stats.get("n_kept", 0)
        df_proj = build_zone_dataframe(curv_stats, prod_proj)
        key = f"proj_{int(tol)}km"
        results[key] = {feat: _corr(df_proj, feat) for feat in HEADLINE_FEATURES}
        results[key]["_events_kept"] = kept_total

    # 4. report -------------------------------------------------------------
    print("\n[4/4] RESULTS  (Pearson r vs M_max)\n")

    cols = ["bounding_box"] + [f"proj_{int(t)}km" for t in TOLERANCES_KM]
    header = f"{'metric':<12}" + "".join(f"{c:>14}" for c in cols) + f"{'paper':>10}"
    print(header)
    print("-" * len(header))
    for feat in HEADLINE_FEATURES:
        row = f"{feat:<12}"
        for c in cols:
            v = results[c].get(feat, np.nan)
            row += f"{v:>14.3f}" if np.isfinite(v) else f"{'--':>14}"
        pv = PAPER.get(feat)
        row += f"{pv:>10.2f}" if pv is not None else f"{'--':>10}"
        print(row)

    print("\nEvents retained after projection:")
    box_total = sum(len(catalogs[z]) for z in grids)
    print(f"  bounding_box : {box_total}")
    for t in TOLERANCES_KM:
        k = f"proj_{int(t)}km"
        print(f"  proj_{int(t)}km   : {results[k]['_events_kept']}")

    # verdict ---------------------------------------------------------------
    print("\n" + "=" * 70)
    base = abs(results["bounding_box"]["mean_absH"])
    best_key = max((f"proj_{int(t)}km" for t in TOLERANCES_KM),
                   key=lambda k: abs(results[k]["mean_absH"]))
    best = abs(results[best_key]["mean_absH"])
    paper = abs(PAPER["mean_absH"])
    print(f"Mean|H| vs M_max:  box={base:.3f}   best_proj={best:.3f} "
          f"({best_key})   paper={paper:.3f}")
    if best > base:
        closed = (best - base) / (paper - base) * 100 if paper > base else 0
        print(f"Depth projection strengthened the correlation, closing "
              f"~{closed:.0f}% of the gap to the published value.")
    else:
        print("Depth projection did not strengthen the correlation; "
              "the gap has another source.")
    print("=" * 70)


if __name__ == "__main__":
    main()
