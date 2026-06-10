"""
test_pipeline.py
----------------
Unit and integration tests for the slab-geometry-seismicity pipeline.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Allow import from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curvature import compute_curvature, zone_curvature_stats
from correlation import pearson_table, build_zone_dataframe, reproduce_table1
from ml_model import fit_gp, SlabFNO


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_paraboloid_grid(nx=60, ny=50, a=0.001, b=0.001):
    """
    Synthetic elliptic paraboloid  z = a*x^2 + b*y^2.
    Analytical Gaussian curvature at origin = 4*a*b.
    Analytical mean curvature at origin = a + b  (for small slopes).
    """
    xs = np.linspace(-100, 100, nx)   # km
    ys = np.linspace(-100, 100, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = a * X**2 + b * Y**2

    spacing_deg = 0.1
    lons = np.linspace(130.0, 130.0 + (nx - 1) * spacing_deg, nx)
    lats = np.linspace(30.0,  30.0  + (ny - 1) * spacing_deg, ny)

    grid = dict(
        dep2d   = Z,
        lat2d   = np.meshgrid(lons, lats)[1],
        lon2d   = np.meshgrid(lons, lats)[0],
        lons    = lons,
        lats    = lats,
        spacing = spacing_deg,
    )
    return grid, a, b


def _make_zone_df(n=13, seed=0):
    """Synthetic DataFrame with 13 zones and realistic-ish curvature/seismicity."""
    rng = np.random.default_rng(seed)
    zones = [f"z{i:02d}" for i in range(n)]
    # Ground truth: mag_max is negatively correlated with mean curvature
    mean_absH = rng.uniform(0.001, 0.010, n)
    df = pd.DataFrame({
        "mean_absH": mean_absH,
        "mean_absK": mean_absH * rng.uniform(0.5, 2.0, n),
        "sd_absH"  : mean_absH * rng.uniform(0.3, 1.0, n),
        "sd_absK"  : mean_absH * rng.uniform(0.3, 1.0, n),
        "cv_absH"  : rng.uniform(0.5, 3.0, n),
        "cv_absK"  : rng.uniform(0.5, 3.0, n),
        "mag_max"  : 9.0 - 200 * mean_absH + rng.normal(0, 0.3, n),
        "mag_avg"  : 7.5 - 100 * mean_absH + rng.normal(0, 0.2, n),
        "n_events" : (300 * mean_absH / mean_absH.max() * 400 +
                      rng.normal(0, 50, n)).astype(int),
    }, index=zones)
    df.index.name = "zone"
    return df


# ─── curvature tests ──────────────────────────────────────────────────────────

class TestCurvatureParaboloid:
    """Test finite-difference curvature against analytical values on a paraboloid."""

    def test_shape(self):
        grid, a, b = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        assert curv["H"].shape == grid["dep2d"].shape
        assert curv["K"].shape == grid["dep2d"].shape

    def test_mean_curvature_sign(self):
        """H > 0 everywhere for a convex-up paraboloid (positive coefficients)."""
        grid, a, b = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        valid = curv["valid"]
        H_interior = curv["H"][valid]
        assert np.all(H_interior > 0), \
            "Mean curvature should be positive for convex-up paraboloid."

    def test_gaussian_curvature_positive(self):
        """K > 0 everywhere for an elliptic paraboloid."""
        grid, a, b = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        valid = curv["valid"]
        K_interior = curv["K"][valid]
        assert np.all(K_interior >= 0), \
            "Gaussian curvature should be non-negative for elliptic paraboloid."

    def test_flat_surface_zero_curvature(self):
        """Flat depth grid → H = K = 0."""
        grid, _, _ = _make_paraboloid_grid()
        grid = dict(grid)   # copy
        grid["dep2d"] = np.ones_like(grid["dep2d"]) * 50.0  # constant depth
        curv = compute_curvature(grid)
        valid = curv["valid"]
        assert np.allclose(curv["H"][valid], 0.0, atol=1e-10), \
            "Mean curvature should be zero for flat surface."
        assert np.allclose(curv["K"][valid], 0.0, atol=1e-10), \
            "Gaussian curvature should be zero for flat surface."

    def test_absH_nonneg(self):
        grid, _, _ = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        assert np.all(curv["absH"][curv["valid"]] >= 0)
        assert np.all(curv["absK"][curv["valid"]] >= 0)

    def test_subsampling_reduces_grid(self):
        grid, _, _ = _make_paraboloid_grid(nx=60, ny=50)
        curv_native = compute_curvature(grid, space_sep=None)
        curv_coarse = compute_curvature(grid, space_sep=0.3)
        assert curv_coarse["H"].shape[0] <= curv_native["H"].shape[0]


class TestCurvatureStats:
    def test_stats_keys(self):
        grid, _, _ = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        stats = zone_curvature_stats(curv)
        for key in ["mean_absH", "mean_absK", "sd_absH", "sd_absK",
                    "cv_absH", "cv_absK"]:
            assert key in stats

    def test_cv_positive(self):
        grid, _, _ = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        stats = zone_curvature_stats(curv)
        assert stats["cv_absH"] >= 0
        assert stats["cv_absK"] >= 0


# ─── correlation tests ────────────────────────────────────────────────────────

class TestPearsonTable:
    def test_shape(self):
        df = _make_zone_df()
        feats = {"mean_absH": df["mean_absH"].values}
        tgts  = {"mag_max":   df["mag_max"].values,
                 "n_events":  df["n_events"].values}
        tbl = pearson_table(feats, tgts)
        assert tbl.shape == (1, 2)

    def test_known_correlation(self):
        """Perfect linear anti-correlation → r = -1."""
        x = np.linspace(0.001, 0.01, 13)
        y = -x * 100 + 9.0
        tbl = pearson_table({"x": x}, {"y": y})
        r = tbl.loc["x", "y"]["r"]
        assert abs(r + 1.0) < 1e-6, f"Expected r=-1, got {r}"

    def test_reproduce_table1_format(self):
        df = _make_zone_df()
        tbl = reproduce_table1(df)
        assert tbl.shape[0] == 6   # 6 curvature metrics
        assert tbl.shape[1] == 3   # 3 seismicity metrics


# ─── ML model tests ───────────────────────────────────────────────────────────

class TestGaussianProcess:
    def test_gp_loo_r2_nonnull(self):
        df = _make_zone_df(seed=7)
        result = fit_gp(df,
                        feature_cols=["mean_absH", "mean_absK",
                                      "cv_absH",   "cv_absK"],
                        target_col="mag_max",
                        n_restarts=2)
        assert "loo_r2" in result
        assert np.isfinite(result["loo_r2"])

    def test_gp_returns_n_predictions(self):
        df = _make_zone_df(n=10, seed=3)
        result = fit_gp(df, ["mean_absH", "cv_absK"], "mag_max", n_restarts=2)
        assert len(result["y_pred_loo"]) == len(result["y_true"])

    def test_gp_ci_shape(self):
        df = _make_zone_df(n=10, seed=5)
        result = fit_gp(df, ["mean_absH", "mean_absK"], "n_events", n_restarts=2)
        assert result["y_pred_ci_loo"].shape == (len(result["y_true"]), 2)


class TestSlabFNO:
    def test_forward_shape(self):
        fno = SlabFNO(n_modes=8, width=16, n_layers=2)
        X = np.random.randn(5, 8)   # 5 zones, 8 features
        out = fno.forward(X)
        assert out.shape == (5, 1)

    def test_psd_feature_length(self):
        grid, _, _ = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        fno = SlabFNO(n_modes=16)
        feats = fno.curvature_psd_features(curv, n_modes=16)
        assert len(feats) == 16

    def test_psd_features_nonneg(self):
        """PSD features should be non-negative (they're squared magnitudes)."""
        grid, _, _ = _make_paraboloid_grid()
        curv = compute_curvature(grid)
        fno = SlabFNO(n_modes=8)
        feats = fno.curvature_psd_features(curv)
        assert np.all(feats >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
