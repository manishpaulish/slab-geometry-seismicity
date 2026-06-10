"""
curvature.py
------------
Computes mean curvature H and Gaussian curvature K of a subduction slab
surface from Slab2 depth grids, using the finite-difference formulae for
the first and second fundamental forms of a parametric surface.

This reproduces the curvature analysis of Chau, Bendick, Choi, Mahadevan
(2026), arXiv:2606.02520, Section S1 ("Slab surface geometry").

Mathematical background
~~~~~~~~~~~~~~~~~~~~~~~
Given a depth grid z = z(x, y) where x = lon * cos(lat) * R_E and
y = lat * R_E (equirectangular approximation in km), the surface is
parameterised as r(x,y) = (x, y, z(x,y)).

First fundamental form coefficients:
    E = 1 + z_x²,  F = z_x z_y,  G = 1 + z_y²

Second fundamental form coefficients (with unit normal n):
    L = z_xx / sqrt(1 + z_x² + z_y²)
    M = z_xy / sqrt(1 + z_x² + z_y²)
    N = z_yy / sqrt(1 + z_x² + z_y²)

Gaussian curvature:   K = (LN - M²) / (EG - F²)
Mean curvature:       H = (EN + GL - 2FM) / (2(EG - F²))

All curvature values are in km⁻¹.
"""

import numpy as np
from scipy.ndimage import uniform_filter

# Earth radius for degree-to-km conversion
_R_EARTH_KM = 6371.0


def _deg_to_km_scale(lat_deg: float) -> tuple:
    """
    Returns (dx_per_deg_lon, dy_per_deg_lat) in km at a given latitude.
    """
    lat_rad = np.deg2rad(lat_deg)
    dy = np.pi * _R_EARTH_KM / 180.0              # km per degree latitude
    dx = dy * np.cos(lat_rad)                       # km per degree longitude
    return dx, dy


def compute_curvature(grid: dict, space_sep: float = None) -> dict:
    """
    Compute H and K for a slab depth grid.

    Parameters
    ----------
    grid : dict
        Output of slab_loader.load_zone_grid() — must contain
        'dep2d', 'lat2d', 'lats', 'lons', 'spacing'.
    space_sep : float or None
        If given, spatially subsample the grid to this spacing (degrees)
        before computing curvature (Chau et al. §S4 multi-scale analysis).
        Must be a multiple of grid['spacing'].  If None, use native spacing.

    Returns
    -------
    dict with keys:
        H       : (ny, nx) mean curvature array (km⁻¹)
        K       : (ny, nx) Gaussian curvature array (km⁻¹²)
        absH    : |H|
        absK    : |K|
        valid   : boolean mask of valid (non-NaN) grid points
        spacing : float, effective spacing used (degrees)
    """
    dep2d = grid["dep2d"].copy()
    lats  = grid["lats"]
    lons  = grid["lons"]
    spacing = grid["spacing"]

    # ── optional spatial subsampling ──────────────────────────────────────────
    if space_sep is not None and space_sep > spacing:
        step = max(1, round(space_sep / spacing))
        dep2d = dep2d[::step, ::step]
        lats  = lats[::step]
        lons  = lons[::step]
        spacing = float(lats[1] - lats[0]) if len(lats) > 1 else spacing

    ny, nx = dep2d.shape
    if ny < 3 or nx < 3:
        return dict(H=np.full_like(dep2d, np.nan),
                    K=np.full_like(dep2d, np.nan),
                    absH=np.full_like(dep2d, np.nan),
                    absK=np.full_like(dep2d, np.nan),
                    valid=np.zeros_like(dep2d, dtype=bool),
                    spacing=spacing)

    # Mean latitude for dx/dy conversion
    lat_center = float(np.nanmean(lats))
    dx_per_deg, dy_per_deg = _deg_to_km_scale(lat_center)
    dx = spacing * dx_per_deg   # km between columns
    dy = spacing * dy_per_deg   # km between rows

    # ── first derivatives (central differences, interior; one-sided at edges) ─
    z = dep2d  # shape (ny, nx)

    # Pad with NaN so np.gradient handles NaN borders gracefully
    # np.gradient uses second-order central differences in the interior
    # and first-order differences at the boundary.
    z_x = np.full_like(z, np.nan)
    z_y = np.full_like(z, np.nan)
    z_xx = np.full_like(z, np.nan)
    z_yy = np.full_like(z, np.nan)
    z_xy = np.full_like(z, np.nan)

    valid_full = np.isfinite(z)

    # Compute gradients on the full grid; NaN propagation is acceptable here.
    # We mask out results at points where the stencil crossed a NaN below.
    with np.errstate(invalid="ignore"):
        gy, gx = np.gradient(z, dy, dx)          # ∂z/∂y, ∂z/∂x
        gyy, gyx = np.gradient(gy, dy, dx)
        _,   gxx = np.gradient(gx, dy, dx)

    z_x[:] = gx
    z_y[:] = gy
    z_xx[:] = gxx
    z_yy[:] = gyy
    z_xy[:] = gyx   # mixed partial (should be equal to gxy by Schwarz)

    # ── fundamental forms ─────────────────────────────────────────────────────
    E = 1.0 + z_x**2
    F = z_x * z_y
    G = 1.0 + z_y**2

    det_I = E * G - F**2    # determinant of first form (always > 0)
    det_I = np.where(det_I > 1e-12, det_I, np.nan)

    denom = np.sqrt(1.0 + z_x**2 + z_y**2)
    L = z_xx / denom
    M = z_xy / denom
    N = z_yy / denom

    # ── curvatures ────────────────────────────────────────────────────────────
    K = (L * N - M**2) / det_I
    H = (E * N + G * L - 2.0 * F * M) / (2.0 * det_I)

    # Mask boundary and NaN-contaminated points
    # (points where any stencil value was NaN will already be NaN)
    valid = valid_full & np.isfinite(K) & np.isfinite(H)

    # Zero out curvature at invalid points for downstream safety
    K = np.where(valid, K, np.nan)
    H = np.where(valid, H, np.nan)

    return dict(
        H       = H,
        K       = K,
        absH    = np.abs(H),
        absK    = np.abs(K),
        valid   = valid,
        spacing = spacing,
    )


def zone_curvature_stats(curv: dict) -> dict:
    """
    Compute the aggregate curvature statistics for one zone,
    matching the metrics used in Chau et al. Table 1.

    Returns a dict with:
        mean_absH, mean_absK  : mean of |H|, |K|
        sd_absH,   sd_absK    : standard deviation of |H|, |K|
        cv_absH,   cv_absK    : coefficient of variation (SD/mean) of |H|, |K|
    """
    valid = curv["valid"]
    absH = curv["absH"][valid]
    absK = curv["absK"][valid]

    def _stats(arr):
        if len(arr) == 0:
            return np.nan, np.nan, np.nan
        m  = np.mean(arr)
        sd = np.std(arr, ddof=1)
        cv = sd / m if m > 1e-15 else np.nan
        return float(m), float(sd), float(cv)

    mH, sdH, cvH = _stats(absH)
    mK, sdK, cvK = _stats(absK)

    return dict(
        mean_absH = mH, sd_absH = sdH, cv_absH = cvH,
        mean_absK = mK, sd_absK = sdK, cv_absK = cvK,
    )
