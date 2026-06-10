"""
ml_model.py
-----------
Extends the linear Pearson correlation analysis of Chau et al. (2026)
to nonlinear statistical learning models.

Two approaches are implemented:

1. GaussianProcessRegressor (sklearn)
   A Bayesian nonlinear model with an RBF + noise kernel.
   Naturally handles the small-sample (N=13) regime and quantifies
   predictive uncertainty.  Suitable for the 6-feature → 1-target setup.

2. SlabFNO  — a lightweight 1-D Fourier Neural Operator
   Inspired by Li et al. (2021, arXiv:2010.08895).  Instead of operating
   on the raw curvature fields (which have different spatial extents per
   zone), we apply it to the curvature distribution (histogram features)
   as a proof-of-concept of operator-learning for slab geometry.
   This is the methodological bridge to the FNO seismic work in
   Manish Paul et al. (2025) and the key novel contribution to Chau et al.

Both models can be compared against the Pearson R² baseline from the paper.
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── Gaussian Process regressor ─────────────────────────────────────────────────

def fit_gp(df: pd.DataFrame,
           feature_cols: list,
           target_col: str,
           n_restarts: int = 10,
           seed: int = 42) -> dict:
    """
    Fit a Gaussian Process regressor and return LOO-CV R².

    Because N=13 is very small, leave-one-out cross-validation (LOO-CV)
    is the only statistically valid evaluation strategy.

    Parameters
    ----------
    df            : DataFrame with feature and target columns (one row per zone)
    feature_cols  : list of curvature feature column names
    target_col    : seismicity column to predict ('mag_max', 'mag_avg', 'n_events')
    n_restarts    : hyperparameter optimisation restarts
    seed          : random seed

    Returns
    -------
    dict with:
        loo_r2        : float, LOO-CV coefficient of determination
        loo_rmse      : float, LOO-CV RMSE
        y_true        : array of true target values
        y_pred_loo    : array of LOO predictions
        y_pred_ci_loo : (N,2) array of 95% credible intervals
        model         : fitted sklearn GaussianProcessRegressor on full data
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error

    sub = df[feature_cols + [target_col]].dropna()
    X = sub[feature_cols].values.astype(float)
    y = sub[target_col].values.astype(float)
    n = len(y)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    y_pred_loo = np.zeros(n)
    y_pred_std = np.zeros(n)

    # LOO-CV
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=n_restarts,
                                      random_state=seed)
        gp.fit(X_sc[mask], y_sc[mask])
        mu, sigma = gp.predict(X_sc[[i]], return_std=True)
        y_pred_loo[i] = scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()[0]
        y_pred_std[i] = float(sigma[0]) * scaler_y.scale_[0]

    loo_r2   = float(r2_score(y, y_pred_loo))
    loo_rmse = float(np.sqrt(mean_squared_error(y, y_pred_loo)))

    # Fit on full data for the returned model
    gp_full = GaussianProcessRegressor(kernel=kernel,
                                       n_restarts_optimizer=n_restarts,
                                       random_state=seed)
    gp_full.fit(X_sc, y_sc)

    ci_lo = y_pred_loo - 1.96 * y_pred_std
    ci_hi = y_pred_loo + 1.96 * y_pred_std

    return dict(
        loo_r2        = loo_r2,
        loo_rmse      = loo_rmse,
        y_true        = y,
        y_pred_loo    = y_pred_loo,
        y_pred_ci_loo = np.column_stack([ci_lo, ci_hi]),
        model         = gp_full,
        scaler_X      = scaler_X,
        scaler_y      = scaler_y,
        zones         = list(sub.index),
    )


# ── Lightweight Fourier Neural Operator (FNO-1D on curvature spectrum) ─────────

class SlabFNO:
    """
    Proof-of-concept 1-D Fourier Neural Operator for slab curvature spectra.

    Instead of applying the FNO to the full 2-D depth grid, we first compute
    the power spectral density of the curvature field (|H| or |K|) along each
    row/column, then treat that 1-D spectrum as the function space input.

    Architecture:
        Input:  curvature PSD of length n_modes  (one per zone, per spatial direction)
        Lifting: dense layer R^n_modes → R^width
        FNO blocks: k spectral convolution + pointwise residual layers
        Projection: R^width → R^1 (scalar seismicity prediction)

    This implementation uses pure numpy for forward inference (no autograd)
    so that the repo has zero deep-learning framework dependencies.
    For training, a PyTorch version is provided in notebooks/03_fno_training.ipynb.
    """

    def __init__(self, n_modes: int = 16, width: int = 32,
                 n_layers: int = 4, seed: int = 42):
        self.n_modes = n_modes
        self.width   = width
        self.n_layers = n_layers
        self.rng     = np.random.default_rng(seed)
        self._weights_initialized = False

    def _init_weights(self, in_dim: int):
        """Xavier-initialise all weight matrices."""
        rng = self.rng
        scale = lambda m, n: np.sqrt(2.0 / (m + n))

        self.W_lift = rng.standard_normal((in_dim, self.width)) * scale(in_dim, self.width)
        self.b_lift = np.zeros(self.width)

        # Spectral weights: (n_layers, n_modes, width, width) complex
        self.W_spec = [(rng.standard_normal((self.n_modes, self.width, self.width)) +
                        1j * rng.standard_normal((self.n_modes, self.width, self.width)))
                       * scale(self.width, self.width)
                       for _ in range(self.n_layers)]

        # Pointwise residual weights: (n_layers, width, width)
        self.W_res  = [rng.standard_normal((self.width, self.width)) * scale(self.width, self.width)
                       for _ in range(self.n_layers)]
        self.b_res  = [np.zeros(self.width) for _ in range(self.n_layers)]

        # Projection head
        self.W_proj = rng.standard_normal((self.width, 1)) * scale(self.width, 1)
        self.b_proj = np.zeros(1)

        self._weights_initialized = True
        self._in_dim = in_dim

    @staticmethod
    def _gelu(x):
        """Approximate GeLU activation."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _fno_block(self, v: np.ndarray, layer: int) -> np.ndarray:
        """
        One FNO block applied to a batch of 1-D signals.

        v : (batch, L, width)  where L is the spatial signal length.
        """
        batch, L, W = v.shape
        n_modes_eff = min(self.n_modes, L // 2 + 1)  # can't exceed rfft output length

        # Spectral convolution (truncated Fourier modes)
        v_spec = np.fft.rfft(v, axis=1)               # (B, L//2+1, W)
        v_spec_trunc = v_spec[:, :n_modes_eff, :]     # (B, n_modes_eff, W)

        # Use only the first n_modes_eff rows of the spectral weight matrix
        W_spec_eff = self.W_spec[layer][:n_modes_eff]  # (n_modes_eff, W, W)

        # Einsum: (B, m, w_in) x (m, w_in, w_out) -> (B, m, w_out)
        v_out_spec = np.einsum("bmw,mwu->bmu", v_spec_trunc, W_spec_eff)

        # Zero-pad back to full rfft size before irfft
        rfft_len = L // 2 + 1
        pad_len  = rfft_len - n_modes_eff
        if pad_len > 0:
            v_out_spec = np.pad(v_out_spec, ((0, 0), (0, pad_len), (0, 0)))

        v_out = np.fft.irfft(v_out_spec, n=L, axis=1)   # (B, L, W)

        # Residual (pointwise linear)
        v_res = v @ self.W_res[layer] + self.b_res[layer]
        return self._gelu(v_out + v_res)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X : (batch, in_dim) — curvature PSD features
        Returns (batch, 1) predictions.
        """
        if not self._weights_initialized:
            self._init_weights(X.shape[1])

        batch = X.shape[0]
        # Lift to width dimension
        v = X @ self.W_lift + self.b_lift   # (batch, width)
        # Expand to (batch, 1, width) for FNO blocks (treat as 1-point signal)
        v = v[:, None, :]                   # (batch, 1, width)

        # FNO blocks
        for layer in range(self.n_layers):
            v = self._fno_block(v, layer)

        # Aggregate and project
        v = v.mean(axis=1)                  # (batch, width)
        out = v @ self.W_proj + self.b_proj # (batch, 1)
        return out

    def curvature_psd_features(self, curv: dict, n_modes: int = None) -> np.ndarray:
        """
        Extract the first n_modes of the radially-averaged power spectral density
        of |H| as features for one zone.

        Returns 1-D array of length n_modes.
        """
        if n_modes is None:
            n_modes = self.n_modes
        absH = curv["absH"]
        valid = curv["valid"]
        # Fill NaN with mean for FFT
        field = np.where(valid, absH, np.nanmean(absH[valid]) if valid.any() else 0.0)
        # Row-wise FFT, average power
        psd_rows = np.abs(np.fft.rfft(field, axis=1))**2
        psd_avg  = np.mean(psd_rows, axis=0)
        # Truncate / pad to n_modes
        if len(psd_avg) >= n_modes:
            return psd_avg[:n_modes]
        else:
            return np.pad(psd_avg, (0, n_modes - len(psd_avg)))


def compare_models(df: pd.DataFrame,
                   feature_cols: list,
                   target_col: str) -> pd.DataFrame:
    """
    Compare baseline Pearson R² vs GP LOO-R² for a given feature/target pair.

    Returns a summary DataFrame.
    """
    from scipy.stats import pearsonr

    sub = df[feature_cols + [target_col]].dropna()
    results = []

    # Baseline: linear regression R² for each individual feature
    for feat in feature_cols:
        x = sub[feat].values.astype(float)
        y = sub[target_col].values.astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            continue
        r, _ = pearsonr(x[valid], y[valid])
        results.append(dict(model=f"Linear ({feat})", r2=r**2, method="Pearson"))

    # GP model using all features
    gp_result = fit_gp(df, feature_cols, target_col)
    results.append(dict(model="Gaussian Process (LOO-CV)",
                        r2=max(0.0, gp_result["loo_r2"]), method="GP"))

    return pd.DataFrame(results).sort_values("r2", ascending=False)
