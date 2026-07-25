"""
metrics_common.py
=================
Single source of truth for evaluation metrics, so CDE-AP and every baseline
are scored by *identical code*. These functions are copied verbatim (pure
NumPy) from the metrics in experiment_v9.py.

To guarantee one definition everywhere, you may optionally replace the
metrics()/metrics_conformal() bodies in experiment_v9.py with:

    from metrics_common import metrics, metrics_conformal

Joint-mode metrics (mode="joint"):
    MSE   = mean over points and coords of (mu - y)^2
    MAD   = mean over points and coords of |mu - y|
    MD    = mean over points of the per-point Euclidean error ||mu - y||_2
    CRPS  = multivariate energy score from predictive samples
    COV95 = joint coverage: fraction of points inside the rectangle on BOTH
            coordinates (from 2.5/97.5 sample percentiles, or from explicit
            lo/hi via metrics_conformal)
"""
import numpy as np


def metrics(samples, mean, y, mode="joint"):
    """
    samples : (M, n, 2) predictive samples, or None for point-only predictors
    mean    : (n, 2) point predictions
    y       : (n, 2) ground truth
    """
    o = {}
    if mode == "joint":
        o["MSE"] = ((mean - y) ** 2).mean()
        o["MAD"] = np.abs(mean - y).mean()
        o["MD"]  = np.linalg.norm(mean - y, axis=1).mean()
        if samples is not None:
            a = np.linalg.norm(samples - y[None], axis=-1).mean()
            b = 0.5 * np.linalg.norm(samples[:, None] - samples[None], axis=-1).mean()
            o["CRPS"] = a - b
            lo = np.percentile(samples, 2.5, 0)
            hi = np.percentile(samples, 97.5, 0)
            o["COV95"] = np.all((y >= lo) & (y <= hi), axis=1).mean()
    else:
        for j, n in enumerate(["v1", "v2"]):
            o[f"MSE_{n}"] = ((mean[:, j] - y[:, j]) ** 2).mean()
            o[f"MAD_{n}"] = np.abs(mean[:, j] - y[:, j]).mean()
            if samples is not None:
                lo = np.percentile(samples[:, :, j], 2.5, 0)
                hi = np.percentile(samples[:, :, j], 97.5, 0)
                o[f"COV95_{n}"] = ((y[:, j] >= lo) & (y[:, j] <= hi)).mean()
    return o


def metrics_conformal(samples, mean, y, lo, hi, mode="joint"):
    """Like metrics(), but COV95 is recomputed from EXPLICIT intervals lo/hi.
       Use for interval-native methods (CDE-AP conformal, GeoConformal)."""
    o = metrics(samples, mean, y, mode)
    if mode == "joint":
        o["COV95"] = np.all((y >= lo) & (y <= hi), axis=1).mean()
    else:
        for j, n in enumerate(["v1", "v2"]):
            o[f"COV95_{n}"] = ((y[:, j] >= lo[:, j]) & (y[:, j] <= hi[:, j])).mean()
    return o


def joint_interval_width(lo, hi):
    """Mean joint rectangle width = mean over points of sqrt(w1 * w2),
       the geometric mean of per-coordinate widths. Comparable across all
       interval methods."""
    w = hi - lo
    return float(np.sqrt(np.clip(w[:, 0], 0, None) * np.clip(w[:, 1], 0, None)).mean())
