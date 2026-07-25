"""
geoconformal_comparison.py
==========================
Run GeoConformal (GeoCP and GeoSIMCP) as a baseline against CDE-AP on the
ABCD settings, using the SAME PointNet point predictor so the only thing
being compared is the conformal/uncertainty layer.

Key design decisions (see notes in the accompanying message):
  * GeoConformal is UNIVARIATE -> we fit it once per output coordinate and
    combine into a joint rectangle, matching CDE-AP's joint region.
  * GeoConformal is a CALIBRATION WRAPPER, not a model -> base predictor is
    your PointNet (trained with plain MSE), identical to CDE-AP's Stage 1.
  * Domain is [0,1]^2 -> bandwidth grid is rescaled accordingly (the README's
    2.0/5.0 defaults oversmooth on a unit square and collapse GeoCP to plain
    split conformal).
  * miscoverage_level = 0.05 to match your 95% target (README default is 0.1).

Metrics reported per setting (means over replicates):
  MSE, MAD, MD                  -- point error (identical across CP methods;
                                   determined solely by the shared PointNet)
  COV95_joint                   -- fraction of test points inside BOTH coords
  COV95_v1, COV95_v2            -- per-coordinate coverage
  Width_joint                   -- geometric-mean joint interval width
                                   (sqrt(w1 * w2)), comparable to your
                                   joint_interval_width() in focused_ablation.py
  CRPS                          -- omitted: GeoConformal yields intervals, not
                                   predictive samples. Reported as NaN and must
                                   be left blank in the comparison table (do NOT
                                   fabricate a CRPS for an interval-only method).

Usage:
    python geoconformal_comparison.py setting_C 50
    python geoconformal_comparison.py setting_A 50 --geosimcp
"""
import sys, argparse, warnings
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from geoconformal import (
    GeoConformalSpatialRegression,
    GeoSIMConformalSpatialRegression,
)

# Reuse YOUR PointNet so the base predictor is identical to CDE-AP Stage 1.
from experiment_v9 import PointNet
# Shared scoring so coverage/width definitions match CDE-AP exactly.
from metrics_common import metrics_conformal, joint_interval_width


# ─── domain-appropriate tuning grids for [0,1]^2 ────────────────────────
BANDWIDTH_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
LAMBDA_GRID    = [0.0, 0.25, 0.5, 0.75, 1.0]   # GeoSIMCP only
ALPHA          = 0.05                           # 95% target (matches CDE-AP)


# ─── Stage-1 PointNet, trained exactly as in CDE-AP ─────────────────────
def train_pointnet(Xtr_s, Ytr, epochs=800, lr=1e-3, wd=1e-5,
                   batch_size=128, dev=None):
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PointNet(Xtr_s.shape[1]).to(dev)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32),
                                      torch.tensor(Ytr,   dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.mse_loss(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, dev


def interval_score(lo, hi, y, alpha=ALPHA):
    """Winkler / interval score: width + miscoverage penalty (for tuning)."""
    width = hi - lo
    penalty = (2.0 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return np.mean(width + penalty)


def fit_geocp_one_coord(predict_f, bandwidth,
                        coord_calib, coord_eval,
                        X_calib, y_calib_j, X_eval, y_eval_j,
                        use_simcp=False, lambda_weight=1.0):
    """Fit GeoCP/GeoSIMCP for a single output coordinate; return (lo, hi, cov)."""
    common = dict(
        predict_f=predict_f,
        miscoverage_level=ALPHA,
        bandwidth=bandwidth,
        coord_calib=coord_calib, coord_test=coord_eval,
        X_calib=X_calib, y_calib=y_calib_j,
        X_test=X_eval, y_test=y_eval_j,
    )
    if use_simcp:
        model = GeoSIMConformalSpatialRegression(
            lambda_weight=lambda_weight,
            distance_metric="euclidean",
            standardize_weights=True,
            **common,
        )
    else:
        model = GeoConformalSpatialRegression(**common)
    res = model.analyze()
    return res.lower_bound, res.upper_bound


def tune_and_eval(net, dev, Xtr_s, X_mean, X_std,
                  coords_calib, X_calib, Y_calib,
                  coords_val,   X_val,   Y_val,
                  coords_test,  X_test,  Y_test,
                  use_simcp=False):
    """Grid-search bandwidth (+lambda) on the VALIDATION slice per coordinate,
       then evaluate the joint rectangle on the TEST slice."""
    def predict_f(Xraw):
        Xs = (Xraw - X_mean) / X_std
        with torch.no_grad():
            mu = net(torch.tensor(Xs, dtype=torch.float32).to(dev)).cpu().numpy()
        return mu  # (n, 2)

    # Per-coordinate predict functions (GeoConformal is univariate)
    def predict_f_j(j):
        return lambda Xraw: predict_f(Xraw)[:, j]

    lo_te = np.zeros_like(Y_test)
    hi_te = np.zeros_like(Y_test)

    for j in range(2):
        best = (np.inf, None, None)   # (score, bw, lam)
        lam_grid = LAMBDA_GRID if use_simcp else [1.0]
        for bw in BANDWIDTH_GRID:
            for lam in lam_grid:
                try:
                    lo_v, hi_v = fit_geocp_one_coord(
                        predict_f_j(j), bw,
                        coords_calib, coords_val,
                        X_calib, Y_calib[:, j], X_val, Y_val[:, j],
                        use_simcp=use_simcp, lambda_weight=lam,
                    )
                except Exception as e:                       # numerical / sparse
                    warnings.warn(f"coord {j} bw={bw} lam={lam} failed: {e}")
                    continue
                cov_v = np.mean((Y_val[:, j] >= lo_v) & (Y_val[:, j] <= hi_v))
                if cov_v >= (1 - ALPHA):                     # keep valid coverage
                    s = interval_score(lo_v, hi_v, Y_val[:, j])
                    if s < best[0]:
                        best = (s, bw, lam)
        # Fallback if nothing reached nominal coverage on val: widest bandwidth
        bw_star  = best[1] if best[1] is not None else BANDWIDTH_GRID[-1]
        lam_star = best[2] if best[2] is not None else 1.0

        lo_te[:, j], hi_te[:, j] = fit_geocp_one_coord(
            predict_f_j(j), bw_star,
            coords_calib, coords_test,
            X_calib, Y_calib[:, j], X_test, Y_test[:, j],
            use_simcp=use_simcp, lambda_weight=lam_star,
        )

    mu_te = predict_f(X_test)
    inside = (Y_test >= lo_te) & (Y_test <= hi_te)
    # Score through the shared module so MSE/MAD/MD/COV95 match CDE-AP exactly.
    # GeoConformal returns intervals, not samples -> no CRPS (left as NaN; the
    # comparison table must leave CRPS blank for this interval-only method).
    m = metrics_conformal(None, mu_te, Y_test, lo_te, hi_te, mode="joint")
    m.update(metrics_conformal(None, mu_te, Y_test, lo_te, hi_te, mode="marginal"))
    w = hi_te - lo_te
    m["Width_joint"] = joint_interval_width(lo_te, hi_te)
    m["Width_v1"]    = float(w[:, 0].mean())
    m["Width_v2"]    = float(w[:, 1].mean())
    m["CRPS"]        = np.nan   # interval-only method: leave blank in the table
    return m


def run(tag, n_sim, use_simcp):
    rows = []
    for i in range(1, n_sim + 1):
        tr = pd.read_csv(f"{tag}/training_data/2D_{tag}_1200_{i}-train.csv")
        te = pd.read_csv(f"{tag}/testing_data/2D_{tag}_1200_{i}-test.csv")
        cov_cols = [c for c in tr.columns if c.startswith("cov")]
        feat = ["x", "y"] + cov_cols
        coord_cols = ["x", "y"]

        Xtr_all = tr[feat].values
        Ytr_all = tr[["var1", "var2"]].values
        Ctr_all = tr[coord_cols].values

        # train / calib / val split inside the training set (test is given)
        rng = np.random.RandomState(i)
        idx = rng.permutation(len(Xtr_all))
        n_cal = int(0.15 * len(idx)); n_val = int(0.15 * len(idx))
        cal_idx = idx[:n_cal]
        val_idx = idx[n_cal:n_cal + n_val]
        fit_idx = idx[n_cal + n_val:]

        X_fit, Y_fit = Xtr_all[fit_idx], Ytr_all[fit_idx]
        X_cal, Y_cal, C_cal = Xtr_all[cal_idx], Ytr_all[cal_idx], Ctr_all[cal_idx]
        X_val, Y_val, C_val = Xtr_all[val_idx], Ytr_all[val_idx], Ctr_all[val_idx]
        X_te,  Y_te,  C_te  = te[feat].values, te[["var1","var2"]].values, te[coord_cols].values

        # standardize using the fit slice (same convention as CDE-AP)
        X_mean = X_fit.mean(0); X_std = X_fit.std(0) + 1e-6
        net, dev = train_pointnet((X_fit - X_mean) / X_std, Y_fit)

        m = tune_and_eval(net, dev, (X_fit - X_mean) / X_std, X_mean, X_std,
                          C_cal, X_cal, Y_cal,
                          C_val, X_val, Y_val,
                          C_te,  X_te,  Y_te,
                          use_simcp=use_simcp)
        m["sim"] = i
        m["model"] = "GeoSIMCP+PointNet" if use_simcp else "GeoCP+PointNet"
        rows.append(m)
        print(f"[{tag}] replicate {i}/{n_sim}  "
            #   f"COV95={m['COV95_joint']:.3f}  Width={m['Width_joint']:.3f}")
            f"COV95={m['COV95']:.3f}  Width={m['Width_joint']:.3f}")

    df = pd.DataFrame(rows)
    summ = df.drop(columns=["sim"]).groupby("model").mean(numeric_only=True).round(4)
    print(f"\n=== {tag}  GeoConformal baseline (means over {n_sim} reps) ===")
    print(summ.to_string())
    out = f"geocp_{tag}{'_simcp' if use_simcp else ''}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="setting_C")
    ap.add_argument("n_sim", nargs="?", type=int, default=50)
    ap.add_argument("--geosimcp", action="store_true",
                    help="use GeoSIMCP (geo+feature weighting) instead of GeoCP")
    args = ap.parse_args()
    run(args.tag, args.n_sim, args.geosimcp)
