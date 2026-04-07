#!/usr/bin/env python
# coding: utf-8

"""
Bivariate DeepKriging — Nonstationary Data
Dual MSE Comparison: Standard MSE vs Variance-Weighted MSE

Based on original code by Pratik (2023), modified to compare two loss functions.
BUG FIX: str(1) → str(sim+1) on data loading line.
Extended metrics: MSE, MAD, CRPS, Mahalanobis Distance, 95% Coverage.
Compatibility: Keras 3 / TensorFlow 2.16+
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Input
from keras.callbacks import EarlyStopping
import keras.ops as ops          # Keras 3 replacement for keras.backend
import numpy as np
import time
from scipy import stats
from sklearn.model_selection import train_test_split
import sys

num_sim = 50  # int(sys.argv[1])

# ─────────────────────── Metric Functions ───────────────────────
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def mad(y_pred, y_true):
    """Median Absolute Deviation — robust to outliers."""
    return float(np.median(np.abs(y_pred - y_true)))

def crps_gaussian(y_true, mu, sigma):
    """
    Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma^2).

    CRPS(F, y) = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.

    Lower is better (sharper + better-calibrated forecast).
    """
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.clip(sigma, 1e-12, None)
    z = (np.asarray(y_true) - np.asarray(mu)) / sigma
    crps_per_obs = sigma * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_per_obs))

def estimate_sigma_from_residuals(y_pred_train, y_true_train):
    """Estimate a constant sigma (RMSE) from training residuals."""
    residuals = np.asarray(y_pred_train) - np.asarray(y_true_train)
    return float(np.sqrt(np.mean(residuals ** 2)))

def mahalanobis_distance(y_pred, y_true):
    """
    Mean Mahalanobis distance for bivariate predictions.

    Uses the residual covariance matrix to capture cross-variable
    correlation in the errors.
    """
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    cov_matrix = np.cov(residuals, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)
    left = residuals @ cov_inv
    d_sq = np.sum(left * residuals, axis=1)
    d = np.sqrt(np.clip(d_sq, 0, None))
    return {
        'mean':   float(np.mean(d)),
        'median': float(np.median(d)),
    }

def coverage_95(y_true, mu, sigma, alpha=0.05):
    """
    Empirical coverage and width of a (1-alpha) Gaussian prediction interval.

    Ideal coverage = 1 - alpha = 0.95.
    """
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lower = mu - z * sigma
    upper = mu + z * sigma
    y = np.asarray(y_true, dtype=float)
    inside = (y >= lower) & (y <= upper)
    return {
        'coverage':   float(np.mean(inside)),
        'mean_width': float(np.mean(upper - lower)),
    }

def compute_all_metrics(y_pred, y_true, sigma_var1, sigma_var2):
    """Compute every metric for a single simulation run."""
    p1, t1 = y_pred[:, 0], y_true[:, 0]
    p2, t2 = y_pred[:, 1], y_true[:, 1]

    mah = mahalanobis_distance(y_pred, y_true)
    cov1 = coverage_95(t1, p1, sigma_var1)
    cov2 = coverage_95(t2, p2, sigma_var2)

    return {
        'mse_var1':            float(np.mean((p1 - t1) ** 2)),
        'mse_var2':            float(np.mean((p2 - t2) ** 2)),
        'mad_var1':            mad(p1, t1),
        'mad_var2':            mad(p2, t2),
        'crps_var1':           crps_gaussian(t1, p1, sigma_var1),
        'crps_var2':           crps_gaussian(t2, p2, sigma_var2),
        'mahal_mean':          mah['mean'],
        'mahal_median':        mah['median'],
        'coverage_var1':       cov1['coverage'],
        'coverage_var2':       cov2['coverage'],
        'interval_width_var1': cov1['mean_width'],
        'interval_width_var2': cov2['mean_width'],
    }


# ─────────────────────── Model Builder ──────────────────────────
def build_network(input_dim):
    """Construct the DeepKriging architecture."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(100, kernel_initializer='he_uniform', activation='linear'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(100, activation='linear'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(100, activation='linear'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(100, activation='linear'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(100, activation='linear'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(50, activation='linear'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(50, activation='linear'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(2, activation='linear'))
    return model

# ─────────────────────── Training Function ──────────────────────
def train_model(phi, y, loss_fn, sim_iteration, loss_name):
    """Train model with the specified loss function."""
    model = build_network(phi.shape[1])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae', 'mse'])

    print(f'  [{loss_name}] Fitting DNN for simulation {sim_iteration + 1}')
    # Phase 1: warm-up
    model.fit(phi, y,
              validation_split=0.1, epochs=500, batch_size=256, verbose=0)

    # Phase 2: fine-tune with early stopping
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    ]
    model.fit(phi, y, callbacks=callbacks,
              validation_split=0.1, epochs=1000, batch_size=256, verbose=0)
    return model


# ─────────────────────── Summary Printer ────────────────────────
def print_full_summary(results_std, results_wt, num_sim):
    """Print a comparison table of ALL metrics across simulations."""
    df_s = pd.DataFrame(results_std)
    df_w = pd.DataFrame(results_wt)

    metrics = [
        ('mse_var1',            'MSE (var1)'),
        ('mse_var2',            'MSE (var2)'),
        ('mad_var1',            'MAD (var1)'),
        ('mad_var2',            'MAD (var2)'),
        ('crps_var1',           'CRPS (var1)'),
        ('crps_var2',           'CRPS (var2)'),
        ('mahal_mean',          'Mahalanobis (mean)'),
        ('mahal_median',        'Mahalanobis (median)'),
        ('coverage_var1',       '95% Coverage (var1)'),
        ('coverage_var2',       '95% Coverage (var2)'),
        ('interval_width_var1', 'Interval Width (var1)'),
        ('interval_width_var2', 'Interval Width (var2)'),
    ]

    print()
    print("=" * 82)
    print(f"  FULL METRIC SUMMARY — NONSTATIONARY  ({num_sim} simulations)")
    print("=" * 82)
    header = (f"{'Metric':<28} {'Std-MSE Mean':>14} {'Std-MSE Std':>12} "
              f"{'Wt-MSE Mean':>14} {'Wt-MSE Std':>12}")
    print(header)
    print("-" * 82)
    for col, label in metrics:
        sm = df_s[col].mean()
        ss = df_s[col].std()
        wm = df_w[col].mean()
        ws = df_w[col].std()
        print(f"{label:<28} {sm:>14.6f} {ss:>12.6f} {wm:>14.6f} {ws:>12.6f}")
    print("=" * 82)
    print()


# ─────────────────────── Main ───────────────────────────────────
def main():
    print()
    print("#" * 70)
    print("# NONSTATIONARY DATA: Standard MSE vs Weighted MSE Comparison")
    print("#" * 70)
    print()

    base_path = "~/Documents/indicator_deep_kriging/Bivariate_DeepKriging-main"

    # Load precomputed basis functions
    phi = pd.read_csv(f"{base_path}/src/python_scripts/phi.csv", sep=",")

    # Storage: list of dicts, one per simulation
    results_standard = []
    results_weighted = []

    for sim in range(num_sim):
        print(f'\n{"="*60}')
        print(f'  Simulation {sim + 1}/{num_sim}')
        print(f'{"="*60}')

        # ── Load data ──
        df_loc = pd.read_csv(
            f"{base_path}/synthetic_data_simulations_non-Stationary/"
            f"2d_nonstationary_1200_{sim+1}.csv", sep=",")

        df_train, df_test, phi_train, phi_test = train_test_split(
            df_loc, pd.DataFrame.to_numpy(phi),
            test_size=0.1, random_state=123)
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        # ── Compute statistics from original training data ──
        variance_var1 = np.var(df_train["var1"])
        variance_var2 = np.var(df_train["var2"])
        mean1 = np.mean(df_train["var1"])
        mean2 = np.mean(df_train["var2"])

        # ── Standardize training targets ──
        df_train_norm = df_train.copy()
        df_train_norm["var1"] = (df_train["var1"] - mean1) / np.sqrt(variance_var1)
        df_train_norm["var2"] = (df_train["var2"] - mean2) / np.sqrt(variance_var2)
        y_train = np.array(df_train_norm[["var1", "var2"]])

        y_test = np.array(df_test[["var1", "var2"]])

        # ── Original training targets (un-standardized) for sigma estimation ──
        y_train_orig = np.column_stack([
            df_train["var1"].values,
            df_train["var2"].values
        ])

        # ── Define custom weighted MSE (Keras 3 compatible) ──
        vv1 = float(variance_var1)
        vv2 = float(variance_var2)

        def custom_mse(y_true, y_pred):
            loss = ops.square(y_pred - y_true)
            loss = loss * [1.0 / vv1, 1.0 / vv2]
            loss = ops.sum(loss, axis=1)
            return loss

        # ══════════════════════════════════════════════════════════
        # STANDARD MSE
        # ══════════════════════════════════════════════════════════
        model_std = train_model(phi_train, y_train, 'mse', sim, 'standard')

        # Test predictions (de-standardized)
        y_pred_std = model_std.predict(phi_test, verbose=0)
        y_pred_std[:, 0] = y_pred_std[:, 0] * np.sqrt(variance_var1) + mean1
        y_pred_std[:, 1] = y_pred_std[:, 1] * np.sqrt(variance_var2) + mean2

        # Training predictions (de-standardized) → estimate sigma for CRPS & CI
        y_pred_train_std = model_std.predict(phi_train, verbose=0)
        y_pred_train_std[:, 0] = y_pred_train_std[:, 0] * np.sqrt(variance_var1) + mean1
        y_pred_train_std[:, 1] = y_pred_train_std[:, 1] * np.sqrt(variance_var2) + mean2
        sigma1_std = estimate_sigma_from_residuals(y_pred_train_std[:, 0], y_train_orig[:, 0])
        sigma2_std = estimate_sigma_from_residuals(y_pred_train_std[:, 1], y_train_orig[:, 1])

        # Compute all metrics
        metrics_std = compute_all_metrics(y_pred_std, y_test, sigma1_std, sigma2_std)
        results_standard.append(metrics_std)

        # ══════════════════════════════════════════════════════════
        # WEIGHTED MSE
        # ══════════════════════════════════════════════════════════
        model_wt = train_model(phi_train, y_train, custom_mse, sim, 'weighted')

        # Test predictions (de-standardized)
        y_pred_wt = model_wt.predict(phi_test, verbose=0)
        y_pred_wt[:, 0] = y_pred_wt[:, 0] * np.sqrt(variance_var1) + mean1
        y_pred_wt[:, 1] = y_pred_wt[:, 1] * np.sqrt(variance_var2) + mean2

        # Training predictions (de-standardized) → estimate sigma
        y_pred_train_wt = model_wt.predict(phi_train, verbose=0)
        y_pred_train_wt[:, 0] = y_pred_train_wt[:, 0] * np.sqrt(variance_var1) + mean1
        y_pred_train_wt[:, 1] = y_pred_train_wt[:, 1] * np.sqrt(variance_var2) + mean2
        sigma1_wt = estimate_sigma_from_residuals(y_pred_train_wt[:, 0], y_train_orig[:, 0])
        sigma2_wt = estimate_sigma_from_residuals(y_pred_train_wt[:, 1], y_train_orig[:, 1])

        # Compute all metrics
        metrics_wt = compute_all_metrics(y_pred_wt, y_test, sigma1_wt, sigma2_wt)
        results_weighted.append(metrics_wt)

        # ── Per-simulation print ──
        print(f"  Standard -> MSE v1: {metrics_std['mse_var1']:.6f}, "
              f"v2: {metrics_std['mse_var2']:.6f} | "
              f"MAD v1: {metrics_std['mad_var1']:.6f}, v2: {metrics_std['mad_var2']:.6f} | "
              f"CRPS v1: {metrics_std['crps_var1']:.6f}, v2: {metrics_std['crps_var2']:.6f}")
        print(f"  Weighted -> MSE v1: {metrics_wt['mse_var1']:.6f}, "
              f"v2: {metrics_wt['mse_var2']:.6f} | "
              f"MAD v1: {metrics_wt['mad_var1']:.6f}, v2: {metrics_wt['mad_var2']:.6f} | "
              f"CRPS v1: {metrics_wt['crps_var1']:.6f}, v2: {metrics_wt['crps_var2']:.6f}")
        print(f"           Mahal(std): {metrics_std['mahal_mean']:.4f}, "
              f"Mahal(wt): {metrics_wt['mahal_mean']:.4f} | "
              f"Cov95 std: ({metrics_std['coverage_var1']:.3f}, {metrics_std['coverage_var2']:.3f}), "
              f"wt: ({metrics_wt['coverage_var1']:.3f}, {metrics_wt['coverage_var2']:.3f})")

    # ── Save results ──
    save_path = f"{base_path}/plot_results"
    os.makedirs(os.path.expanduser(save_path), exist_ok=True)

    df_std = pd.DataFrame(results_standard)
    df_wt  = pd.DataFrame(results_weighted)
    df_std.to_csv(f"{save_path}/DeepKriging_nonstationary_metrics_standard.csv", index=False)
    df_wt.to_csv(f"{save_path}/DeepKriging_nonstationary_metrics_weighted.csv", index=False)

    # ── Full summary ──
    print_full_summary(results_standard, results_weighted, num_sim)


if __name__ == '__main__':
    main()
