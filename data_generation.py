"""
Data Generation for Bivariate Spatial Simulations
===================================================
Python translation of Data_generation.R

Two simulation schemes:
  1. Non-Stationary: Wendland basis functions with random coefficients
  2. Non-Gaussian with covariates: Bivariate Matérn covariance + Tukey g-h transform

Each scheme produces 1200 spatial observations on [0,1]^2,
split into 1080 training / 120 testing points.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import kv as besselk, gamma as gammafn
from scipy.spatial.distance import cdist
from itertools import product as cartesian_product

# ──────────────────────────────────────────────────────────────
# Utility: Matérn covariance (matches geoR::matern)
# ──────────────────────────────────────────────────────────────

def matern(d, phi, nu):
    """
    Matérn covariance function.
    C(d) = [2^(1-nu) / Gamma(nu)] * (d/phi)^nu * K_nu(d/phi)
    with C(0) = 1.
    """
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    idx = d > 0
    scaled = d[idx] / phi
    out[idx] = (
        (2.0 ** (1.0 - nu)) / gammafn(nu)
        * (scaled ** nu)
        * besselk(nu, scaled)
    )
    out[~idx] = 1.0
    return out


# ──────────────────────────────────────────────────────────────
# Wendland compactly-supported basis (C6 Wendland)
# ──────────────────────────────────────────────────────────────

def wendland_basis(s, knots):
    """
    Wendland C6 basis:  (1-r)^6 * (35r^2 + 18r + 3) / 3  for 0 <= r <= 1
    Parameters
    ----------
    s : (N, 2) array of spatial locations
    knots : (K, 2) array of knot locations
    Returns
    -------
    phi : (N, K) basis matrix
    """
    N = s.shape[0]
    K = knots.shape[0]
    phi = np.zeros((N, K))
    for i in range(K):
        r = np.sqrt(np.sum((s - knots[i]) ** 2, axis=1))
        mask = (r >= 0) & (r <= 1)
        phi[mask, i] = (
            (1 - r[mask]) ** 6
            * (35 * r[mask] ** 2 + 18 * r[mask] + 3)
            / 3.0
        )
    return phi


# ══════════════════════════════════════════════════════════════
# SCHEME 1: Non-Stationary Simulation
# ══════════════════════════════════════════════════════════════

def generate_nonstationary(num_sim=50, seed=18, save=False, out_dir="non_stationary"):
    """
    Generate non-stationary bivariate spatial data using Wendland basis
    functions with randomly sampled polynomial coefficients.
    """
    rng = np.random.RandomState(seed)

    # --- grid and sampling ------------------------------------------------
    x = np.linspace(0, 1, 80)
    y = np.linspace(0, 1, 80)
    grid = np.array(list(cartesian_product(x, y)))       # 6400 × 2
    idx_all = rng.choice(6400, 1200, replace=False)
    s = grid[idx_all]
    N = s.shape[0]

    # --- multi-resolution Wendland basis ----------------------------------
    num_basis = [4, 9, 25]  # 2^2, 3^2, 5^2
    knots_1d = [np.linspace(0, 1, int(np.sqrt(k))) for k in num_basis]

    phi_list = []
    for res, nb in enumerate(num_basis):
        knots_2d = np.array(list(cartesian_product(knots_1d[res], knots_1d[res])))
        phi_list.append(wendland_basis(s, knots_2d))

    phi = np.hstack(phi_list)                             # (1200, 38)
    total_basis = phi.shape[1]

    if save:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "training_data"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "testing_data"), exist_ok=True)

    datasets = []
    coeff_range = np.linspace(-2.5, 2.5, 100)

    for sim in range(1, num_sim + 1):
        even_idx = np.arange(1, total_basis, 2)           # 0-based even columns
        odd_idx = np.arange(0, total_basis, 2)
        even_cols = phi[:, even_idx]
        odd_cols = phi[:, odd_idx]

        a, b, c, d_coef, e, f = rng.choice(coeff_range, 6)

        var1 = np.sum(
            a * np.abs(even_cols) ** 1.5 * np.sign(even_cols)
            + c * odd_cols
            - b * np.sqrt(np.abs(even_cols * odd_cols)) * np.sign(even_cols * odd_cols),
            axis=1,
        )
        var2 = np.sum(
            d_coef * even_cols
            - f * np.abs(odd_cols) ** 1.5 * np.sign(odd_cols),
            axis=1,
        )

        df = pd.DataFrame({"x": s[:, 0], "y": s[:, 1], "var1": var1, "var2": var2})

        # train / test split
        train_idx = rng.choice(1200, 1080, replace=False)
        test_idx = np.setdiff1d(np.arange(1200), train_idx)
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        datasets.append({"full": df, "train": df_train, "test": df_test})

        if save:
            df.to_csv(os.path.join(out_dir, f"2d_nonstationary_1200_{sim}.csv"), index=False)
            df_train.to_csv(os.path.join(out_dir, "training_data", f"2D_nonstationary_1200_{sim}-train.csv"), index=False)
            df_test.to_csv(os.path.join(out_dir, "testing_data", f"2D_nonstationary_1200_{sim}-test.csv"), index=False)

    return datasets, phi


# ══════════════════════════════════════════════════════════════
# SCHEME 2: Non-Gaussian with Covariates
# ══════════════════════════════════════════════════════════════

def generate_nongaussian(num_sim=50, seed=12345567, save=False, out_dir="non_gaussian"):
    """
    Generate non-Gaussian bivariate spatial data using:
      - Bivariate Matérn cross-covariance
      - Tukey g-h marginal transformation
      - Nonlinear covariate mean function
    """
    rng = np.random.RandomState(seed)

    # --- grid and sampling ------------------------------------------------
    x = np.linspace(0, 1, 80)
    y = np.linspace(0, 1, 80)
    grid = np.array(list(cartesian_product(x, y)))
    idx_all = rng.choice(6400, 1200, replace=False)
    X = grid[idx_all, 0]
    Y = grid[idx_all, 1]
    coords = np.column_stack([X, Y])
    m = cdist(coords, coords)                              # distance matrix

    # --- bivariate Matérn parameters --------------------------------------
    R = 0.5
    s11, s22 = 0.7, 0.8
    nu11, nu22 = 0.3, 0.6
    nu12 = (nu11 + nu22) / 2.0
    alpha11, alpha22 = 0.05, 0.1
    alpha12 = (alpha11 + alpha22) / 2.0

    constant = (
        np.sqrt(s11 * s22) * R
        * (gammafn(nu12) / np.sqrt(gammafn(nu11) * gammafn(nu22)))
        / (alpha12 ** nu12 / np.sqrt(alpha11 ** nu11 * alpha22 ** nu22))
    )

    matern_cov1 = s11 * matern(m, np.sqrt(alpha11), nu11)
    matern_cov2 = constant * matern(m, np.sqrt(alpha12), nu12)
    matern_cov4 = s22 * matern(m, np.sqrt(alpha22), nu22)

    full_matern_cov = np.block([
        [matern_cov1, matern_cov2],
        [matern_cov2.T, matern_cov4],
    ])
    # regularise for numerical stability
    full_matern_cov += np.eye(2400) * 1e-8

    # --- covariate covariance ---------------------------------------------
    a_cov, s_cov, nu_cov = 0.1, 0.9, 0.5
    cov_mat = s_cov * matern(m, a_cov, nu_cov)
    cov_mat += np.eye(1200) * 1e-8

    if save:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "training_data"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "testing_data"), exist_ok=True)

    coeff_range = np.linspace(-2.5, 2.5, 100)
    datasets = []

    for sim in range(1, num_sim + 1):
        # generate 5 spatial covariates
        covariates = []
        for _ in range(5):
            mu_val = rng.choice(coeff_range)
            cov_sample = rng.multivariate_normal(np.full(1200, mu_val), cov_mat)
            covariates.append(cov_sample)
        x1, x2, x3, x4, x5 = covariates

        # nonlinear mean function
        mean_field = (
            x1**2 - x2**2 + x3**2 - x4**2 - x5**2
            + 2*x1*x2 + 3*x2*x3 - 2*x3*x5 + 10*x1*x4
            + np.sin(x1)*x2*x3 + np.cos(x2)*x3*x5
            + x1*x2*x4*x5
        )

        # bivariate Matérn field
        simulation = rng.multivariate_normal(np.zeros(2400), full_matern_cov)
        var1_latent = simulation[:1200]
        var2_latent = simulation[1200:]

        # Tukey g-h transformation
        g1, h1 = 0.8, 0.5
        tukey_var1 = (np.exp(g1 * var1_latent) - 1) / g1 * np.exp(h1 * var1_latent**2 / 2) + mean_field

        g2, h2 = -0.8, 0.5
        tukey_var2 = (np.exp(g2 * var2_latent) - 1) / g2 * np.exp(h2 * var2_latent**2 / 2) + mean_field

        df = pd.DataFrame({
            "cov1": x1, "cov2": x2, "cov3": x3, "cov4": x4, "cov5": x5,
            "x": X, "y": Y,
            "var1": tukey_var1, "var2": tukey_var2,
        })

        train_idx = rng.choice(1200, 1080, replace=False)
        test_idx = np.setdiff1d(np.arange(1200), train_idx)
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        datasets.append({"full": df, "train": df_train, "test": df_test})

        if save:
            df.to_csv(os.path.join(out_dir, f"2d_nongaussian_1200_{sim}.csv"), index=False)
            df_train.to_csv(os.path.join(out_dir, "training_data", f"2D_nonGaussian_1200_{sim}-train.csv"), index=False)
            df_test.to_csv(os.path.join(out_dir, "testing_data", f"2D_nonGaussian_1200_{sim}-test.csv"), index=False)

    return datasets


# ══════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Non-Stationary datasets …")
    print("=" * 60)
    ns_data, phi = generate_nonstationary(num_sim=50, save=True, out_dir="non_stationary")
    print(f"  → {len(ns_data)} simulations saved.")

    print("=" * 60)
    print("Generating Non-Gaussian datasets …")
    print("=" * 60)
    ng_data = generate_nongaussian(num_sim=50, save=True, out_dir="non_gaussian")
    print(f"  → {len(ng_data)} simulations saved.")

    print("Done.")
