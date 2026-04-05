import os, warnings, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def mse(y_true, y_pred):
    """Per-variable MSE, then averaged over variables."""
    return np.mean((y_true - y_pred) ** 2)

def mad(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

def mahalanobis_distance(y_true, y_pred):
    """
    Average Mahalanobis distance across test points.
    Uses the empirical covariance of the residuals.
    """
    residuals = y_true - y_pred
    cov = np.cov(residuals.T)
    if np.linalg.matrix_rank(cov) < cov.shape[0]:
        cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    md_vals = np.array([
        mahalanobis(y_true[i], y_pred[i], cov_inv) for i in range(len(y_true))
    ])
    return np.mean(md_vals)

def crps_gaussian(y_true, mu, sigma):
    """
    CRPS for Gaussian predictive distribution.
    CRPS = sigma * [z*Φ(z) + φ(z) - 1/√π]  where z = (y - mu)/sigma
    """
    from scipy.stats import norm
    z = (y_true - mu) / (sigma + 1e-10)
    crps = sigma * (z * norm.cdf(z) + norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return np.mean(crps)

def coverage_95(y_true, mu, sigma):
    """Fraction of true values within ±1.96σ of the predicted mean."""
    lower = mu - 1.96 * sigma
    upper = mu + 1.96 * sigma
    return np.mean((y_true >= lower) & (y_true <= upper))


# ──────────────────────────────────────────────────────────────
# cGAN Architecture
# ──────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Generator: conditions on (x, y [, covariates]) plus noise z,
    and outputs predicted (var1, var2).
    """
    def __init__(self, cond_dim, noise_dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + noise_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 2),     # output: (var1, var2)
        )
        self.noise_dim = noise_dim

    def forward(self, cond, z):
        return self.net(torch.cat([cond, z], dim=1))


class Discriminator(nn.Module):
    """
    Discriminator: takes (condition, var1, var2) → real/fake score.
    """
    def __init__(self, cond_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, cond, y):
        return self.net(torch.cat([cond, y], dim=1))


def train_cgan(X_train, Y_train, cond_dim,
               noise_dim=32, hidden=128,
               epochs=600, batch_size=128, lr=2e-4):
    """Train a conditional GAN and return the trained generator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(cond_dim, noise_dim, hidden).to(device)
    D = Discriminator(cond_dim, hidden).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for cond_batch, real_batch in loader:
            bs = cond_batch.size(0)
            cond_batch = cond_batch.to(device)
            real_batch = real_batch.to(device)

            real_label = torch.ones(bs, 1, device=device) * 0.9   # label smoothing
            fake_label = torch.zeros(bs, 1, device=device) + 0.1

            # --- Discriminator step ---
            z = torch.randn(bs, noise_dim, device=device)
            fake = G(cond_batch, z).detach()

            loss_D = (
                criterion(D(cond_batch, real_batch), real_label)
                + criterion(D(cond_batch, fake), fake_label)
            ) / 2
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Generator step ---
            z = torch.randn(bs, noise_dim, device=device)
            fake = G(cond_batch, z)
            loss_G = criterion(D(cond_batch, fake), torch.ones(bs, 1, device=device))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

    G.eval()
    return G, device, noise_dim


def predict_cgan(G, X_test, device, noise_dim, n_samples=200):
    """
    Generate n_samples realisations for each test point, then
    return the mean prediction, std, and the full sample matrix.
    """
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(X_t.size(0), noise_dim, device=device)
            preds.append(G(X_t, z).cpu().numpy())
    preds = np.stack(preds, axis=0)          # (n_samples, N_test, 2)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu, sigma, preds


# ──────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment_on_dataset(df_train, df_test, feature_cols, target_cols,
                              scheme_name="", sim_id=0, verbose=False):
    """
    Fit three methods on df_train, predict on df_test, compute metrics.
    Returns a dict of metric values per method.
    """
    X_train = df_train[feature_cols].values
    Y_train = df_train[target_cols].values
    X_test = df_test[feature_cols].values
    Y_test = df_test[target_cols].values

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(Y_train)
    Xs_train = scaler_x.transform(X_train)
    Xs_test = scaler_x.transform(X_test)
    Ys_train = scaler_y.transform(Y_train)

    results = {}

    # ─── 1. cGAN ─────────────────────────────────────────────
    G, device, noise_dim = train_cgan(
        Xs_train, Ys_train, cond_dim=Xs_train.shape[1],
        noise_dim=32, hidden=128, epochs=600, batch_size=128,
    )
    mu_s, sigma_s, _ = predict_cgan(G, Xs_test, device, noise_dim, n_samples=200)
    mu_cgan = scaler_y.inverse_transform(mu_s)
    sigma_cgan = sigma_s * scaler_y.scale_                    # rescale std

    results["cGAN"] = {
        "MSE":   mse(Y_test, mu_cgan),
        "MAD":   mad(Y_test, mu_cgan),
        "MD":    mahalanobis_distance(Y_test, mu_cgan),
        "CRPS":  np.mean([
            crps_gaussian(Y_test[:, v], mu_cgan[:, v], sigma_cgan[:, v])
            for v in range(2)
        ]),
        "COV95": np.mean([
            coverage_95(Y_test[:, v], mu_cgan[:, v], sigma_cgan[:, v])
            for v in range(2)
        ]),
    }

    # ─── 2. KNN ──────────────────────────────────────────────
    knn = KNeighborsRegressor(n_neighbors=10, weights="distance")
    knn.fit(Xs_train, Y_train)
    mu_knn = knn.predict(Xs_test)

    results["KNN"] = {
        "MSE":  mse(Y_test, mu_knn),
        "MAD":  mad(Y_test, mu_knn),
        "MD":   mahalanobis_distance(Y_test, mu_knn),
        "CRPS": np.nan,      # no distributional prediction
        "COV95": np.nan,
    }

    # ─── 3. Random Forest ────────────────────────────────────
    rf1 = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    rf2 = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    rf1.fit(X_train, Y_train[:, 0])
    rf2.fit(X_train, Y_train[:, 1])
    mu_rf = np.column_stack([rf1.predict(X_test), rf2.predict(X_test)])

    # build prediction intervals from tree variance
    tree_preds_1 = np.array([t.predict(X_test) for t in rf1.estimators_])
    tree_preds_2 = np.array([t.predict(X_test) for t in rf2.estimators_])
    sigma_rf = np.column_stack([tree_preds_1.std(axis=0), tree_preds_2.std(axis=0)])

    results["RF"] = {
        "MSE":  mse(Y_test, mu_rf),
        "MAD":  mad(Y_test, mu_rf),
        "MD":   mahalanobis_distance(Y_test, mu_rf),
        "CRPS": np.mean([
            crps_gaussian(Y_test[:, v], mu_rf[:, v], sigma_rf[:, v])
            for v in range(2)
        ]),
        "COV95": np.mean([
            coverage_95(Y_test[:, v], mu_rf[:, v], sigma_rf[:, v])
            for v in range(2)
        ]),
    }

    if verbose:
        for method, m in results.items():
            print(f"  [{scheme_name} sim {sim_id}] {method:6s} | "
                  f"MSE={m['MSE']:.4f}  MAD={m['MAD']:.4f}  "
                  f"MD={m['MD']:.4f}  CRPS={m.get('CRPS', np.nan):.4f}  "
                  f"COV95={m.get('COV95', np.nan):.3f}")

    return results


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    from data_generation import generate_nonstationary, generate_nongaussian

    num_sim = 50           # number of Monte-Carlo replications (set to 50 for full run)

    # ── Scheme 1: Non-Stationary ─────────────────────────────
    print("=" * 70)
    print("SCHEME 1: Non-Stationary Data")
    print("=" * 70)
    ns_data, _ = generate_nonstationary(num_sim=num_sim, seed=18, save=False)

    ns_results = {m: {k: [] for k in ["MSE","MAD","MD","CRPS","COV95"]}
                  for m in ["cGAN","KNN","RF"]}

    for i, ds in enumerate(ns_data):
        t0 = time.time()
        res = run_experiment_on_dataset(
            ds["train"], ds["test"],
            feature_cols=["x", "y"],
            target_cols=["var1", "var2"],
            scheme_name="NS", sim_id=i+1, verbose=True,
        )
        print(f"    (elapsed {time.time()-t0:.1f}s)")
        for method in ns_results:
            for metric in ns_results[method]:
                ns_results[method][metric].append(res[method][metric])

    # ── Scheme 2: Non-Gaussian ───────────────────────────────
    print("\n" + "=" * 70)
    print("SCHEME 2: Non-Gaussian with Covariates")
    print("=" * 70)
    ng_data = generate_nongaussian(num_sim=num_sim, seed=12345567, save=False)

    ng_results = {m: {k: [] for k in ["MSE","MAD","MD","CRPS","COV95"]}
                  for m in ["cGAN","KNN","RF"]}

    for i, ds in enumerate(ng_data):
        t0 = time.time()
        res = run_experiment_on_dataset(
            ds["train"], ds["test"],
            feature_cols=["x", "y", "cov1", "cov2", "cov3", "cov4", "cov5"],
            target_cols=["var1", "var2"],
            scheme_name="NG", sim_id=i+1, verbose=True,
        )
        print(f"    (elapsed {time.time()-t0:.1f}s)")
        for method in ng_results:
            for metric in ng_results[method]:
                ng_results[method][metric].append(res[method][metric])

    # ── Summary tables ───────────────────────────────────────
    for label, all_res in [("Non-Stationary", ns_results), ("Non-Gaussian", ng_results)]:
        print(f"\n{'─'*70}")
        print(f"Summary: {label}  ({num_sim} simulations)")
        print(f"{'─'*70}")
        rows = []
        for method in ["cGAN", "KNN", "RF"]:
            row = {"Method": method}
            for metric in ["MSE", "MAD", "MD", "CRPS", "COV95"]:
                vals = [v for v in all_res[method][metric] if not np.isnan(v)]
                if vals:
                    row[f"{metric}_mean"] = np.mean(vals)
                    row[f"{metric}_std"] = np.std(vals)
                else:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan
            rows.append(row)

        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False, float_format="%.4f"))

    print("\nDone.")


if __name__ == "__main__":
    main()
