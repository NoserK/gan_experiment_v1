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
# Metrics — joint (vector) versions
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
# Per-variable metrics (for report_mode="per_var")
# ──────────────────────────────────────────────────────────────

def mse_1d(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mad_1d(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

def mahalanobis_distance_1d(y_true, y_pred):
    residuals = y_true - y_pred
    sd = max(np.std(residuals), 1e-12)
    return np.mean(np.abs(residuals) / sd)


# ──────────────────────────────────────────────────────────────
# Adaptive tail-prior utilities
# ──────────────────────────────────────────────────────────────
# These are used exclusively by the "cGAN-AP" method.
# The base "cGAN" always uses Gaussian noise.

def classify_tail_behavior(Y_train):
    """
    Classify the tail heaviness of the training targets and return
    the appropriate prior family string:

        "gaussian"  – Gaussian or sub-Gaussian tails  (κ ≤ 1.5)
        "laplace"   – super-Gaussian but sub-exponential tails  (1.5 < κ ≤ 6)
        "t"         – heavier-than-exponential tails  (κ > 6)

    Decision is based on the *excess kurtosis* (κ) of each column.
    The maximum κ across columns drives the choice so that the
    prior is at least as heavy-tailed as the heaviest marginal.
    """
    from scipy.stats import kurtosis as sp_kurtosis
    kappas = [sp_kurtosis(Y_train[:, c], fisher=True)
              for c in range(Y_train.shape[1])]
    kappa_max = np.max(kappas)

    if kappa_max <= 1.5:
        return "gaussian"
    elif kappa_max <= 6.0:
        return "laplace"
    else:
        return "t"


def _sample_noise(batch_size, noise_dim, device, prior_type="gaussian"):
    """
    Sample latent noise z according to the chosen prior family.

        "gaussian" → z ~ N(0, 1)
        "laplace"  → z ~ Laplace(0, 1/√2)   (unit variance)
        "t"        → z ~ t(df=5)             (heavy tails)
    """
    if prior_type == "gaussian":
        return torch.randn(batch_size, noise_dim, device=device)
    elif prior_type == "laplace":
        z = torch.distributions.Laplace(0.0, 1.0 / np.sqrt(2.0)).sample(
            (batch_size, noise_dim)
        )
        return z.to(device)
    else:   # "t"
        z_np = np.random.standard_t(
            df=5, size=(batch_size, noise_dim)
        ).astype(np.float32)
        return torch.from_numpy(z_np).to(device)


# ──────────────────────────────────────────────────────────────
# cGAN Architecture — vanilla BCE GAN (V4 baseline)
# ──────────────────────────────────────────────────────────────
# Both "cGAN" and "cGAN-AP" share the identical architecture,
# loss function, optimiser, and hyperparameters.  The ONLY
# difference is where the noise z is sampled from:
#   cGAN    → z ~ N(0,1)           (prior_type="gaussian")
#   cGAN-AP → z ~ adaptive prior   (prior_type from kurtosis)
# ──────────────────────────────────────────────────────────────

class Generator(nn.Module):
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
            nn.Linear(hidden // 2, 2),
        )
        self.noise_dim = noise_dim

    def forward(self, cond, z):
        return self.net(torch.cat([cond, z], dim=1))


class Discriminator(nn.Module):
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
               epochs=800, batch_size=128, lr=2e-4,
               lambda_recon=0.1,
               prior_type="gaussian"):
    """
    Train a conditional GAN and return the trained generator.

    Parameters
    ----------
    prior_type : str
        "gaussian" for the base cGAN, or "laplace"/"t" for cGAN-AP.
        Only affects how z is sampled; everything else stays identical.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(cond_dim, noise_dim, hidden).to(device)
    D = Discriminator(cond_dim, hidden).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    l1_loss = nn.L1Loss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for cond_batch, real_batch in loader:
            bs = cond_batch.size(0)
            cond_batch = cond_batch.to(device)
            real_batch = real_batch.to(device)

            real_label = torch.ones(bs, 1, device=device) * 0.9
            fake_label = torch.zeros(bs, 1, device=device) + 0.1

            # --- Discriminator step ---
            z = _sample_noise(bs, noise_dim, device, prior_type)
            fake = G(cond_batch, z).detach()

            loss_D = (
                criterion(D(cond_batch, real_batch), real_label)
                + criterion(D(cond_batch, fake), fake_label)
            ) / 2
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Generator step ---
            z = _sample_noise(bs, noise_dim, device, prior_type)
            fake = G(cond_batch, z)
            loss_adv = criterion(D(cond_batch, fake),
                                torch.ones(bs, 1, device=device))
            loss_rec = l1_loss(fake, real_batch)
            loss_G = loss_adv + lambda_recon * loss_rec
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

    G.eval()
    return G, device, noise_dim


def predict_cgan(G, X_test, device, noise_dim, n_samples=500,
                 prior_type="gaussian"):
    """
    Generate n_samples realisations for each test point.
    Returns (mean, std, full_samples).
    """
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = _sample_noise(X_t.size(0), noise_dim, device, prior_type)
            preds.append(G(X_t, z).cpu().numpy())
    preds = np.stack(preds, axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu, sigma, preds


# ──────────────────────────────────────────────────────────────
# Unified metric computation
# ──────────────────────────────────────────────────────────────

def _compute_method_metrics(Y_test, mu, sigma=None):
    """Compute both joint and per-variable metrics for one method."""
    m = {}

    # Joint metrics
    m["MSE"]   = mse(Y_test, mu)
    m["MAD"]   = mad(Y_test, mu)
    m["MD"]    = mahalanobis_distance(Y_test, mu)
    if sigma is not None:
        m["CRPS"]  = np.mean([
            crps_gaussian(Y_test[:, v], mu[:, v], sigma[:, v])
            for v in range(2)
        ])
        m["COV95"] = np.mean([
            coverage_95(Y_test[:, v], mu[:, v], sigma[:, v])
            for v in range(2)
        ])
    else:
        m["CRPS"]  = np.nan
        m["COV95"] = np.nan

    # Per-variable metrics
    for v, tag in enumerate(["var1", "var2"]):
        m[f"MSE_{tag}"]  = mse_1d(Y_test[:, v], mu[:, v])
        m[f"MAD_{tag}"]  = mad_1d(Y_test[:, v], mu[:, v])
        m[f"MD_{tag}"]   = mahalanobis_distance_1d(Y_test[:, v], mu[:, v])
        if sigma is not None:
            m[f"CRPS_{tag}"]  = crps_gaussian(Y_test[:, v], mu[:, v], sigma[:, v])
            m[f"COV95_{tag}"] = coverage_95(Y_test[:, v], mu[:, v], sigma[:, v])
        else:
            m[f"CRPS_{tag}"]  = np.nan
            m[f"COV95_{tag}"] = np.nan

    return m


# ──────────────────────────────────────────────────────────────
# Experiment runner  (4 methods)
# ──────────────────────────────────────────────────────────────

def run_experiment_on_dataset(df_train, df_test, feature_cols, target_cols,
                              scheme_name="", sim_id=0, verbose=False):
    """
    Fit four methods on df_train, predict on df_test, compute metrics.

    Methods
    -------
    1. cGAN      – vanilla BCE GAN + L1 recon, Gaussian noise (z ~ N(0,1))
    2. cGAN-AP   – same architecture, adaptive-prior noise
                   (z ~ Gaussian / Laplace / t  chosen by kurtosis)
    3. KNN       – distance-weighted k-nearest-neighbours
    4. RF        – Random Forest (separate per variable)

    The comparison between cGAN and cGAN-AP isolates the effect
    of the adaptive prior: architecture, loss, optimiser, and all
    hyper-parameters are identical.
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

    # ─── 1. cGAN (Gaussian prior) ────────────────────────────
    G, device, noise_dim = train_cgan(
        Xs_train, Ys_train, cond_dim=Xs_train.shape[1],
        noise_dim=32, hidden=128, epochs=800, batch_size=128,
        lambda_recon=0.1,
        prior_type="gaussian",
    )
    mu_s, sigma_s, _ = predict_cgan(G, Xs_test, device, noise_dim,
                                     n_samples=500, prior_type="gaussian")
    mu_cgan = scaler_y.inverse_transform(mu_s)
    sigma_cgan = sigma_s * scaler_y.scale_

    results["cGAN"] = _compute_method_metrics(Y_test, mu_cgan, sigma_cgan)

    # ─── 2. cGAN-AP (adaptive prior) ─────────────────────────
    #   Classify tail heaviness on the standardised targets,
    #   then train an otherwise-identical GAN with that prior.
    prior_type = classify_tail_behavior(Ys_train)
    prior_labels = {"gaussian": "N(0,1)",
                    "laplace": "Laplace(0,1/√2)",
                    "t": "t(df=5)"}
    if verbose:
        print(f"  [{scheme_name} sim {sim_id}] cGAN-AP prior → "
              f"{prior_labels[prior_type]}  "
              f"(kurtosis-based selection)")

    G_ap, dev_ap, nd_ap = train_cgan(
        Xs_train, Ys_train, cond_dim=Xs_train.shape[1],
        noise_dim=32, hidden=128, epochs=800, batch_size=128,
        lambda_recon=0.1,
        prior_type=prior_type,
    )
    mu_s_ap, sigma_s_ap, _ = predict_cgan(G_ap, Xs_test, dev_ap, nd_ap,
                                           n_samples=500,
                                           prior_type=prior_type)
    mu_cgan_ap = scaler_y.inverse_transform(mu_s_ap)
    sigma_cgan_ap = sigma_s_ap * scaler_y.scale_

    results["cGAN-AP"] = _compute_method_metrics(Y_test, mu_cgan_ap,
                                                  sigma_cgan_ap)

    # ─── 3. KNN ──────────────────────────────────────────────
    knn = KNeighborsRegressor(n_neighbors=10, weights="distance")
    knn.fit(Xs_train, Y_train)
    mu_knn = knn.predict(Xs_test)

    results["KNN"] = _compute_method_metrics(Y_test, mu_knn, sigma=None)

    # ─── 4. Random Forest ────────────────────────────────────
    rf1 = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    rf2 = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    rf1.fit(X_train, Y_train[:, 0])
    rf2.fit(X_train, Y_train[:, 1])
    mu_rf = np.column_stack([rf1.predict(X_test), rf2.predict(X_test)])

    tree_preds_1 = np.array([t.predict(X_test) for t in rf1.estimators_])
    tree_preds_2 = np.array([t.predict(X_test) for t in rf2.estimators_])
    sigma_rf = np.column_stack([tree_preds_1.std(axis=0),
                                tree_preds_2.std(axis=0)])

    results["RF"] = _compute_method_metrics(Y_test, mu_rf, sigma_rf)

    # ─── Verbose per-simulation output ───────────────────────
    if verbose:
        for method, m in results.items():
            print(f"  [{scheme_name} sim {sim_id}] {method:8s} | "
                  f"MSE={m['MSE']:.4f}  MAD={m['MAD']:.4f}  "
                  f"MD={m['MD']:.4f}  CRPS={m.get('CRPS', np.nan):.4f}  "
                  f"COV95={m.get('COV95', np.nan):.3f}")

    return results


# ──────────────────────────────────────────────────────────────
# Report-mode metric name sets
# ──────────────────────────────────────────────────────────────

JOINT_METRICS = ["MSE", "MAD", "MD", "CRPS", "COV95"]

PER_VAR_METRICS = [
    "MSE_var1",  "MSE_var2",
    "MAD_var1",  "MAD_var2",
    "MD_var1",   "MD_var2",
    "CRPS_var1", "CRPS_var2",
    "COV95_var1","COV95_var2",
]


def _print_summary(label, all_res, num_sim, methods, report_mode):
    """
    Print one summary table.

    Layout (both modes):
        rows    = metric names
        columns = Method_mean / Method_std  for each method
    """
    metric_names = PER_VAR_METRICS if report_mode == "per_var" else JOINT_METRICS

    print(f"\n{'─'*100}")
    print(f"Summary: {label}  ({num_sim} simulations)  "
          f"[report_mode={report_mode!r}]")
    print(f"{'─'*100}")

    rows = []
    for metric in metric_names:
        row = {"Metric": metric}
        for method in methods:
            vals = [v for v in all_res[method][metric] if not np.isnan(v)]
            if vals:
                row[f"{method}_mean"] = np.mean(vals)
                row[f"{method}_std"]  = np.std(vals)
            else:
                row[f"{method}_mean"] = np.nan
                row[f"{method}_std"]  = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%.4f"))


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main(report_mode="joint"):
    """
    Run the full experiment with four methods.

    Parameters
    ----------
    report_mode : str, {"joint", "per_var"}
        "joint"   – report MSE, MAD, MD, CRPS, COV95 as joint vector
                    quantities (original experiment.py format).
        "per_var" – report every metric separately for var1 and var2
                    (experiment_v3.py format).
    """
    assert report_mode in ("joint", "per_var"), \
        f"report_mode must be 'joint' or 'per_var', got {report_mode!r}"

    from data_generation import generate_nonstationary, generate_nongaussian

    num_sim = 50
    methods = ["cGAN", "cGAN-AP", "KNN", "RF"]

    all_metric_keys = JOINT_METRICS + PER_VAR_METRICS

    # ── Scheme 1: Non-Stationary ─────────────────────────────
    print("=" * 70)
    print("SCHEME 1: Non-Stationary Data")
    print("=" * 70)
    ns_data, _ = generate_nonstationary(num_sim=num_sim, seed=18, save=False)

    ns_results = {m: {k: [] for k in all_metric_keys} for m in methods}

    for i, ds in enumerate(ns_data):
        t0 = time.time()
        res = run_experiment_on_dataset(
            ds["train"], ds["test"],
            feature_cols=["x", "y"],
            target_cols=["var1", "var2"],
            scheme_name="NS", sim_id=i+1, verbose=True,
        )
        print(f"    (elapsed {time.time()-t0:.1f}s)")
        for method in methods:
            for metric in all_metric_keys:
                ns_results[method][metric].append(res[method][metric])

    # ── Scheme 2: Non-Gaussian ───────────────────────────────
    print("\n" + "=" * 70)
    print("SCHEME 2: Non-Gaussian with Covariates")
    print("=" * 70)
    ng_data = generate_nongaussian(num_sim=num_sim, seed=12345567, save=False)

    ng_results = {m: {k: [] for k in all_metric_keys} for m in methods}

    for i, ds in enumerate(ng_data):
        t0 = time.time()
        res = run_experiment_on_dataset(
            ds["train"], ds["test"],
            feature_cols=["x", "y", "cov1", "cov2", "cov3", "cov4", "cov5"],
            target_cols=["var1", "var2"],
            scheme_name="NG", sim_id=i+1, verbose=True,
        )
        print(f"    (elapsed {time.time()-t0:.1f}s)")
        for method in methods:
            for metric in all_metric_keys:
                ng_results[method][metric].append(res[method][metric])

    # ── Summary tables ───────────────────────────────────────
    _print_summary("Non-Stationary", ns_results, num_sim, methods, report_mode)
    _print_summary("Non-Gaussian",   ng_results, num_sim, methods, report_mode)

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-mode", choices=["joint", "per_var"], default="joint",
        help="'joint' for overall vector metrics, "
             "'per_var' for separate var1/var2 metrics (default: joint)",
    )
    args = parser.parse_args()
    main(report_mode=args.report_mode)
