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
from torch.nn.utils import spectral_norm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Metrics  (1-d versions for per-variable reporting)
# ──────────────────────────────────────────────────────────────

def mse_1d(y_true, y_pred):
    """MSE for a single variable (1-d arrays)."""
    return np.mean((y_true - y_pred) ** 2)

def mad_1d(y_true, y_pred):
    """MAD for a single variable (1-d arrays)."""
    return np.median(np.abs(y_true - y_pred))

def mahalanobis_distance_1d(y_true, y_pred):
    """
    Average Mahalanobis distance for a single variable.
    In 1-d this reduces to mean(|residual| / std(residual)).
    """
    residuals = y_true - y_pred
    sd = np.std(residuals)
    if sd < 1e-12:
        sd = 1e-12
    return np.mean(np.abs(residuals) / sd)

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
# Tail-behaviour classification  (Modification 3, revised)
# ──────────────────────────────────────────────────────────────

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

    The Gaussian threshold is set at κ = 1.5 (midpoint between
    Gaussian κ=0 and Laplace κ=3) rather than 0, because finite-
    sample kurtosis estimates fluctuate: truly Gaussian data with
    n ≈ 500–2000 routinely yields κ up to ~0.5–1.0.

    Reference excess-kurtosis values:
        Uniform   −1.2   (sub-Gaussian)
        Gaussian   0
        Laplace    3
        Exponential 6
        t(5)       6     (boundary)
        t(3)      ∞      (heavy-tailed)
    """
    from scipy.stats import kurtosis as sp_kurtosis

    # Fisher=True gives *excess* kurtosis (Gaussian → 0)
    kappas = [sp_kurtosis(Y_train[:, c], fisher=True)
              for c in range(Y_train.shape[1])]
    kappa_max = np.max(kappas)

    if kappa_max <= 1.5:
        return "gaussian"
    elif kappa_max <= 6.0:
        return "laplace"
    else:
        return "t"


# ──────────────────────────────────────────────────────────────
# cGAN Architecture  (WGAN-GP + Spectral Norm + larger G)
# ──────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Generator: conditions on (x, y [, covariates]) plus noise z,
    and outputs predicted (var1, var2).
    Hidden size increased to 256 (Modification 4).
    Spectral normalisation applied here (Modification 2) to control
    the Lipschitz constant globally and stabilise training.
    """
    def __init__(self, cond_dim, noise_dim=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(cond_dim + noise_dim, hidden)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            spectral_norm(nn.Linear(hidden, hidden // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden // 2, 2)),     # output: (var1, var2)
        )
        self.noise_dim = noise_dim

    def forward(self, cond, z):
        return self.net(torch.cat([cond, z], dim=1))


class Critic(nn.Module):
    """
    WGAN-GP Critic (was Discriminator).
    - Sigmoid removed (Modification 1).
    - Dropout removed (incompatible with WGAN-GP).

    Note: spectral_norm is NOT applied here because the gradient
    penalty already enforces the 1-Lipschitz constraint.  Combining
    both over-constrains the critic, starving the generator of useful
    gradients.  Spectral normalisation is instead applied in the
    *Generator* (see below) where it stabilises training without
    conflicting with GP.
    """
    def __init__(self, cond_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
            # No Sigmoid — raw Wasserstein score
        )

    def forward(self, cond, y):
        return self.net(torch.cat([cond, y], dim=1))


def _gradient_penalty(critic, cond, real, fake, device, lambda_gp=10.0):
    """
    Compute the gradient penalty for WGAN-GP.
    Interpolate between real and fake samples, then penalise
    the critic gradient's deviation from unit norm.
    """
    alpha = torch.rand(real.size(0), 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(cond, interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


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
        # Laplace(0, b) has variance 2b².  Set b = 1/√2 so that Var = 1.
        z = torch.distributions.Laplace(0.0, 1.0 / np.sqrt(2.0)).sample(
            (batch_size, noise_dim)
        )
        return z.to(device)
    else:  # "t"
        z_np = np.random.standard_t(df=5,
                                     size=(batch_size, noise_dim)).astype(np.float32)
        return torch.from_numpy(z_np).to(device)


def train_cgan(X_train, Y_train, cond_dim,
               noise_dim=32, hidden=256,
               epochs=600, batch_size=128, lr=2e-4,
               n_critic=5, lambda_gp=10.0,
               prior_type="gaussian"):
    """
    Train a conditional WGAN-GP and return the trained generator.

    Changes from original:
      - Wasserstein loss + gradient penalty  (Mod 1)
      - Spectral-normalised critic           (Mod 2)
      - Adaptive prior selection             (Mod 3)
      - Generator hidden=256                 (Mod 4)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(cond_dim, noise_dim, hidden).to(device)
    C = Critic(cond_dim, hidden).to(device)

    # Adam with betas recommended for WGAN-GP
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for cond_batch, real_batch in loader:
            bs = cond_batch.size(0)
            cond_batch = cond_batch.to(device)
            real_batch = real_batch.to(device)

            # ── Critic steps (n_critic per generator update) ──
            for _ in range(n_critic):
                z = _sample_noise(bs, noise_dim, device, prior_type)
                fake = G(cond_batch, z).detach()

                # Wasserstein loss: maximise  E[C(real)] - E[C(fake)]
                # ⟺ minimise  E[C(fake)] - E[C(real)] + GP
                loss_C = (
                    C(cond_batch, fake).mean()
                    - C(cond_batch, real_batch).mean()
                    + _gradient_penalty(C, cond_batch, real_batch, fake,
                                        device, lambda_gp)
                )
                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

            # ── Generator step ────────────────────────────────
            z = _sample_noise(bs, noise_dim, device, prior_type)
            fake = G(cond_batch, z)
            loss_G = -C(cond_batch, fake).mean()   # maximise critic score
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

    G.eval()
    return G, device, noise_dim, prior_type


def predict_cgan(G, X_test, device, noise_dim, n_samples=200,
                 prior_type="gaussian"):
    """
    Generate n_samples realisations for each test point, then
    return the mean prediction, std, and the full sample matrix.
    """
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = _sample_noise(X_t.size(0), noise_dim, device, prior_type)
            preds.append(G(X_t, z).cpu().numpy())
    preds = np.stack(preds, axis=0)          # (n_samples, N_test, 2)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu, sigma, preds


# ──────────────────────────────────────────────────────────────
# Per-variable metric helper
# ──────────────────────────────────────────────────────────────

METRIC_NAMES = [
    "MSE_var1", "MSE_var2",
    "MAD_var1", "MAD_var2",
    "MD_var1",  "MD_var2",
    "CRPS_var1","CRPS_var2",
    "COV95_var1","COV95_var2",
]


def _compute_metrics_per_var(Y_test, mu, sigma=None):
    """
    Compute all five metrics separately for var1 (col 0) and var2 (col 1).
    If sigma is None the distributional metrics (CRPS, COV95) are NaN.
    """
    m = {}
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
# Experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment_on_dataset(df_train, df_test, feature_cols, target_cols,
                              scheme_name="", sim_id=0, verbose=False):
    """
    Fit three methods on df_train, predict on df_test, compute metrics.
    Returns a dict of per-variable metric values per method.
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

    # ─── Tail-behaviour check for prior selection (Mod 3) ──────
    prior_type = classify_tail_behavior(Ys_train)
    prior_labels = {"gaussian": "Gaussian", "laplace": "Laplace(0, 1/√2)", "t": "t(df=5)"}
    if verbose:
        print(f"  [{scheme_name} sim {sim_id}] Prior selected: {prior_labels[prior_type]}")

    # ─── 1. cGAN (WGAN-GP) ──────────────────────────────────
    G, device, noise_dim, _pt = train_cgan(
        Xs_train, Ys_train, cond_dim=Xs_train.shape[1],
        noise_dim=32, hidden=256, epochs=600, batch_size=128,
        n_critic=5, lambda_gp=10.0,
        prior_type=prior_type,
    )
    mu_s, sigma_s, _ = predict_cgan(G, Xs_test, device, noise_dim,
                                     n_samples=200,
                                     prior_type=prior_type)
    mu_cgan = scaler_y.inverse_transform(mu_s)
    sigma_cgan = sigma_s * scaler_y.scale_

    results["cGAN"] = _compute_metrics_per_var(Y_test, mu_cgan, sigma_cgan)

    # ─── 2. KNN ──────────────────────────────────────────────
    knn = KNeighborsRegressor(n_neighbors=10, weights="distance")
    knn.fit(Xs_train, Y_train)
    mu_knn = knn.predict(Xs_test)

    results["KNN"] = _compute_metrics_per_var(Y_test, mu_knn, sigma=None)

    # ─── 3. Random Forest ────────────────────────────────────
    rf1 = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    rf2 = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    rf1.fit(X_train, Y_train[:, 0])
    rf2.fit(X_train, Y_train[:, 1])
    mu_rf = np.column_stack([rf1.predict(X_test), rf2.predict(X_test)])

    tree_preds_1 = np.array([t.predict(X_test) for t in rf1.estimators_])
    tree_preds_2 = np.array([t.predict(X_test) for t in rf2.estimators_])
    sigma_rf = np.column_stack([tree_preds_1.std(axis=0), tree_preds_2.std(axis=0)])

    results["RF"] = _compute_metrics_per_var(Y_test, mu_rf, sigma_rf)

    if verbose:
        for method, m in results.items():
            print(f"  [{scheme_name} sim {sim_id}] {method:6s} | "
                  f"MSE=({m['MSE_var1']:.4f},{m['MSE_var2']:.4f})  "
                  f"MAD=({m['MAD_var1']:.4f},{m['MAD_var2']:.4f})  "
                  f"MD=({m['MD_var1']:.4f},{m['MD_var2']:.4f})  "
                  f"CRPS=({m['CRPS_var1']:.4f},{m['CRPS_var2']:.4f})  "
                  f"COV95=({m['COV95_var1']:.3f},{m['COV95_var2']:.3f})")

    return results


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    from data_generation import generate_nonstationary, generate_nongaussian

    num_sim = 50           # number of Monte-Carlo replications

    # ── Scheme 1: Non-Stationary ─────────────────────────────
    print("=" * 70)
    print("SCHEME 1: Non-Stationary Data")
    print("=" * 70)
    ns_data, _ = generate_nonstationary(num_sim=num_sim, seed=18, save=False)

    ns_results = {m: {k: [] for k in METRIC_NAMES}
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
            for metric in METRIC_NAMES:
                ns_results[method][metric].append(res[method][metric])

    # ── Scheme 2: Non-Gaussian ───────────────────────────────
    print("\n" + "=" * 70)
    print("SCHEME 2: Non-Gaussian with Covariates")
    print("=" * 70)
    ng_data = generate_nongaussian(num_sim=num_sim, seed=12345567, save=False)

    ng_results = {m: {k: [] for k in METRIC_NAMES}
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
            for metric in METRIC_NAMES:
                ng_results[method][metric].append(res[method][metric])

    # ── Summary tables ───────────────────────────────────────
    for label, all_res in [("Non-Stationary", ns_results), ("Non-Gaussian", ng_results)]:
        print(f"\n{'─'*90}")
        print(f"Summary: {label}  ({num_sim} simulations)")
        print(f"{'─'*90}")
        rows = []
        for method in ["cGAN", "KNN", "RF"]:
            row = {"Method": method}
            for metric in METRIC_NAMES:
                vals = [v for v in all_res[method][metric] if not np.isnan(v)]
                if vals:
                    row[f"{metric}_mean"] = np.mean(vals)
                    row[f"{metric}_std"]  = np.std(vals)
                else:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"]  = np.nan
            rows.append(row)

        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False, float_format="%.4f"))

    print("\nDone.")


if __name__ == "__main__":
    main()
