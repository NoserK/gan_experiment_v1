"""
focused_ablation.py — companion to experiment_v9.py

Isolates the contribution of the AP family-selection rule by holding the
PointNet, calibration split, and conformal procedure FIXED across four
density-head variants. Only the parametric family changes:

    CDE-Gauss      : Gaussian residual density, σ(x) head only
    CDE-Laplace    : Laplace residual density, σ(x) head only
    CDE-t (global) : Student-t residual density, σ(x) head + single global ν
    CDE-AP (t loc) : Student-t residual density, σ(x) head + ν(x) head
                     (this is what AP selects on heavy-tailed residuals)

Headline metrics (in order of importance for the story):
    1. Test NLL on calibration residuals (pure density fit, no conformal)
    2. Mean JOINT interval width after conformal calibration
    3. Joint and marginal COV95 (sanity check — should all be ≈0.95)
    4. Conditional COV95 by σ̂-decile (uneven coverage = wrong family)
    5. CRPS of parametric samples
    6. MSE / MAD / MD (should be ~identical — same PointNet)

Usage:
    python focused_ablation.py setting_B          # default
    python focused_ablation.py setting_B 50       # 50 replicates
"""
import sys, json
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Reuse architectures from experiment_v9.py
from experiment_v9 import (
    PointNet, classify_tail_behavior,
    _nll_gaussian, _nll_laplace, _nll_t, metrics_conformal
)


# ─── Density head with explicit family / ν-mode control ─────────────────
class DensityHead(nn.Module):
    """
    Same trunk as DensityNet in experiment_v9.py, but with an explicit
    `nu_mode` flag controlling the Student-t degrees-of-freedom:
        'none'   : not Student-t (family is gaussian or laplace)
        'global' : single trainable scalar ν, shared across all locations
        'local'  : per-location ν(x) head (this is CDE-AP's behaviour)
    """
    def __init__(self, in_dim, family, nu_mode="none", hidden=128, n_layers=4):
        super().__init__()
        assert family in ("gaussian", "laplace", "t")
        if family == "t":
            assert nu_mode in ("global", "local")
        else:
            assert nu_mode == "none"
        self.family, self.nu_mode = family, nu_mode

        layers = [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
        self.trunk = nn.Sequential(*layers)
        self.log_scale = nn.Linear(hidden, 2)

        if nu_mode == "global":
            # Single learnable scalar; same parameterisation as the local head:
            # ν = 2.1 + exp(log_df), with log_df clamped to [-2, 4].
            self.log_df_global = nn.Parameter(torch.zeros(1))
        elif nu_mode == "local":
            self.log_df_local = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        sigma = torch.exp(self.log_scale(h).clamp(-4.0, 3.0))
        nu = None
        if self.nu_mode == "global":
            nu = 2.1 + torch.exp(self.log_df_global.clamp(-2.0, 4.0))
            nu = nu.expand(x.shape[0], 1)
        elif self.nu_mode == "local":
            nu = 2.1 + torch.exp(self.log_df_local(h).clamp(-2.0, 4.0))
        return sigma, nu


def _nll(family, r, sigma, nu):
    if family == "gaussian": return _nll_gaussian(r, sigma)
    if family == "laplace":  return _nll_laplace(r, sigma)
    return _nll_t(r, sigma, nu)


# ─── Forced-family CDE trainer (mirrors train_cde_ap structurally) ──────
def train_cde_forced(X, Y, family, nu_mode,
                     cal_frac=0.2,
                     point_epochs=800, density_epochs=400,
                     batch_size=128, lr_point=1e-3, lr_density=5e-4,
                     weight_decay=1e-5, seed=0):
    rng = np.random.RandomState(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(X)
    idx = rng.permutation(n)
    n_cal = max(20, int(cal_frac * n))
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    Xf, Yf = X[fit_idx], Y[fit_idx]
    Xc, Yc = X[cal_idx], Y[cal_idx]

    X_mean, X_std = Xf.mean(0), Xf.std(0) + 1e-6
    Xf_s = (Xf - X_mean) / X_std
    Xc_s = (Xc - X_mean) / X_std

    # ── Stage 1: PointNet (identical to experiment_v9.py) ───────────────
    pnet = PointNet(Xf.shape[1]).to(dev)
    opt = optim.Adam(pnet.parameters(), lr=lr_point, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.tensor(Xf_s, dtype=torch.float32),
                                      torch.tensor(Yf,   dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, drop_last=False)
    for _ in range(point_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.mse_loss(pnet(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    pnet.eval()

    with torch.no_grad():
        mu_fit = pnet(torch.tensor(Xf_s, dtype=torch.float32).to(dev)).cpu().numpy()
    R_fit = Yf - mu_fit

    # ── Stage 2: forced-family density head ─────────────────────────────
    dnet = DensityHead(Xf.shape[1], family, nu_mode).to(dev)
    opt = optim.Adam(dnet.parameters(), lr=lr_density, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.tensor(Xf_s,  dtype=torch.float32),
                                      torch.tensor(R_fit, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, drop_last=False)
    for _ in range(density_epochs):
        for xb, rb in loader:
            xb, rb = xb.to(dev), rb.to(dev)
            sigma, nu = dnet(xb)
            loss = _nll(family, rb, sigma, nu)
            opt.zero_grad(); loss.backward(); opt.step()
    dnet.eval()

    # ── Stage 3: split-conformal (same procedure as CDE-AP) ─────────────
    with torch.no_grad():
        Xc_t = torch.tensor(Xc_s, dtype=torch.float32).to(dev)
        mu_cal = pnet(Xc_t).cpu().numpy()
        sigma_cal, nu_cal = dnet(Xc_t)
        sigma_cal_np = sigma_cal.cpu().numpy()
    R_cal = Yc - mu_cal
    nll_cal = _nll(family, torch.tensor(R_cal, dtype=torch.float32).to(dev),
                   sigma_cal, nu_cal).item()
    scores = np.max(np.abs(R_cal) / (sigma_cal_np + 1e-8), axis=1)
    alpha = 0.05
    q_level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    q_hat = float(np.quantile(scores, q_level, method="higher"))

    return {"pnet": pnet, "dnet": dnet, "family": family, "nu_mode": nu_mode,
            "q_hat": q_hat, "nll_cal": nll_cal,
            "X_mean": X_mean, "X_std": X_std, "dev": dev}


def predict_cde_forced(model, X_te, n_samples=500, rng=None):
    if rng is None: rng = np.random
    dev = model["dev"]
    Xt = torch.tensor((X_te - model["X_mean"]) / model["X_std"],
                      dtype=torch.float32).to(dev)
    with torch.no_grad():
        mu = model["pnet"](Xt).cpu().numpy()
        sigma, nu = model["dnet"](Xt)
        sigma = sigma.cpu().numpy()
        nu_np = nu.cpu().numpy() if nu is not None else None

    B = len(X_te)
    fam = model["family"]
    if fam == "gaussian":
        noise = rng.standard_normal((n_samples, B, 2)).astype(np.float32)
    elif fam == "laplace":
        noise = rng.laplace(0.0, 1.0/np.sqrt(2.0), (n_samples, B, 2)).astype(np.float32)
    else:  # Student-t
        g = rng.standard_normal((n_samples, B, 2)).astype(np.float32)
        nu_flat = nu_np[:, 0]
        chi = rng.chisquare(nu_flat, size=(n_samples, B)).astype(np.float32) + 1e-3
        inv = np.sqrt(nu_flat[None, :] / chi)[:, :, None]
        noise = g * inv
    samples = mu[None, :, :] + sigma[None, :, :] * noise

    q = model["q_hat"]
    lo = mu - q * sigma
    hi = mu + q * sigma
    return mu, sigma, samples, lo, hi


# ─── Differentiating diagnostics ────────────────────────────────────────
def joint_interval_width(lo, hi):
    """Mean joint rectangle 'width' = product of side lengths^(1/2),
       i.e. geometric mean of per-coord widths. Reported per test point."""
    w = hi - lo                                  # (n_te, 2)
    return float(np.sqrt(w[:, 0] * w[:, 1]).mean())


def per_coord_width(lo, hi):
    w = hi - lo
    return {"width_v1": float(w[:, 0].mean()),
            "width_v2": float(w[:, 1].mean())}


def conditional_cov_by_sigma_decile(y, mu, sigma, lo, hi, n_bins=10):
    """Conditional coverage in deciles of σ̂ (averaged over coords).
       Returns coverage in each decile — flat ≈0.95 means well-calibrated,
       sloped means the global q_hat is over/under-compensating somewhere."""
    s_bar = sigma.mean(axis=1)
    edges = np.quantile(s_bar, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_id = np.digitize(s_bar, edges[1:-1], right=False)
    inside = np.all((y >= lo) & (y <= hi), axis=1)
    cov_per_bin = []
    for b in range(n_bins):
        mask = bin_id == b
        cov_per_bin.append(float(inside[mask].mean()) if mask.any() else np.nan)
    return cov_per_bin


# ─── Main ablation runner ───────────────────────────────────────────────
CONDITIONS = [
    # (label,        family,     nu_mode)
    ("CDE-Gauss",     "gaussian", "none"),
    ("CDE-Laplace",   "laplace",  "none"),
    ("CDE-t-global",  "t",        "global"),
    ("CDE-AP",        "t",        "local"),     # full AP-style: t with ν(x)
]


def run_one_replicate(Xtr, Ytr, Xte, Yte, seed):
    out = []
    for label, fam, nu_mode in CONDITIONS:
        model = train_cde_forced(Xtr, Ytr, family=fam, nu_mode=nu_mode, seed=seed)
        mu, sigma, samples, lo, hi = predict_cde_forced(model, Xte)
        m = metrics_conformal(samples, mu, Yte, lo, hi, mode="joint")
        m.update(metrics_conformal(samples, mu, Yte, lo, hi, mode="marginal"))
        m["model"]     = label
        m["family"]    = fam
        m["nu_mode"]   = nu_mode
        m["q_hat"]     = model["q_hat"]
        m["nll_cal"]   = model["nll_cal"]
        m["width"]     = joint_interval_width(lo, hi)
        m.update(per_coord_width(lo, hi))
        m["cond_cov"]  = conditional_cov_by_sigma_decile(Yte, mu, sigma, lo, hi)
        out.append(m)
    return out


def run_ablation(tag="setting_B", n_sim=50):
    rows = []
    for i in range(1, n_sim + 1):
        tr = pd.read_csv(f"{tag}/training_data/2D_{tag}_1200_{i}-train.csv")
        te = pd.read_csv(f"{tag}/testing_data/2D_{tag}_1200_{i}-test.csv")
        cov_cols = [c for c in tr.columns if c.startswith("cov")]
        feat = ["x", "y"] + cov_cols
        Xtr, Xte = tr[feat].values, te[feat].values
        Ytr, Yte = tr[["var1", "var2"]].values, te[["var1", "var2"]].values
        rep = run_one_replicate(Xtr, Ytr, Xte, Yte, seed=i)
        for r in rep: r["sim"] = i
        rows += rep
        print(f"[{tag}] replicate {i}/{n_sim} done")

    df = pd.DataFrame(rows)

    # Summary table (the headline numbers go in the paper)
    summary_cols = ["MSE", "MAD", "MD", "CRPS", "COV95",
                    "q_hat", "nll_cal", "width", "width_v1", "width_v2"]
    summary = df.groupby("model")[summary_cols].mean().round(4)
    summary = summary.reindex([c[0] for c in CONDITIONS])

    print("\n=== Headline summary (means over %d replicates, %s) ===" % (n_sim, tag))
    print(summary.to_string())

    # Conditional coverage by σ̂-decile (the second figure)
    cond = (df.groupby("model")["cond_cov"]
              .apply(lambda s: np.nanmean(np.stack(s.values), axis=0).round(3)))
    cond = cond.reindex([c[0] for c in CONDITIONS])
    print("\n=== Conditional COV95 by σ̂-decile (target 0.95) ===")
    for label, vec in cond.items():
        print(f"  {label:14s}  {list(vec)}")

    # Save raw results for plotting
    df.to_csv(f"ablation_{tag}.csv", index=False)
    summary.to_csv(f"ablation_{tag}_summary.csv")
    with open(f"ablation_{tag}_cond_cov.json", "w") as f:
        json.dump({k: list(map(float, v)) for k, v in cond.items()}, f, indent=2)
    print(f"\nSaved: ablation_{tag}.csv, ablation_{tag}_summary.csv,"
          f" ablation_{tag}_cond_cov.json")
    return df


if __name__ == "__main__":
    tag   = sys.argv[1] if len(sys.argv) > 1 else "setting_B"
    n_sim = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_ablation(tag=tag, n_sim=n_sim)
