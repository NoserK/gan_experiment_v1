"""
Experiment v9: cGAN, cGAN-AP (v5 verbatim) + CDE-AP + KNN/RF

CDE-AP (Conditional Density Estimator with Adaptive Parametric family):
  Stage 1 — PointNet. Deep MLP trained with plain MSE on standardized
            inputs. Its only job is to estimate E[Y | X].
  Stage 2 — DensityNet. Parametric residual-density head. The family
            (Gaussian / Laplace / Student-t) is chosen by the AP
            kurtosis rule applied to Stage-1 RESIDUALS (not to Y
            directly — residual tail shape is what actually matters).
            Outputs sigma(x), plus per-location nu(x) if family = "t".
            Trained by NLL on held-out residuals.
  Stage 3 — Split-conformal calibration. A 20% slice of training data
            is reserved from both networks. On that slice, compute
            nonconformity scores
                s_i = max_j  |r_{i,j}| / sigma_j(x_i)
            and take q_hat = the finite-sample 95th percentile. The
            prediction interval for a test point is then
                [ mu_j(x) - q_hat * sigma_j(x),
                  mu_j(x) + q_hat * sigma_j(x) ].
            Joint rectangular coverage is >= 95% by construction.

Samples drawn from the parametric residual family give CRPS;
COV95 is taken from the conformal intervals (not sample percentiles).
"""
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import kurtosis as sp_kurtosis

# ── AP classifier (shared across cGAN-AP and CDE-AP) ────────────────
def classify_tail_behavior(A):
    kappas = [sp_kurtosis(A[:, c], fisher=True) for c in range(A.shape[1])]
    km = np.max(kappas)
    if km <= 1.5: return "gaussian"
    if km <= 6.0: return "laplace"
    return "t"

def _sample_noise(bs, noise_dim, device, prior_type="gaussian"):
    if prior_type == "gaussian":
        return torch.randn(bs, noise_dim, device=device)
    if prior_type == "laplace":
        z = torch.distributions.Laplace(0.0, 1.0/np.sqrt(2.0)).sample((bs, noise_dim))
        return z.to(device)
    z = np.random.standard_t(df=5, size=(bs, noise_dim)).astype(np.float32)
    return torch.from_numpy(z).to(device)

# ── v5 cGAN (verbatim) ──────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, cond_dim, noise_dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim+noise_dim,hidden), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden),
            nn.Linear(hidden,hidden), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden),
            nn.Linear(hidden,hidden//2), nn.LeakyReLU(0.2),
            nn.Linear(hidden//2,2))
        self.noise_dim = noise_dim
    def forward(self, cond, z): return self.net(torch.cat([cond,z],1))

class Discriminator(nn.Module):
    def __init__(self, cond_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim+2,hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden,hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden,hidden//2), nn.LeakyReLU(0.2),
            nn.Linear(hidden//2,1), nn.Sigmoid())
    def forward(self, cond, y): return self.net(torch.cat([cond,y],1))

def train_cgan(X, Y, cond_dim, noise_dim=32, hidden=128, epochs=800,
               batch_size=128, lr=2e-4, lambda_recon=0.1, prior_type="gaussian"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, D = Generator(cond_dim,noise_dim,hidden).to(dev), Discriminator(cond_dim,hidden).to(dev)
    oG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    oD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    bce, l1 = nn.BCELoss(), nn.L1Loss()
    loader = DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32),
                                      torch.tensor(Y,dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    for _ in range(epochs):
        for cb, rb in loader:
            bs = cb.size(0); cb, rb = cb.to(dev), rb.to(dev)
            rl = torch.ones(bs,1,device=dev)*0.9
            fl = torch.zeros(bs,1,device=dev)+0.1
            z = _sample_noise(bs,noise_dim,dev,prior_type)
            fake = G(cb,z).detach()
            lossD = (bce(D(cb,rb),rl)+bce(D(cb,fake),fl))/2
            oD.zero_grad(); lossD.backward(); oD.step()
            z = _sample_noise(bs,noise_dim,dev,prior_type)
            fake = G(cb,z)
            lossG = bce(D(cb,fake), torch.ones(bs,1,device=dev)) + lambda_recon*l1(fake,rb)
            oG.zero_grad(); lossG.backward(); oG.step()
    G.eval()
    return G, dev, noise_dim

def predict_cgan(G, X, dev, noise_dim, n_samples=500, prior_type="gaussian"):
    Xt = torch.tensor(X,dtype=torch.float32).to(dev)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = _sample_noise(Xt.size(0), noise_dim, dev, prior_type)
            preds.append(G(Xt,z).cpu().numpy())
    p = np.stack(preds,0)
    return p.mean(0), p.std(0), p

# ── CDE-AP architecture ─────────────────────────────────────────────
class PointNet(nn.Module):
    """Pure MSE regressor. No stochasticity, no adversarial term."""
    def __init__(self, in_dim, hidden=128, n_layers=4, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DensityNet(nn.Module):
    """Outputs sigma(x) per coordinate; also nu(x) if family='t'."""
    def __init__(self, in_dim, family, hidden=128, n_layers=4):
        super().__init__()
        self.family = family
        layers = [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
        self.trunk = nn.Sequential(*layers)
        self.log_scale = nn.Linear(hidden, 2)
        if family == "t":
            self.log_df = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.trunk(x)
        sigma = torch.exp(self.log_scale(h).clamp(-4.0, 3.0))
        nu = None
        if self.family == "t":
            nu = 2.1 + torch.exp(self.log_df(h).clamp(-2.0, 4.0))   # ensures Var < inf
        return sigma, nu

# Residual NLLs. sigma is the marginal standard deviation in every case
# (Laplace is parameterized so that Var = sigma^2 to keep the scale
# interpretation consistent across families).
def _nll_gaussian(r, sigma):
    return (0.5 * (r / sigma)**2 + torch.log(sigma)
            + 0.5 * np.log(2 * np.pi)).sum(-1).mean()

def _nll_laplace(r, sigma):
    b = sigma / np.sqrt(2.0)
    return (torch.log(2.0 * b) + torch.abs(r) / b).sum(-1).mean()

def _nll_t(r, sigma, nu):
    # Independent Student-t per coordinate, sharing nu across coords
    z = r / sigma
    half_nu = 0.5 * nu
    log_const = (torch.lgamma(half_nu + 0.5) - torch.lgamma(half_nu)
                 - 0.5 * torch.log(nu * np.pi))
    log_kernel = -0.5 * (nu + 1.0) * torch.log1p(z.pow(2) / nu)
    log_p = log_const + log_kernel - torch.log(sigma)
    return -log_p.sum(-1).mean()

def _nll(family, r, sigma, nu):
    if family == "gaussian": return _nll_gaussian(r, sigma)
    if family == "laplace":  return _nll_laplace(r, sigma)
    return _nll_t(r, sigma, nu)

def train_cde_ap(X, Y, cal_frac=0.2,
                 point_epochs=800, density_epochs=400,
                 batch_size=128, lr_point=1e-3, lr_density=5e-4,
                 weight_decay=1e-5, seed=0):
    rng = np.random.RandomState(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split: fit vs calibration
    n = len(X)
    idx = rng.permutation(n)
    n_cal = max(20, int(cal_frac * n))
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    Xf, Yf = X[fit_idx], Y[fit_idx]
    Xc, Yc = X[cal_idx], Y[cal_idx]

    # Standardize X using the fit slice
    X_mean = Xf.mean(0); X_std = Xf.std(0) + 1e-6
    Xf_s, Xc_s = (Xf - X_mean) / X_std, (Xc - X_mean) / X_std

    # ── Stage 1: PointNet (pure MSE) ────────────────────────────────
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

    # Residuals on fit set — used for AP classification and Stage 2
    with torch.no_grad():
        mu_fit = pnet(torch.tensor(Xf_s, dtype=torch.float32).to(dev)).cpu().numpy()
    R_fit = Yf - mu_fit

    # ── AP: family chosen from residual kurtosis ────────────────────
    family = classify_tail_behavior(R_fit)

    # ── Stage 2: DensityNet on residuals ────────────────────────────
    dnet = DensityNet(Xf.shape[1], family).to(dev)
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

    # ── Stage 3: split-conformal calibration ────────────────────────
    with torch.no_grad():
        Xc_t = torch.tensor(Xc_s, dtype=torch.float32).to(dev)
        mu_cal = pnet(Xc_t).cpu().numpy()
        sigma_cal, _ = dnet(Xc_t)
        sigma_cal = sigma_cal.cpu().numpy()
    R_cal = Yc - mu_cal
    scores = np.max(np.abs(R_cal) / (sigma_cal + 1e-8), axis=1)
    alpha = 0.05
    q_level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    q_hat = float(np.quantile(scores, q_level, method="higher"))

    return {"pnet": pnet, "dnet": dnet, "family": family, "q_hat": q_hat,
            "X_mean": X_mean, "X_std": X_std, "dev": dev}

def predict_cde_ap(model, X_te, n_samples=500, rng=None):
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
    family = model["family"]
    if family == "gaussian":
        noise = rng.standard_normal((n_samples, B, 2)).astype(np.float32)
    elif family == "laplace":
        noise = rng.laplace(0.0, 1.0/np.sqrt(2.0), (n_samples, B, 2)).astype(np.float32)
    else:  # Student-t with per-location nu
        g = rng.standard_normal((n_samples, B, 2)).astype(np.float32)
        nu_flat = nu_np[:, 0]                                    # (B,)
        chi = rng.chisquare(nu_flat, size=(n_samples, B)).astype(np.float32) + 1e-3
        inv = np.sqrt(nu_flat[None, :] / chi)[:, :, None]
        noise = g * inv
    samples = mu[None, :, :] + sigma[None, :, :] * noise

    q = model["q_hat"]
    lo = mu - q * sigma
    hi = mu + q * sigma
    return mu, sigma, samples, lo, hi

# ── Metrics ─────────────────────────────────────────────────────────
def metrics(samples, mean, y, mode="joint"):
    o = {}
    if mode == "joint":
        o["MSE"] = ((mean-y)**2).mean()
        o["MAD"] = np.abs(mean-y).mean()
        o["MD"]  = np.linalg.norm(mean-y, axis=1).mean()
        if samples is not None:
            a = np.linalg.norm(samples-y[None], axis=-1).mean()
            b = 0.5*np.linalg.norm(samples[:,None]-samples[None], axis=-1).mean()
            o["CRPS"] = a-b
            lo = np.percentile(samples,2.5,0); hi = np.percentile(samples,97.5,0)
            o["COV95"] = np.all((y>=lo)&(y<=hi),axis=1).mean()
    else:
        for j,n in enumerate(["v1","v2"]):
            o[f"MSE_{n}"] = ((mean[:,j]-y[:,j])**2).mean()
            o[f"MAD_{n}"] = np.abs(mean[:,j]-y[:,j]).mean()
            if samples is not None:
                lo = np.percentile(samples[:,:,j],2.5,0)
                hi = np.percentile(samples[:,:,j],97.5,0)
                o[f"COV95_{n}"] = ((y[:,j]>=lo)&(y[:,j]<=hi)).mean()
    return o

def metrics_conformal(samples, mean, y, lo, hi, mode="joint"):
    """Like metrics(), but COV95 is recomputed from conformal intervals."""
    o = metrics(samples, mean, y, mode)
    if mode == "joint":
        o["COV95"] = np.all((y >= lo) & (y <= hi), axis=1).mean()
    else:
        for j, n in enumerate(["v1", "v2"]):
            o[f"COV95_{n}"] = ((y[:, j] >= lo[:, j]) & (y[:, j] <= hi[:, j])).mean()
    return o

# ── Runner ──────────────────────────────────────────────────────────
def run_scheme(tag, n_sim=50, report_mode="joint"):
    rows = []
    for i in range(1, n_sim+1):
        tr = pd.read_csv(f"{tag}/training_data/2D_{tag}_1200_{i}-train.csv")
        te = pd.read_csv(f"{tag}/testing_data/2D_{tag}_1200_{i}-test.csv")
        cov_cols = [c for c in tr.columns if c.startswith("cov")]
        feat = ["x","y"] + cov_cols
        Xtr, Xte = tr[feat].values, te[feat].values
        Ytr, Yte = tr[["var1","var2"]].values, te[["var1","var2"]].values

        # cGAN
        G,dv,nd = train_cgan(Xtr,Ytr,Xtr.shape[1],prior_type="gaussian")
        mu,_,sm = predict_cgan(G,Xte,dv,nd,prior_type="gaussian")
        m = metrics(sm,mu,Yte,report_mode); m.update(model="cGAN",sim=i); rows.append(m)

        # cGAN-AP
        pr = classify_tail_behavior(Ytr)
        G,dv,nd = train_cgan(Xtr,Ytr,Xtr.shape[1],prior_type=pr)
        mu,_,sm = predict_cgan(G,Xte,dv,nd,prior_type=pr)
        m = metrics(sm,mu,Yte,report_mode); m.update(model="cGAN-AP",sim=i,prior=pr); rows.append(m)

        # CDE-AP
        cde = train_cde_ap(Xtr, Ytr, seed=i)
        mu, sigma, sm, lo, hi = predict_cde_ap(cde, Xte)
        m = metrics_conformal(sm, mu, Yte, lo, hi, report_mode)
        m.update(model="CDE-AP", sim=i, family=cde["family"])
        rows.append(m)

        for name,pred in [("KNN",KNeighborsRegressor(10).fit(Xtr,Ytr).predict(Xte)),
                          ("RF",RandomForestRegressor(200,n_jobs=-1).fit(Xtr,Ytr).predict(Xte))]:
            m = metrics(None,pred,Yte,report_mode); m.update(model=name,sim=i); rows.append(m)

    df = pd.DataFrame(rows)
    print(f"\n=== {tag} [{report_mode}] ===")
    print(df.groupby("model").mean(numeric_only=True).round(4))
    return df

if __name__ == "__main__":
    for tag in ["setting_A","setting_B"]:
        for mode in ["joint","marginal"]:
            run_scheme(tag, n_sim=50, report_mode=mode)
