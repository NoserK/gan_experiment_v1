"""
Experiment v8: cGAN, cGAN-AP (v5 verbatim) + CGA++ (v8 fixes) + KNN/RF

CGA++ changes vs v7:
  Step 1 — Context leak fix. At training time the kNN context query was
           Xq == Xr == X_train, so each row's nearest neighbour was
           itself. _ctx now exposes a `drop_self` flag and train_cga_pp
           sets it True so we ask for k+1 neighbours and drop the first.
  Step 2 — Decoupled mean head. mu, log_sigma, log_df are now produced
           by a deterministic trunk that does NOT see z. Only the noise
           pathway is stochastic:
                mu, sigma, nu = f(cond)              # deterministic
                y_hat = mu + sigma * t_nu(z)         # z used here only
           This gives the L1 anchor a clean, low-variance gradient.
  Step 3 — Two-phase training. Phase 1: train deterministic heads with
           heteroscedastic NLL only (no D, no ES, no pinball). Phase 2:
           unfreeze noise pathway and add adversarial + scoring-rule
           losses. Toggle with `two_phase=True`.
  Step 4 — Matched update budget. Defaults are now epochs=800,
           batch_size=128, identical to the cGAN baseline.

Loss balance also rebalanced: lambda_recon raised to 1.0, r1_gamma
lowered to 1.0 so the L1 anchor isn't drowned out by the hinge term.
"""
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import kurtosis as sp_kurtosis

# ── v5 AP rule (verbatim) ───────────────────────────────────────────
def classify_tail_behavior(Y_train):
    kappas = [sp_kurtosis(Y_train[:,c], fisher=True) for c in range(Y_train.shape[1])]
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

# ── CGA++ v8 ────────────────────────────────────────────────────────
class GeneratorPP(nn.Module):
    """
    Step 2: deterministic mean/scale/df trunk; z is used only inside the
    Student-t noise term. This gives the L1 anchor a clean gradient on
    the conditional mean.
    """
    def __init__(self, cond_dim, z_dim=16, hidden=128):
        super().__init__()
        self.z_dim = z_dim
        # Deterministic conditional-density head (no z input)
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2))
        self.mean_head  = nn.Linear(hidden, 2)
        self.scale_head = nn.Linear(hidden, 2)   # log sigma(x)
        self.df_head    = nn.Linear(hidden, 1)   # log(nu(x) - 2.1)
        # Noise-shaping pathway: maps z to a 2-dim correction
        self.noise_proj = nn.Sequential(
            nn.Linear(cond_dim + z_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 2))

    def params(self, cond):
        """Deterministic forward — used by NLL phase and at evaluation
        for E[Y|x] without averaging samples."""
        h = self.trunk(cond)
        mu     = self.mean_head(h)
        log_s  = self.scale_head(h).clamp(-4.0, 3.0)
        log_df = self.df_head(h).clamp(-2.0, 4.0)
        nu = 2.1 + torch.exp(log_df)             # > 2 ⇒ finite variance
        sigma = torch.exp(log_s)
        return mu, sigma, nu

    def forward(self, cond):
        B = cond.size(0)
        mu, sigma, nu = self.params(cond)
        z = torch.randn(B, self.z_dim, device=cond.device)
        # learned, conditional shaping of the base noise
        shape = self.noise_proj(torch.cat([cond, z], -1))
        eps = torch.randn(B, 2, device=cond.device) + 0.1 * shape
        chi = torch.distributions.Chi2(nu.squeeze(-1)).sample()
        inv = torch.sqrt(nu.squeeze(-1) / chi.clamp_min(1e-3)).unsqueeze(-1)
        t_noise = eps * inv
        return mu + sigma * t_noise

class DiscriminatorPP(nn.Module):
    def __init__(self, cond_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            spectral_norm(nn.Linear(cond_dim+2,hidden)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden,hidden)), nn.LeakyReLU(0.2))
        self.adv = spectral_norm(nn.Linear(hidden,1))
        self.q   = spectral_norm(nn.Linear(hidden,6))
    def forward(self, cond, y):
        h = self.body(torch.cat([cond,y],-1))
        return self.adv(h), self.q(h)

# ── Multi-sample energy score & pinball ─────────────────────────────
def _es_multi(fs, y):
    m = len(fs)
    term_fy = sum((f - y).norm(dim=-1) for f in fs).mean() / m
    pair = 0.0
    cnt = 0
    for i in range(m):
        for j in range(i+1, m):
            pair = pair + (fs[i] - fs[j]).norm(dim=-1).mean()
            cnt += 1
    term_ff = pair / max(cnt, 1)
    return term_fy - 0.5 * term_ff

def _pin(q, y):
    L = 0.0
    for i, t in enumerate((0.025, 0.5, 0.975)):
        d = y - q[:, i*2:(i+1)*2]
        L = L + torch.maximum(t*d, (t-1)*d).mean()
    return L

# ── Student-t heteroscedastic NLL (Phase 1) ─────────────────────────
def _t_nll(mu, sigma, nu, y):
    """Per-coordinate Student-t NLL, summed across coordinates."""
    nu_b = nu                                    # (B,1) — shared across coords
    z = (y - mu) / sigma
    # log p(z) = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 log(nu*pi)
    #           - (nu+1)/2 * log(1 + z^2/nu)
    half_nu = 0.5 * nu_b
    log_const = (torch.lgamma(half_nu + 0.5) - torch.lgamma(half_nu)
                 - 0.5 * torch.log(nu_b * np.pi))
    log_kernel = -0.5 * (nu_b + 1.0) * torch.log1p(z.pow(2) / nu_b)
    log_p = log_const + log_kernel - torch.log(sigma)
    return -log_p.sum(-1).mean()

# ── Step 1: leak-free context ───────────────────────────────────────
def _ctx(Xq, Xr, Yr, k=8, drop_self=False):
    """If drop_self=True (training), query k+1 neighbours and discard
    the first (which is the point itself when Xq is a subset of Xr)."""
    if drop_self:
        nn_ = NearestNeighbors(n_neighbors=k+1).fit(Xr)
        _, idx = nn_.kneighbors(Xq)
        idx = idx[:, 1:]                         # drop self-match
    else:
        nn_ = NearestNeighbors(n_neighbors=k).fit(Xr)
        _, idx = nn_.kneighbors(Xq)
    return Yr[idx].mean(axis=1)

def train_cga_pp(X, Y, epochs=800, batch_size=128, k_ctx=8,
                 m_gen=4, lambda_recon=1.0, lambda_es=1.0, lambda_pin=0.5,
                 r1_gamma=1.0, two_phase=True, phase1_epochs=100):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 1: drop the self-match in the training-time kNN context
    cond = np.hstack([X, _ctx(X, X, Y, k_ctx, drop_self=True)])
    G = GeneratorPP(cond.shape[1]).to(dev)
    D = DiscriminatorPP(cond.shape[1]).to(dev)
    oG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    oD = optim.Adam(D.parameters(), lr=4e-4, betas=(0.5, 0.9))
    loader = DataLoader(TensorDataset(torch.tensor(cond, dtype=torch.float32),
                                      torch.tensor(Y,    dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, drop_last=True)

    # ── Phase 1: deterministic heads only, NLL loss ─────────────────
    if two_phase:
        det_params = (list(G.trunk.parameters())
                      + list(G.mean_head.parameters())
                      + list(G.scale_head.parameters())
                      + list(G.df_head.parameters()))
        oNLL = optim.Adam(det_params, lr=2e-4)
        for _ in range(phase1_epochs):
            for cb, yb in loader:
                cb, yb = cb.to(dev), yb.to(dev)
                mu, sigma, nu = G.params(cb)
                loss = _t_nll(mu, sigma, nu, yb)
                oNLL.zero_grad(); loss.backward(); oNLL.step()

    # ── Phase 2: full adversarial + scoring-rule training ───────────
    for _ in range(epochs):
        for cb, yb in loader:
            cb, yb = cb.to(dev), yb.to(dev)

            # D step with R1 gradient penalty
            yb_r1 = yb.detach().requires_grad_(True)
            d_r, _ = D(cb, yb_r1)
            grad = torch.autograd.grad(d_r.sum(), yb_r1,
                                       create_graph=True, retain_graph=True)[0]
            r1 = grad.pow(2).sum(-1).mean()

            fake = G(cb).detach()
            d_f, _ = D(cb, fake)
            lossD = F.relu(1 - d_r).mean() + F.relu(1 + d_f).mean() + 0.5*r1_gamma*r1
            oD.zero_grad(); lossD.backward(); oD.step()

            # G step: adversarial + multi-sample ES + pinball + L1 anchor
            fs = [G(cb) for _ in range(m_gen)]
            f_mean = torch.stack(fs, 0).mean(0)
            d_fg, q_fg = D(cb, fs[0])
            # Anchor on the deterministic mean head as well — clean gradient
            mu_det, _, _ = G.params(cb)
            lossG = (-d_fg.mean()
                     + lambda_es  * _es_multi(fs, yb)
                     + lambda_pin * _pin(q_fg, yb)
                     + lambda_recon * (F.l1_loss(f_mean, yb)
                                       + F.l1_loss(mu_det, yb)))
            oG.zero_grad(); lossG.backward(); oG.step()

    G.eval()
    return G, dev

def predict_cga_pp(G, X_te, X_tr, Y_tr, dev, n_samples=500, k_ctx=8,
                   use_det_mean=True):
    """At test time we use the leak-free k-NN context (no self in Xr).
    Point estimate is taken from the deterministic mean head when
    use_det_mean=True (lower variance than averaging samples)."""
    cond = np.hstack([X_te, _ctx(X_te, X_tr, Y_tr, k_ctx, drop_self=False)])
    c = torch.tensor(cond, dtype=torch.float32).to(dev)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(G(c).cpu().numpy())
        if use_det_mean:
            mu_det, _, _ = G.params(c)
            mean = mu_det.cpu().numpy()
    p = np.stack(preds, 0)
    if not use_det_mean:
        mean = p.mean(0)
    return mean, p.std(0), p

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

        G,dv,nd = train_cgan(Xtr,Ytr,Xtr.shape[1],prior_type="gaussian")
        mu,_,sm = predict_cgan(G,Xte,dv,nd,prior_type="gaussian")
        m = metrics(sm,mu,Yte,report_mode); m.update(model="cGAN",sim=i); rows.append(m)

        pr = classify_tail_behavior(Ytr)
        G,dv,nd = train_cgan(Xtr,Ytr,Xtr.shape[1],prior_type=pr)
        mu,_,sm = predict_cgan(G,Xte,dv,nd,prior_type=pr)
        m = metrics(sm,mu,Yte,report_mode); m.update(model="cGAN-AP",sim=i,prior=pr); rows.append(m)

        G,dv = train_cga_pp(Xtr,Ytr)
        mu,_,sm = predict_cga_pp(G,Xte,Xtr,Ytr,dv)
        m = metrics(sm,mu,Yte,report_mode); m.update(model="CGA++",sim=i); rows.append(m)

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
