"""
Experiment v6: cGAN, cGAN-AP (v5 verbatim) + CGA++ + KNN/RF
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

# ── CGA++ ───────────────────────────────────────────────────────────
class MixturePrior(nn.Module):
    def __init__(self, cond_dim, z_dim=16, K=4):
        super().__init__()
        self.K, self.z_dim = K, z_dim
        self.net = nn.Sequential(nn.Linear(cond_dim,64), nn.ReLU(),
                                 nn.Linear(64, K*(2*z_dim+1)))
    def sample(self, cond):
        h = self.net(cond); B = cond.size(0)
        logits = h[:,:self.K]
        mus = h[:, self.K:self.K+self.K*self.z_dim].view(B,self.K,self.z_dim)
        logs = h[:, self.K+self.K*self.z_dim:].view(B,self.K,self.z_dim)
        k = torch.multinomial(F.softmax(logits,-1),1).squeeze(-1)
        mu = mus[torch.arange(B),k]
        sd = torch.exp(0.5*logs[torch.arange(B),k].clamp(-5,2))
        return mu + sd*torch.randn_like(mu)

class GeneratorPP(nn.Module):
    def __init__(self, cond_dim, z_dim=16, t_dim=4, hidden=128):
        super().__init__()
        self.prior = MixturePrior(cond_dim, z_dim); self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim+z_dim+t_dim,hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden,hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden,2))
    def forward(self, cond):
        z = self.prior.sample(cond)
        g = torch.randn(cond.size(0), self.t_dim, device=cond.device)
        chi = torch.distributions.Chi2(3).sample((cond.size(0), self.t_dim)).to(cond.device)/3
        t = g/torch.sqrt(chi)
        return self.net(torch.cat([cond,z,t],-1))

class DiscriminatorPP(nn.Module):
    def __init__(self, cond_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            spectral_norm(nn.Linear(cond_dim+2,hidden)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden,hidden)), nn.LeakyReLU(0.2))
        self.adv = spectral_norm(nn.Linear(hidden,1))
        self.q = spectral_norm(nn.Linear(hidden,6))
    def forward(self, cond, y):
        h = self.body(torch.cat([cond,y],-1))
        return self.adv(h), self.q(h)

def _es(f1,f2,y):
    return (f1-y).norm(dim=-1).mean() - 0.5*(f1-f2).norm(dim=-1).mean()
def _pin(q,y):
    L = 0.0
    for i,t in enumerate((0.025,0.5,0.975)):
        d = y - q[:, i*2:(i+1)*2]
        L = L + torch.maximum(t*d,(t-1)*d).mean()
    return L
def _ctx(Xq, Xr, Yr, k=8):
    nn_ = NearestNeighbors(n_neighbors=k).fit(Xr)
    _, idx = nn_.kneighbors(Xq)
    return Yr[idx].mean(axis=1)

def train_cga_pp(X, Y, epochs=400, batch_size=256, k_ctx=8):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond = np.hstack([X, _ctx(X, X, Y, k_ctx)])
    G = GeneratorPP(cond.shape[1]).to(dev)
    D = DiscriminatorPP(cond.shape[1]).to(dev)
    oG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))
    oD = optim.Adam(D.parameters(), lr=4e-4, betas=(0.5,0.9))
    loader = DataLoader(TensorDataset(torch.tensor(cond,dtype=torch.float32),
                                      torch.tensor(Y,dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    for _ in range(epochs):
        for cb, yb in loader:
            cb, yb = cb.to(dev), yb.to(dev)
            fake = G(cb).detach()
            d_r,_ = D(cb,yb); d_f,_ = D(cb,fake)
            lossD = F.relu(1-d_r).mean() + F.relu(1+d_f).mean()
            oD.zero_grad(); lossD.backward(); oD.step()
            f1, f2 = G(cb), G(cb)
            d_fg, q_fg = D(cb, f1)
            lossG = -d_fg.mean() + 2.0*_es(f1,f2,yb) + 1.0*_pin(q_fg,yb)
            oG.zero_grad(); lossG.backward(); oG.step()
    G.eval()
    return G, dev

def predict_cga_pp(G, X_te, X_tr, Y_tr, dev, n_samples=500, k_ctx=8):
    cond = np.hstack([X_te, _ctx(X_te, X_tr, Y_tr, k_ctx)])
    c = torch.tensor(cond, dtype=torch.float32).to(dev)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(G(c).cpu().numpy())
    p = np.stack(preds,0)
    return p.mean(0), p.std(0), p

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
    for tag in ["setting_A","setting_B","setting_C"]:
        for mode in ["joint","marginal"]:
            run_scheme(tag, n_sim=50, report_mode=mode)
