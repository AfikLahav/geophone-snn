"""Fold-scheme experiment ("optimal 5-fold"): same E2 model + finetune protocol as
finetune_5fold.py, but comparing fold CONSTRUCTIONS:
  contiguous : k-th temporal fifth of each file (current protocol)
  seg30      : 30 s segments (20 windows) dealt round-robin to folds — spreads content
               across folds, keeps overlapping windows together (leakage ~1 win/boundary)
  seg30_sig  : seg30 on the SIGNAL-ONLY dataset (sub-floor present windows excluded,
               p90-ambient gate per paired nothing session)
  win_ileave : window i -> fold i%5 — INVALID (50%-overlap windows leak train<->test);
               computed only to quantify the inflation.
Reports zero-shot + fine-tuned(ens5) mean/std/per-fold per scheme. Writes EXP_FOLDS.json.
Usage: python exp_folds.py <model_out_dir>
"""
import os, sys, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import warnings; warnings.filterwarnings("ignore")
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4"))
import features as F, label as L
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, SCENE, HOP, K, SCALE = 4, 30 * int(F.FS), 1500, 5, 25.4
SEG = 20                                                     # windows per segment (30 s unique)
torch.manual_seed(0); np.random.seed(0)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "snn_v4_out", "E2_uniform_sampler")
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
CLIPZ = scj["clip"]; fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]


class SeqBN(nn.Module):
    def __init__(s, c): super().__init__(); s.bn = nn.BatchNorm1d(c)
    def forward(s, x): T_, B_, C_ = x.shape; return s.bn(x.reshape(T_ * B_, C_)).reshape(T_, B_, C_)
class FeatureSNN(nn.Module):
    def __init__(s, nf, w=(512, 256, 128), dp=0.1):
        super().__init__(); s.gate = nn.Parameter(torch.ones(nf)); b = []; d = nf
        for ww in w:
            b += [layer.Linear(d, ww), SeqBN(ww), neuron.ParametricLIFNode(init_tau=2.0,
                  surrogate_function=surrogate.ATan(2.0), detach_reset=True, step_mode="m"), layer.Dropout(dp)]; d = ww
        s.body = nn.Sequential(*b)
        def h(o): return nn.Sequential(layer.Linear(d, o), neuron.LIFNode(v_threshold=float("inf"),
                  surrogate_function=surrogate.ATan(), step_mode="m", store_v_seq=True, backend="torch"))
        s.human, s.animal, s.vehicle = h(2), h(2), h(1)
    def forward(s, x):
        xs = (x * s.gate).unsqueeze(0).repeat(T, 1, 1); hh = s.body(xs); o = {}
        for nm, hd in (("human", s.human), ("animal", s.animal), ("vehicle", s.vehicle)): hd(hh); o[nm] = hd[-1].v_seq.mean(0)
        return o


PAIRS = {"car.csv": ("vehicle", "car_nothing.csv"), "human.csv": ("human", "human_nothing.csv"),
         "car_nothing.csv": ("nothing", None), "human_nothing.csv": ("nothing", None)}


def featurize(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE
    fe, rms = [], []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP):
            fe.append(F.window_features(pre, i0).astype(np.float32))
            w = seg[i0:i0 + F.NW].astype(np.float64)
            rms.append({b: L._band_rms(w, *L.BANDS[b]) for b in ("human", "vehicle")})
    return np.nan_to_num(np.stack(fe))[:, fidx], rms


files = {}
for fn in PAIRS:
    files[fn] = featurize(os.path.join(REAL_DIR, fn))
floors = {fn: {b: np.percentile([w[b] for w in files[fn][1]], 90) for b in ("human", "vehicle")}
          for fn in ("car_nothing.csv", "human_nothing.csv")}


def build(scheme):
    X, y, fold = [], [], []
    for fn, (cls, noth) in PAIRS.items():
        Xf, rms = files[fn]; n = len(Xf)
        keep = np.ones(n, bool)
        if scheme.endswith("_sig") and noth is not None:
            fl = floors[noth][cls]
            keep = np.array([w[cls] > fl for w in rms])
        idx = np.arange(n)[keep]; m = len(idx)
        if scheme.startswith("contiguous"):
            fo = np.arange(m) * K // max(m, 1)
        elif scheme.startswith("seg30"):
            fo = (np.arange(m) // SEG) % K
        elif scheme.startswith("win_ileave"):
            fo = np.arange(m) % K
        X.append(Xf[idx]); y += [cls] * m; fold.append(fo)
    return np.vstack(X), np.array(y), np.concatenate(fold)


def probs(model, Xa):
    model.eval(); ph, pv = [], []
    with torch.no_grad():
        for j in range(0, len(Xa), 4096):
            functional.reset_net(model); o = model(torch.as_tensor(Xa[j:j + 4096]).to(DEV))
            ph.append(torch.sigmoid(o["human"][:, 0]).cpu()); pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
    return np.array(torch.cat(ph)), np.array(torch.cat(pv))
def thr(p, t):
    g = np.linspace(0.05, 0.95, 19); return float(g[np.argmax([balanced_accuracy_score(t, p > x) for x in g])])
def decide(ph, pv, th, tv):
    mh, mv = ph - th, pv - tv; return np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
def finetune(model, Xz, yh, yv, tr, seed, epochs=80, lr=4e-4):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xz[tr]).to(DEV); ht = torch.as_tensor(yh[tr]).to(DEV); vt = torch.as_tensor(yv[tr]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); model.train()
    n = len(Xt); bs = 256
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + np.cos(np.pi * ep / epochs))
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]; functional.reset_net(model); o = model(Xt[idx])
            loss = (Fnn.binary_cross_entropy_with_logits(o["human"][:, 0], ht[idx])
                    + Fnn.binary_cross_entropy_with_logits(o["vehicle"][:, 0], vt[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


ENS = 5
results = {}
for scheme in ("contiguous", "seg30", "seg30_sig", "win_ileave"):
    X, y, fold = build(scheme)
    yh = (y == "human").astype(np.float32); yv = (y == "vehicle").astype(np.float32)
    zero, fine = [], []
    for k in range(K):
        te = fold == k; tr = ~te
        sc = StandardScaler().fit(X[tr]); Xz = np.clip(sc.transform(X), -CLIPZ, CLIPZ).astype(np.float32)
        base = FeatureSNN(NF).to(DEV); functional.set_step_mode(base, "m")
        base.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV))
        ph0, pv0 = probs(base, Xz); th_, tv_ = thr(ph0[tr], yh[tr] > 0), thr(pv0[tr], yv[tr] > 0)
        zero.append(accuracy_score(y[te], decide(ph0[te], pv0[te], th_, tv_)))
        phs, pvs = np.zeros(len(X)), np.zeros(len(X))
        for s in range(ENS):
            ft = finetune(copy.deepcopy(base), Xz, yh, yv, tr, seed=s)
            a, b = probs(ft, Xz); phs += a; pvs += b
        ph, pv = phs / ENS, pvs / ENS
        th_, tv_ = thr(ph[tr], yh[tr] > 0), thr(pv[tr], yv[tr] > 0)
        fine.append(accuracy_score(y[te], decide(ph[te], pv[te], th_, tv_)))
    results[scheme] = {"n": int(len(X)),
                       "zero_mean": round(float(np.mean(zero)), 4), "zero_per_fold": [round(z, 3) for z in zero],
                       "fine_mean": round(float(np.mean(fine)), 4), "fine_std": round(float(np.std(fine)), 4),
                       "fine_per_fold": [round(z, 3) for z in fine]}
    print(f"{scheme:12s} n={len(X):5d} | zero {np.mean(zero):.3f} {np.round(zero,3)} | "
          f"fine {np.mean(fine):.3f}±{np.std(fine):.3f} {np.round(fine,3)}", flush=True)

json.dump(results, open(os.path.join(HERE, "EXP_FOLDS.json"), "w"), indent=1)
print("wrote EXP_FOLDS.json")
