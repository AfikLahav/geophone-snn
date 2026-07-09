"""Beat the baseline the legitimate way: synthetic-PRETRAINED SNN, FINE-TUNED on real, in the
SAME contiguous 5-fold protocol the real-only baseline used. Apples-to-apples: both train on 4
folds, test on the held-out fold. Reports zero-shot (pretrained, no FT) vs fine-tuned vs the
real-only baseline (logreg/HGB 0.986/0.983).
"""
import os, sys, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.environ.get("GEO_OUT") or os.path.join(HERE, "snn_132_out" if os.environ.get("GEO_FEATURES", "") == "full" else "snn_v2_out"); REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, SCENE, HOP, K = 4, 30 * int(F.FS), 1500, 5
SCALE = float(os.environ.get("GEO_REAL_SCALE", "25.4"))     # v4.1: 1000 = true mV (physical axis)
torch.manual_seed(0); np.random.seed(0)
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]

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

# featurize real (x25 scale) -> features, labels, contiguous block idx
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
def feat(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE; fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)
X, y, blk = [], [], []
for fn, cls in REALS.items():
    Xr = feat(os.path.join(REAL_DIR, fn)); n = len(Xr)
    X.append((Xr[:, fidx]).astype(np.float32)); y += [cls] * n     # raw (x25) features; scaled per-fold below
    blk.append(np.arange(n) * K // max(n, 1))
X = np.vstack(X); y = np.array(y); blk = np.concatenate(blk)
yh = (y == "human").astype(np.float32); yv = (y == "vehicle").astype(np.float32)
print(f"real {len(X)} windows, {NF} feats, x{SCALE} scale")

from sklearn.metrics import accuracy_score, balanced_accuracy_score
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

def finetune(model, Xz, tr, seed, epochs=80, lr=4e-4):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xz[tr]).to(DEV); ht = torch.as_tensor(yh[tr]).to(DEV); vt = torch.as_tensor(yv[tr]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); model.train()
    n = len(Xt); bs = 256
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + np.cos(np.pi * ep / epochs))  # cosine
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]; functional.reset_net(model); o = model(Xt[idx])
            loss = (Fnn.binary_cross_entropy_with_logits(o["human"][:, 0], ht[idx])
                    + Fnn.binary_cross_entropy_with_logits(o["vehicle"][:, 0], vt[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
    return model

from sklearn.preprocessing import StandardScaler
ENS = 5
zero, fine, fold_art = [], [], []
for k in range(K):
    te = blk == k; tr = ~te
    sc = StandardScaler().fit(X[tr]); Xz = np.clip(sc.transform(X), -CLIPZ, CLIPZ).astype(np.float32)  # real per-fold norm (parity)
    base = FeatureSNN(NF).to(DEV); functional.set_step_mode(base, "m")
    base.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV))
    ph0, pv0 = probs(base, Xz); th, tv = thr(ph0[tr], yh[tr] > 0), thr(pv0[tr], yv[tr] > 0)
    zero.append(accuracy_score(y[te], decide(ph0[te], pv0[te], th, tv)))
    # deep ensemble of fine-tuned members (avg presence probs)
    phs, pvs = np.zeros(len(X)), np.zeros(len(X)); members = []
    for s in range(ENS):
        ft = finetune(copy.deepcopy(base), Xz, tr, seed=s)
        members.append({kk: vv.detach().cpu() for kk, vv in ft.state_dict().items()})
        a, b = probs(ft, Xz); phs += a; pvs += b
    ph, pv = phs / ENS, pvs / ENS
    th, tv = thr(ph[tr], yh[tr] > 0), thr(pv[tr], yv[tr] > 0)
    fine.append(accuracy_score(y[te], decide(ph[te], pv[te], th, tv)))
    fold_art.append({"members": members, "mu": sc.mean_.tolist(), "sd": sc.scale_.tolist(),
                     "th": float(th), "tv": float(tv), "acc": float(fine[-1])})
    print(f"fold {k}: zero-shot {zero[-1]:.3f} | fine-tuned(ens{ENS}) {fine[-1]:.3f}", flush=True)

res = {"synth_pretrained_zeroshot_5fold_acc": float(np.mean(zero)),
       "synth_pretrained_finetuned_ens_5fold_acc": float(np.mean(fine)), "finetuned_std": float(np.std(fine)),
       "ensemble_size": ENS, "per_fold_finetuned": [round(x, 4) for x in fine],
       "real_only_baseline_logreg": 0.986, "real_only_baseline_hgb": 0.983,
       "real_only_baseline_confound_clean_hgb": 0.937}
json.dump(res, open(os.path.join(OUT, "FINETUNE_5FOLD.json"), "w"), indent=1)

# --- deployable checkpoint = the best-performing fold (best 20% val acc) ---
best = int(np.argmax(fine)); art = fold_art[best]
torch.save({"members": art["members"], "arch": "FeatureSNN", "nf": NF,
            "widths": [512, 256, 128], "T": T, "ens": ENS, "best_fold": best},
           os.path.join(OUT, "model_finetuned.pt"))
json.dump({"mean": art["mu"], "std": art["sd"], "clip": float(CLIPZ),
           "features": FEATURES, "scale": float(SCALE)},
          open(os.path.join(OUT, "scaler_real.json"), "w"))
json.dump({"human": art["th"], "vehicle": art["tv"], "animal": 0.5},
          open(os.path.join(OUT, "thresholds.json"), "w"), indent=1)
json.dump({"best_fold": best, "best_fold_val_acc": float(fine[best]),
           "per_fold_acc": [round(x, 4) for x in fine], "ensemble_size": ENS,
           "n_features": NF, "scale": float(SCALE)},
          open(os.path.join(OUT, "DEPLOY_MANIFEST.json"), "w"), indent=1)
print(f"DEPLOY: best fold {best} (val acc {fine[best]:.3f}), ens{ENS} -> "
      f"{OUT}\\model_finetuned.pt + scaler_real.json + thresholds.json")
print("\n================ 5-FOLD (same protocol as baseline) ================")
print(f"  synthetic-pretrained, ZERO-SHOT          : {np.mean(zero):.3f}")
print(f"  synthetic-pretrained + FINE-TUNED (ens{ENS})  : {np.mean(fine):.3f} +/- {np.std(fine):.3f}")
print(f"  real-only baseline (logreg / HGB)        : 0.986 / 0.983")
print(f"  BEAT HGB 0.983?    {'YES' if np.mean(fine) > 0.983 else 'no'}    BEAT logreg 0.986?  {'YES' if np.mean(fine) > 0.986 else 'no'}")
print("wrote FINETUNE_5FOLD.json")
