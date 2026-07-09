"""The CORRECT baseline: the SAME SNN architecture trained FROM SCRATCH on real only (5-fold),
vs our synthetic-pretrained (+fine-tuned) SNN. Apples-to-apples — isolates the value of synthetic
pretraining. Same contiguous 5-fold, per-fold real StandardScaler, x25 scale, 3-class (h/v/nothing).
"""
import os, sys, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.environ.get("GEO_OUT") or os.path.join(HERE, "snn_v2_out"); REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, SCENE, HOP, K, CLIPZ = 4, 30 * int(F.FS), 1500, 5, 8.0
SCALE = float(os.environ.get("GEO_REAL_SCALE", "25.4"))     # v4.1: 1000 = true mV (physical axis)
torch.manual_seed(0); np.random.seed(0)
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
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
    Xr = feat(os.path.join(REAL_DIR, fn))[:, fidx]; n = len(Xr); X.append(Xr.astype(np.float32))
    y += [cls] * n; blk.append(np.arange(n) * K // max(n, 1))
X = np.vstack(X); y = np.array(y); blk = np.concatenate(blk)
yh = (y == "human").astype(np.float32); yv = (y == "vehicle").astype(np.float32)

def probs(model, Xa):
    model.eval(); ph, pv = [], []
    with torch.no_grad():
        for j in range(0, len(Xa), 4096):
            functional.reset_net(model); o = model(torch.as_tensor(Xa[j:j + 4096]).to(DEV))
            ph.append(torch.sigmoid(o["human"][:, 0]).cpu()); pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
    return np.array(torch.cat(ph)), np.array(torch.cat(pv))
def thr(p, t): g = np.linspace(0.05, 0.95, 19); return float(g[np.argmax([balanced_accuracy_score(t, p > x) for x in g])])
def decide(ph, pv, th, tv):
    mh, mv = ph - th, pv - tv; return np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
def train(model, Xz, tr, epochs, lr, seed):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xz[tr]).to(DEV); ht = torch.as_tensor(yh[tr]).to(DEV); vt = torch.as_tensor(yv[tr]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); model.train(); n = len(Xt)
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + np.cos(np.pi * ep / epochs))
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, 256):
            idx = perm[j:j + 256]; functional.reset_net(model); o = model(Xt[idx])
            loss = (Fnn.binary_cross_entropy_with_logits(o["human"][:, 0], ht[idx])
                    + Fnn.binary_cross_entropy_with_logits(o["vehicle"][:, 0], vt[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
    return model

ENS = 5
scratch, finetune = [], []
for k in range(K):
    te = blk == k; tr = ~te
    sc = StandardScaler().fit(X[tr]); Xz = np.clip(sc.transform(X), -CLIPZ, CLIPZ).astype(np.float32)
    ps, pf = [np.zeros(len(X)), np.zeros(len(X))], [np.zeros(len(X)), np.zeros(len(X))]
    for s in range(ENS):
        # (a) from-scratch SNN on real (THE baseline) — more epochs since no pretrain
        m0 = FeatureSNN(NF).to(DEV); functional.set_step_mode(m0, "m")
        m0 = train(m0, Xz, tr, epochs=200, lr=5e-4, seed=s); a, b = probs(m0, Xz); ps[0] += a; ps[1] += b
        # (b) synthetic-pretrained + fine-tuned (our model)
        m1 = FeatureSNN(NF).to(DEV); functional.set_step_mode(m1, "m")
        m1.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV))
        m1 = train(m1, Xz, tr, epochs=80, lr=4e-4, seed=s); a, b = probs(m1, Xz); pf[0] += a; pf[1] += b
    for tag, p, acc in [("scratch", ps, scratch), ("finetune", pf, finetune)]:
        ph, pv = p[0] / ENS, p[1] / ENS
        th, tv = thr(ph[tr], yh[tr] > 0), thr(pv[tr], yv[tr] > 0)
        acc.append(accuracy_score(y[te], decide(ph[te], pv[te], th, tv)))
    print(f"fold {k}: SNN-from-scratch-on-real {scratch[-1]:.3f} | synth-pretrained+finetuned {finetune[-1]:.3f}", flush=True)

res = {"SNN_from_scratch_on_real_5fold": float(np.mean(scratch)), "SNN_scratch_std": float(np.std(scratch)),
       "synth_pretrained_finetuned_5fold": float(np.mean(finetune)), "finetune_std": float(np.std(finetune)),
       "logreg_ref": 0.986, "hgb_ref": 0.983, "ensemble": ENS}
json.dump(res, open(os.path.join(OUT, "SNN_REAL_BASELINE.json"), "w"), indent=1)
print("\n======== SAME-ARCHITECTURE comparison (5-fold, ens5) ========")
print(f"  SNN trained from scratch on real (BASELINE) : {np.mean(scratch):.3f} +/- {np.std(scratch):.3f}")
print(f"  SNN synthetic-pretrained + fine-tuned (OURS) : {np.mean(finetune):.3f} +/- {np.std(finetune):.3f}")
print(f"  -> synthetic pretraining gain: {np.mean(finetune)-np.mean(scratch):+.3f}")
print(f"  BEAT the SNN-on-real baseline? {'YES' if np.mean(finetune) > np.mean(scratch) else 'no'}")
print("wrote SNN_REAL_BASELINE.json")
