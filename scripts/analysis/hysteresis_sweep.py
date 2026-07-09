"""Post-hysteresis comparison: synthetic-pretrained+finetuned SNN vs same-SNN-from-scratch-on-real,
under a matched hysteresis sweep. Collects out-of-fold per-window presence probs for BOTH models
(5-fold, ens5), then applies hysteresis per recording (entry per head + exit=factor*entry),
sweeping the factor f (f=1.0 = no hysteresis = the raw 0.977/0.966). Same treatment to both models.
"""
import os, sys, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.path.join(HERE, "snn_v2_out"); REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, SCENE, HOP, K, SCALE, CLIPZ = 4, 30 * int(F.FS), 1500, 5, 25.4, 8.0
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
X, y, blk, src = [], [], [], []
for fn, cls in REALS.items():
    Xr = feat(os.path.join(REAL_DIR, fn))[:, fidx]; n = len(Xr); X.append(Xr.astype(np.float32))
    y += [cls] * n; blk.append(np.arange(n) * K // max(n, 1)); src += [fn] * n
X = np.vstack(X); y = np.array(y); blk = np.concatenate(blk); src = np.array(src)
yh = (y == "human").astype(np.float32); yv = (y == "vehicle").astype(np.float32)

def probs(model, Xz):
    model.eval(); ph, pv = [], []
    with torch.no_grad():
        for j in range(0, len(Xz), 4096):
            functional.reset_net(model); o = model(torch.as_tensor(Xz[j:j + 4096]).to(DEV))
            ph.append(torch.sigmoid(o["human"][:, 0]).cpu()); pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
    return np.array(torch.cat(ph)), np.array(torch.cat(pv))
def train(model, Xz, tr, epochs, lr, seed):
    torch.manual_seed(seed); Xt = torch.as_tensor(Xz[tr]).to(DEV); ht = torch.as_tensor(yh[tr]).to(DEV); vt = torch.as_tensor(yv[tr]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); model.train(); n = len(Xt)
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + np.cos(np.pi * ep / epochs))
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, 256):
            idx = perm[j:j + 256]; functional.reset_net(model); o = model(Xt[idx])
            loss = Fnn.binary_cross_entropy_with_logits(o["human"][:, 0], ht[idx]) + Fnn.binary_cross_entropy_with_logits(o["vehicle"][:, 0], vt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

from sklearn.preprocessing import StandardScaler
ENS = 5
oof = {m: {"ph": np.full(len(X), np.nan), "pv": np.full(len(X), np.nan)} for m in ("scratch", "ours")}
for k in range(K):
    te = blk == k; tr = ~te
    sc = StandardScaler().fit(X[tr]); Xz = np.clip(sc.transform(X), -CLIPZ, CLIPZ).astype(np.float32)
    accum = {m: [np.zeros(te.sum()), np.zeros(te.sum())] for m in ("scratch", "ours")}
    for s in range(ENS):
        m0 = FeatureSNN(NF).to(DEV); functional.set_step_mode(m0, "m"); m0 = train(m0, Xz, tr, 200, 5e-4, s)
        a, b = probs(m0, Xz[te]); accum["scratch"][0] += a; accum["scratch"][1] += b
        m1 = FeatureSNN(NF).to(DEV); functional.set_step_mode(m1, "m")
        m1.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); m1 = train(m1, Xz, tr, 80, 4e-4, s)
        a, b = probs(m1, Xz[te]); accum["ours"][0] += a; accum["ours"][1] += b
    for m in ("scratch", "ours"):
        oof[m]["ph"][te] = accum[m][0] / ENS; oof[m]["pv"][te] = accum[m][1] / ENS
    print(f"fold {k} done", flush=True)

def hyst(p, entry, exit_):
    on = False; out = np.zeros(len(p), bool)
    for i, v in enumerate(p):
        on = (v >= entry) if not on else (v >= exit_)
        out[i] = on
    return out
def acc_at(model, eh, ev, f):                # entry per head, exit = f*entry, hysteresis per recording
    ph, pv = oof[model]["ph"], oof[model]["pv"]; pred = np.empty(len(X), object)
    for fn in REALS:
        idx = np.where(src == fn)[0]                       # time-ordered windows of this recording
        ho = hyst(ph[idx], eh, f * eh); vo = hyst(pv[idx], ev, f * ev)
        for j, gi in enumerate(idx):
            pred[gi] = "nothing" if not (ho[j] or vo[j]) else ("human" if (ho[j] and (not vo[j] or ph[gi] >= pv[gi])) else "vehicle")
    return accuracy_score(y, pred)

# entry per head = balanced-acc-optimal on OOF (same for both models, fair)
def bestentry(p, t): grid = np.linspace(0.05, 0.95, 19); return float(grid[np.argmax([balanced_accuracy_score(t, p > x) for x in grid])])
res = {}
print(f"\n{'factor f':>9s} {'ours_acc':>9s} {'scratch_acc':>12s} {'margin':>8s}  (f=1.0 = no hysteresis)")
for f in (1.0, 0.7, 0.5, 0.3, 0.15):
    eh_o = bestentry(oof["ours"]["ph"], yh > 0); ev_o = bestentry(oof["ours"]["pv"], yv > 0)
    eh_s = bestentry(oof["scratch"]["ph"], yh > 0); ev_s = bestentry(oof["scratch"]["pv"], yv > 0)
    ao = acc_at("ours", eh_o, ev_o, f); as_ = acc_at("scratch", eh_s, ev_s, f)
    res[f"f_{f}"] = {"ours": ao, "scratch": as_, "margin": ao - as_}
    print(f"{f:9.2f} {ao:9.3f} {as_:12.3f} {ao-as_:+8.3f}", flush=True)
json.dump(res, open(os.path.join(OUT, "HYSTERESIS_SWEEP.json"), "w"), indent=1)
print("\nwrote HYSTERESIS_SWEEP.json")
