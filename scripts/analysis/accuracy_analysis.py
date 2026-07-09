"""Accuracy analysis under post-processing/parameter tweaks (no retraining).
Levers explored on the trained synthetic SNN, evaluated as 3-class (h/v/nothing) on REAL:
  1. global decision-threshold sweep  -> accuracy curve + REAL-ORACLE best (upper bound)
  2. per-head thresholds CALIBRATED ON SYNTHETIC VAL, applied to real (leakage-free)
  3. EMA vs raw checkpoint
  4. mean-membrane vs last-step readout (recomputed from v_seq, no retrain)
Clearly labels leakage-free vs oracle. Writes ACCURACY_ANALYSIS.json + plots.
"""
import os, sys, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.path.join(HERE, "snn_v2_out"); PNG = os.path.join(OUT, "plots")
FEAT_DIR = r"G:/geophone_synth/features_v2"; REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, SCENE, HOP = 4, 30 * int(F.FS), 1500
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
        xs = (x * s.gate).unsqueeze(0).repeat(T, 1, 1); hh = s.body(xs)
        for nm, hd in (("human", s.human), ("animal", s.animal), ("vehicle", s.vehicle)): hd(hh)
        return  # readouts hold v_seq

def load(ckpt):
    m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m")
    m.load_state_dict(torch.load(os.path.join(OUT, ckpt), map_location=DEV)); m.eval(); return m

@torch.no_grad()
def vseqs(model, X):                       # -> per-head v_seq [T,N,out] (cpu)
    acc = {"human": [], "animal": [], "vehicle": []}
    for j in range(0, len(X), 8192):
        functional.reset_net(model); model(torch.as_tensor(X[j:j + 8192]).to(DEV))
        for nm in acc: acc[nm].append(getattr(model, nm)[-1].v_seq.cpu())
    return {nm: torch.cat(v, 1) for nm, v in acc.items()}                  # [T, N, out]

def pres(vs, agg):                         # presence prob per head from v_seq
    red = (lambda t: t.mean(0)) if agg == "mean" else (lambda t: t[-1])
    return {nm: torch.sigmoid(red(vs[nm])[:, 0]).numpy() for nm in vs}

def decide3(ph, pv, th, tv):
    mh, mv = ph - th, pv - tv
    return np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
def decide4(ph, pa, pv, th, ta, tv):
    M = np.stack([ph - th, pa - ta, pv - tv], 1); nm = np.array(["human", "animal", "vehicle"])
    return np.where(M.max(1) < 0, "nothing", nm[M.argmax(1)])

def best_thr(p, y):                        # threshold maximizing balanced-acc for presence
    grid = np.linspace(0.05, 0.95, 19)
    return float(grid[np.argmax([balanced_accuracy_score(y, p > t) for t in grid])])

# ---- data: real (h/v/nothing) + synthetic val ----
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
def feat_real(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32); fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    X = np.stack(fe) if fe else np.empty((0, F.NFEAT), np.float32)
    return np.clip((np.nan_to_num(X[:, fidx]) - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
rX, ry = [], []
for fn, cls in REALS.items(): Xr = feat_real(os.path.join(REAL_DIR, fn)); rX.append(Xr); ry += [cls] * len(Xr)
rX = np.vstack(rX); ry = np.array(ry)
LVL = {"none": 0, "single": 1, "multiple": 2}
cols = FEATURES + ["split", "human_level", "vehicle_level", "animal_level"]
dv = pd.concat([pd.read_parquet(s, columns=cols) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))],
               ignore_index=True); dv = dv[dv.split == "val"].reset_index(drop=True)
Xv = np.clip((dv[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ)
def lv(c): return dv[c].map(LVL).fillna(dv[c]).to_numpy().astype(int)
H, V, A = lv("human_level"), lv("vehicle_level"), lv("animal_level")
nact = (H > 0).astype(int) + (V > 0).astype(int) + (A > 0).astype(int); single = nact <= 1
yv4 = np.where(H > 0, "human", np.where(A > 0, "animal", np.where(V > 0, "vehicle", "nothing")))

res = {}
for ckpt, tag in [("model_ema.pt", "EMA"), ("model_raw.pt", "raw")]:
    m = load(ckpt); rvs = vseqs(m, rX); vvs = vseqs(m, Xv)
    for agg in ("mean", "last"):
        rp = pres(rvs, agg); vp = pres(vvs, agg)
        # calibrate per-head thresholds on SYNTHETIC VAL (leakage-free)
        th = best_thr(vp["human"], (H > 0)); tv = best_thr(vp["vehicle"], (V > 0)); ta = best_thr(vp["animal"], (A > 0))
        acc_default = accuracy_score(ry, decide3(rp["human"], rp["vehicle"], .5, .5))
        acc_syncal = accuracy_score(ry, decide3(rp["human"], rp["vehicle"], th, tv))
        # real-oracle: best single global threshold ON real (upper bound)
        grid = np.linspace(0.05, 0.95, 19)
        oracle = max((accuracy_score(ry, decide3(rp["human"], rp["vehicle"], t, t)), t) for t in grid)
        # synthetic-val accuracy with the same calibrated thresholds
        acc_val4 = accuracy_score(yv4[single], decide4(vp["human"][single], vp["animal"][single],
                                                       vp["vehicle"][single], th, ta, tv))
        res[f"{tag}_{agg}"] = {"syn_val_4class_acc": float(acc_val4),
                               "real_acc_default_0.5": float(acc_default),
                               "real_acc_syncal_thresholds": float(acc_syncal),
                               "real_acc_oracle_threshold": float(oracle[0]), "oracle_thr": float(oracle[1]),
                               "thresholds_syncal": {"human": th, "vehicle": tv, "animal": ta}}
        print(f"{tag:3s}/{agg:4s}  syn-val4 {acc_val4:.3f} | real@0.5 {acc_default:.3f} | "
              f"real syn-cal {acc_syncal:.3f} | real oracle {oracle[0]:.3f}@{oracle[1]:.2f}", flush=True)

# accuracy-vs-threshold curve (EMA/mean) on real + confusion at syn-cal
m = load("model_ema.pt"); rvs = vseqs(m, rX); rp = pres(rvs, "mean")
grid = np.linspace(0.05, 0.95, 37)
curve = [accuracy_score(ry, decide3(rp["human"], rp["vehicle"], t, t)) for t in grid]
plt.figure(figsize=(7, 4)); plt.plot(grid, curve, "-o", ms=3)
plt.axvline(0.5, color="r", ls="--", lw=.8, label="default 0.5")
plt.xlabel("global threshold"); plt.ylabel("real 3-class accuracy"); plt.legend()
plt.title("Real accuracy vs decision threshold (EMA, mean readout)"); plt.tight_layout()
plt.savefig(os.path.join(PNG, "70_real_acc_vs_threshold.png"), dpi=120); plt.close()
best = res["EMA_mean"]
pred = decide3(rp["human"], rp["vehicle"], best["thresholds_syncal"]["human"], best["thresholds_syncal"]["vehicle"])
labs = ["human", "vehicle", "nothing"]; cm = confusion_matrix(ry, pred, labels=labs)
plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues")
for (a, b), v in np.ndenumerate(cm): plt.text(b, a, int(v), ha="center", va="center")
plt.xticks(range(3), labs); plt.yticks(range(3), labs); plt.xlabel("pred"); plt.ylabel("true")
plt.title(f"Real confusion (syn-cal, acc {best['real_acc_syncal_thresholds']:.3f})"); plt.tight_layout()
plt.savefig(os.path.join(PNG, "71_real_confusion_syncal.png"), dpi=120); plt.close()

json.dump(res, open(os.path.join(OUT, "ACCURACY_ANALYSIS.json"), "w"), indent=1)
print("\nwrote", os.path.join(OUT, "ACCURACY_ANALYSIS.json"))
