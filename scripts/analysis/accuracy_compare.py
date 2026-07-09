"""Single comparable accuracy for both models.
  - Synthetic SNN on synthetic val: 4-class (h/a/v/nothing) and 3-class (h/v/nothing).
  - Synthetic SNN on the REAL data: 3-class (h/v/nothing) — same label space as the baseline.
  - Real-only baseline: 3-class real 5-fold (from real_baseline.json).
Decision: presence sigmoids per head; nothing if max < THR, else argmax. THR=0.5 (uncalibrated;
animal head is miscalibrated at 0.5 — noted). Writes ACCURACY_COMPARE.json + prints a table.
"""
import os, sys, json, glob, sqlite3
import numpy as np, pandas as pd, torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.path.join(HERE, "snn_v2_out"); FEAT_DIR = r"G:/geophone_synth/features_v2"
REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data"); DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, THR, SCENE, HOP = 4, 0.5, 30 * int(F.FS), 1500
sc = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = sc["features"]; NF = len(FEATURES)
mu = np.array(sc["mean"], np.float32); sd = np.array(sc["std"], np.float32); CLIPZ = sc["clip"]
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

net = FeatureSNN(NF).to(DEV); functional.set_step_mode(net, "m")
net.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); net.eval()

@torch.no_grad()
def presence(X):                          # -> dict of presence prob arrays
    ph, pa, pv = [], [], []
    for j in range(0, len(X), 8192):
        functional.reset_net(net); o = net(torch.as_tensor(X[j:j + 8192]).to(DEV))
        ph.append(torch.sigmoid(o["human"][:, 0]).cpu()); pa.append(torch.sigmoid(o["animal"][:, 0]).cpu())
        pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
    return np.array(torch.cat(ph)), np.array(torch.cat(pa)), np.array(torch.cat(pv))

def decide(scores, names):                # nothing if all<THR else argmax
    S = np.stack(scores, 1); pred = np.where(S.max(1) < THR, "nothing", np.array(names)[S.argmax(1)])
    return pred

# ---------- synthetic val ----------
LVL = {"none": 0, "single": 1, "multiple": 2}
cols = FEATURES + ["split", "coarse", "human_level", "vehicle_level", "animal_level"]
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))],
               ignore_index=True)
df = df[df.split == "val"].reset_index(drop=True)
Xv = np.clip((df[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ)
def lvl(c): return df[c].map(LVL).fillna(df[c]).to_numpy().astype(int)
H, A, V = lvl("human_level"), lvl("vehicle_level"), lvl("animal_level")  # NB order
H, V, A = lvl("human_level"), lvl("vehicle_level"), lvl("animal_level")
ph, pa, pv = presence(Xv)
# true single-class label (exclude mixed/co-occurring for a clean N-class accuracy)
nact = (H > 0).astype(int) + (V > 0).astype(int) + (A > 0).astype(int)
single = nact <= 1
def truth4(i): return "human" if H[i] > 0 else "vehicle" if V[i] > 0 else "animal" if A[i] > 0 else "nothing"
yv = np.array([truth4(i) for i in range(len(df))])
m4 = single
acc4 = float((decide([ph[m4], pa[m4], pv[m4]], ["human", "animal", "vehicle"]) == yv[m4]).mean())
m3 = single & (A == 0)                                              # h/v/nothing only
acc3 = float((decide([ph[m3], pv[m3]], ["human", "vehicle"]) == yv[m3]).mean())

# ---------- synthetic model on REAL (3-class) ----------
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
for fn, cls in REALS.items():
    Xr = feat_real(os.path.join(REAL_DIR, fn)); rX.append(Xr); ry += [cls] * len(Xr)
rX = np.vstack(rX); ry = np.array(ry)
rph, rpa, rpv = presence(rX)
real_pred = decide([rph, rpv], ["human", "vehicle"])               # animal head excluded (no real animals)
real_acc = float((real_pred == ry).mean())
real_anim_ff = float((rpa[ry == "nothing"] > THR).mean())          # animal-head false-fire on real nothing (diagnostic)

base = json.load(open(os.path.join(OUT, "real_baseline.json")))
out = {"threshold": THR,
       "synthetic_model": {"synthetic_val_4class_acc": acc4, "synthetic_val_3class_hvn_acc": acc3,
                           "REAL_3class_hvn_acc": real_acc, "animal_head_falsefire_on_real_nothing": real_anim_ff},
       "real_only_baseline_5fold": {"3class_logreg_acc": base["3class_logreg"]["acc_mean"],
                                    "3class_hgb_acc": base["3class_hgb"]["acc_mean"],
                                    "3class_hgb_robustfeat_acc": base["3class_hgb_robustfeat"]["acc_mean"]},
       "note": "THR=0.5 uncalibrated (animal head miscalibrated at 0.5 -> 4class understated); baseline real numbers are session-confounded/optimistic."}
json.dump(out, open(os.path.join(OUT, "ACCURACY_COMPARE.json"), "w"), indent=1)
print("================ ACCURACY ================")
print(f"SYNTHETIC SNN  | synthetic-val 4-class (h/a/v/nothing): {acc4:.3f}")
print(f"               | synthetic-val 3-class (h/v/nothing)  : {acc3:.3f}")
print(f"               | REAL 3-class (h/v/nothing, never seen): {real_acc:.3f}   (animal-head false-fire on real nothing: {real_anim_ff:.3f})")
print(f"REAL BASELINE  | real 5-fold 3-class logreg/HGB        : {base['3class_logreg']['acc_mean']:.3f} / {base['3class_hgb']['acc_mean']:.3f}  (confounded)")
print(f"               | real 5-fold 3-class HGB robust-feat   : {base['3class_hgb_robustfeat']['acc_mean']:.3f}  (confound-clean)")
print("wrote", os.path.join(OUT, "ACCURACY_COMPARE.json"))
