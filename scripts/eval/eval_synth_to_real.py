"""PURE synthetic->real transfer: the synthetic-pretrained model evaluated on the real
recordings with ZERO real data used for training OR threshold selection.

Two kinds of numbers:
  1. Threshold-FREE AUROC (no calibration at all -> the cleanest transfer measure):
       - human-head presence: real human windows vs real nothing windows
       - vehicle-head presence: real car windows vs real nothing windows
       - human-vs-vehicle: real human vs real car (score = p_human)
  2. 3-way accuracy using thresholds calibrated ONLY on the SYNTHETIC val set (never real):
       per-head tau set so synthetic-val FAR-on-nothing = 1%, predict argmax>tau else nothing.
Scaler = the synthetic training scaler (scaler.json). No real data leaks anywhere.

Usage: python dataset_validation/eval_synth_to_real.py <GEO_OUT_dir> <synth_feat_dir>
"""
import os, sys, json, glob
import numpy as np, pandas as pd
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4")); import features as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T = 4
SCALE = float(os.environ.get("GEO_REAL_SCALE", "25.4"))     # v4.1: 1000 = true mV (physical axis)
SCENE, HOP = 30 * int(F.FS), 1500
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "snn_v4_out", "E1_coupling_bump")
SFEAT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_GEO_ROOT, "features_v4b_3s")
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATS = scj["features"]; NF = len(FEATS)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATS]


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


def probs(model, Xz):
    model.eval(); ph, pv = [], []
    with torch.no_grad():
        for j in range(0, len(Xz), 4096):
            functional.reset_net(model); o = model(torch.as_tensor(Xz[j:j + 4096]).to(DEV))
            ph.append(torch.sigmoid(o["human"][:, 0]).cpu()); pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
    return np.array(torch.cat(ph)), np.array(torch.cat(pv))


def feat_real(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE; fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.stack(fe)[:, fidx] if fe else np.empty((0, len(fidx)), np.float32)


# real windows + labels
RX, RY = [], []
for fn, cls in REALS.items():
    X = feat_real(os.path.join(REAL_DIR, fn)); RX.append(X); RY += [cls] * len(X)
RX = np.nan_to_num(np.vstack(RX)); RY = np.array(RY)
RXz = np.clip((RX - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m")
m.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV))
ph, pv = probs(m, RXz)
print(f"model: {OUT}")
print(f"real windows: { {c: int((RY==c).sum()) for c in ['human','vehicle','nothing']} }\n")

rep = {"model": OUT, "threshold_free_auroc": {}, "synth_calibrated_3way": {}}
# --- threshold-free AUROC (pure) ---
hn = np.isin(RY, ["human", "nothing"]); y = (RY[hn] == "human").astype(int)
rep["threshold_free_auroc"]["human_vs_nothing"] = round(float(roc_auc_score(y, ph[hn])), 4)
vn = np.isin(RY, ["vehicle", "nothing"]); y = (RY[vn] == "vehicle").astype(int)
rep["threshold_free_auroc"]["vehicle_vs_nothing"] = round(float(roc_auc_score(y, pv[vn])), 4)
hv = np.isin(RY, ["human", "vehicle"]); y = (RY[hv] == "human").astype(int)
rep["threshold_free_auroc"]["human_vs_vehicle"] = round(float(roc_auc_score(y, ph[hv])), 4)
print("=== threshold-FREE AUROC on real (no calibration, pure transfer) ===")
for k, v in rep["threshold_free_auroc"].items(): print(f"  {k:20s} {v:.4f}")

# --- synthetic-calibrated thresholds (tau from SYNTH val nothing, FAR=1%; never touches real) ---
sh = sorted(glob.glob(os.path.join(SFEAT, "features_shard_*.parquet")))[:4]
sv = pd.concat([pd.read_parquet(s, columns=FEATS + ["split", "coarse"]) for s in sh], ignore_index=True)
sv = sv[(sv.split == "val") & (sv.coarse == "nothing")]
Xsv = np.clip((np.nan_to_num(sv[FEATS].to_numpy(np.float32)) - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
psh, psv = probs(m, Xsv)
tau_h = float(np.quantile(psh, 0.99)); tau_v = float(np.quantile(psv, 0.99))   # 1% synthetic FAR
pred = np.where((ph < tau_h) & (pv < tau_v), "nothing",
                np.where(ph - tau_h >= pv - tau_v, "human", "vehicle"))
acc = float(accuracy_score(RY, pred)); bal = float(balanced_accuracy_score(RY, pred))
cm = confusion_matrix(RY, pred, labels=["human", "vehicle", "nothing"])
rep["synth_calibrated_3way"] = {"tau_human": round(tau_h, 3), "tau_vehicle": round(tau_v, 3),
    "accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
    "recall": {c: round(float((pred[RY == c] == c).mean()), 3) for c in ["human", "vehicle", "nothing"]},
    "confusion_hvn": cm.tolist()}
print("\n=== 3-way accuracy, SYNTHETIC-calibrated thresholds (no real leakage) ===")
print(f"  tau_h {tau_h:.3f} tau_v {tau_v:.3f} | acc {acc:.4f} bal_acc {bal:.4f}")
print(f"  recall: {rep['synth_calibrated_3way']['recall']}")
print(f"  confusion [human,vehicle,nothing]:\n{cm}")
json.dump(rep, open(os.path.join(OUT, "SYNTH_TO_REAL.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(OUT, 'SYNTH_TO_REAL.json')}")
