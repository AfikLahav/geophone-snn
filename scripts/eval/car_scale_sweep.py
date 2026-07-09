"""Gain-mismatch test: the real car session was recorded on a DIFFERENT setup than the human
session (no coupling ring; unknown gain), yet the pipeline applies one global x25.4 scale to
all real data. If the car session's effective gain differs, its windows sit at the wrong
amplitude -> the vehicle head (whose top domain features are amplitude/energy) mis-ranks them.

Sweep the car-session scale and measure vehicle_vs_nothing AUROC + synthetic-threshold recall.
A strong peak away from 25.4 = gain misalignment confirmed (an ALIGNMENT fix, not a model fix).
Model: E0 (v4 low-pass).
"""
import os, sys, json
import numpy as np, pandas as pd
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4")); import features as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T = 4
SCENE, HOP = 30 * int(F.FS), 1500
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
OUT = os.path.join(ROOT, "snn_v4_out", "E0_win3s")
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


m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m")
m.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); m.eval()
raw = {fn: pd.read_csv(os.path.join(REAL_DIR, fn))["amplitude"].to_numpy(np.float32)
       for fn in ("car.csv", "car_nothing.csv")}


def scores(a, scale):
    fe = []
    x = a * scale
    for c0 in range(0, len(x), SCENE):
        seg = x[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    X = np.nan_to_num(np.stack(fe))[:, fidx]
    Xz = np.clip((X - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
    pv = []
    with torch.no_grad():
        for j in range(0, len(Xz), 4096):
            functional.reset_net(m); o = m(torch.as_tensor(Xz[j:j + 4096]).to(DEV))
            pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
    return np.array(torch.cat(pv))


TAU_V = json.load(open(os.path.join(OUT, "SYNTH_TO_REAL.json")))["synth_calibrated_3way"]["tau_vehicle"]
res = {}
print(f"tau_v (synth-calibrated) = {TAU_V}\n scale | veh_vs_noth AUROC | veh recall@tau | noth FA@tau | med(p_veh|car)")
for s in (5.0, 10.0, 16.0, 25.4, 40.0, 64.0, 100.0, 160.0):
    pc = scores(raw["car.csv"], s); pn = scores(raw["car_nothing.csv"], s)
    y = np.r_[np.ones(len(pc)), np.zeros(len(pn))]; p = np.r_[pc, pn]
    auc = float(roc_auc_score(y, p))
    res[s] = {"auroc": round(auc, 4), "recall": round(float((pc >= TAU_V).mean()), 3),
              "fa": round(float((pn >= TAU_V).mean()), 3), "med_p": round(float(np.median(pc)), 3)}
    print(f" {s:5.1f} | {auc:.4f}            | {res[s]['recall']:.3f}          | {res[s]['fa']:.3f}       | {res[s]['med_p']:.3f}")
best = max(res, key=lambda k: res[k]["auroc"])
print(f"\nbest scale by AUROC: {best} (current pipeline: 25.4)")
json.dump({"tau_v": TAU_V, "sweep": {str(k): v for k, v in res.items()}, "best_scale": best},
          open(os.path.join(HERE, "CAR_SCALE_SWEEP.json"), "w"), indent=1)
print("wrote CAR_SCALE_SWEEP.json")
