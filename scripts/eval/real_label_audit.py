"""Why is pure-transfer ACCURACY ~0.86 while AUROC is 0.87-0.99? Decompose into:
  (1) REAL LABEL NOISE: session-level labels mark every car.csv window 'vehicle' even between
      passes. Measure each window's in-band RMS vs the PAIRED nothing-session ambient floor;
      windows indistinguishable from ambient are presence-labeled silence (the same
      presence!=detectability defect v3.1 fixed in synthetic).
  (2) CALIBRATION LOSS: synthetic-calibrated thresholds vs ORACLE thresholds (best possible on
      real) -> how much accuracy the threshold transfer costs, given fixed ranking.
  (3) RESIDUAL RANKING LOSS: what remains with oracle thresholds on activity-clean labels.
Reports accuracy under: as-is labels / sub-floor-excluded / sub-floor-relabeled-nothing,
each with synthetic and oracle thresholds. Model: E0 (v4 low-pass).
"""
import os, sys, json
import numpy as np, pandas as pd
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4"))
import features as F, label as L
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T = 4
SCALE = float(os.environ.get("GEO_REAL_SCALE", "25.4"))     # v4.1: 1000 = true mV (physical axis)
SCENE, HOP = 30 * int(F.FS), 1500
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "snn_v4_out", "E0_win3s")
PAIRS = {"car.csv": ("vehicle", "car_nothing.csv"), "human.csv": ("human", "human_nothing.csv")}
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


def windows_of(path):
    a = pd.read_csv(path)["amplitude"].to_numpy(np.float32) * SCALE
    wins, rms = [], {}
    feats = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP):
            feats.append(F.window_features(pre, i0).astype(np.float32))
            w = seg[i0:i0 + F.NW].astype(np.float64)
            wins.append({b: L._band_rms(w, *L.BANDS[b]) for b in ("human", "vehicle")})
    return np.stack(feats)[:, fidx], wins


m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m")
m.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); m.eval()

X_all, Y_all, act_all = [], [], []
floors = {}
for noth in ("car_nothing.csv", "human_nothing.csv"):
    Xn, wn = windows_of(os.path.join(REAL_DIR, noth))
    floors[noth] = {b: np.percentile([w[b] for w in wn], 90) for b in ("human", "vehicle")}
    X_all.append(Xn); Y_all += ["nothing"] * len(Xn); act_all += [True] * len(Xn)
for fn, (cls, noth) in PAIRS.items():
    Xf, wf = windows_of(os.path.join(REAL_DIR, fn))
    band = cls
    fl = floors[noth][band]
    active = [w[band] > fl for w in wf]
    X_all.append(Xf); Y_all += [cls] * len(Xf); act_all += active
    n_sub = len(active) - sum(active)
    print(f"{fn}: {len(active)} windows | ABOVE ambient floor {sum(active)} | "
          f"indistinguishable-from-ambient {n_sub} ({n_sub/len(active)*100:.0f}%)")
X = np.nan_to_num(np.vstack(X_all)); Y = np.array(Y_all); ACT = np.array(act_all)
Xz = np.clip((X - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
ph, pv = [], []
with torch.no_grad():
    for j in range(0, len(Xz), 4096):
        functional.reset_net(m); o = m(torch.as_tensor(Xz[j:j + 4096]).to(DEV))
        ph.append(torch.sigmoid(o["human"][:, 0]).cpu()); pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
ph = np.array(torch.cat(ph)); pv = np.array(torch.cat(pv))


def acc3(y, th, tv):
    pred = np.where((ph < th) & (pv < tv), "nothing", np.where(ph - th >= pv - tv, "human", "vehicle"))
    return pred


def report(tag, mask, y):
    p_syn = acc3(y, TAU_H, TAU_V)[mask]
    # oracle thresholds: grid-search best on THIS labeling (upper bound of threshold placement)
    best = (0, None)
    for th in np.quantile(ph, np.linspace(0.3, 0.995, 40)):
        for tv in np.quantile(pv, np.linspace(0.3, 0.995, 40)):
            a = accuracy_score(y[mask], acc3(y, th, tv)[mask])
            if a > best[0]: best = (a, (th, tv))
    a_syn = accuracy_score(y[mask], p_syn)
    print(f"{tag:44s} synth-cal acc {a_syn:.4f} | ORACLE acc {best[0]:.4f}")
    return a_syn, best[0]


rep = json.load(open(os.path.join(OUT, "SYNTH_TO_REAL.json")))
TAU_H = rep["synth_calibrated_3way"]["tau_human"]; TAU_V = rep["synth_calibrated_3way"]["tau_vehicle"]
print(f"\nthresholds: synth-calibrated tau_h={TAU_H} tau_v={TAU_V}\n")
all_mask = np.ones(len(Y), bool)
r1 = report("(1) labels AS-IS (session-level)", all_mask, Y)
r2 = report("(2) sub-floor present windows EXCLUDED", ACT, Y)
Y_rel = Y.copy(); Y_rel[(~ACT)] = "nothing"
r3 = report("(3) sub-floor present RELABELED nothing", all_mask, Y_rel)
res = {"floors_pct90_ambient": {k: {b: float(v) for b, v in d.items()} for k, d in floors.items()},
       "as_is": {"synth": r1[0], "oracle": r1[1]}, "excluded": {"synth": r2[0], "oracle": r2[1]},
       "relabeled": {"synth": r3[0], "oracle": r3[1]}}
json.dump(res, open(os.path.join(HERE, "REAL_LABEL_AUDIT.json"), "w"), indent=1)
print("\nwrote REAL_LABEL_AUDIT.json")
