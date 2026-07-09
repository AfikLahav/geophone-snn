"""Does real performance depend on the scene-chunk length used for feature extraction? If yes, the
scene-wide-statistic 'bug' matters and fixing it (consistent sim/real featurization) could help.
Re-featurize real at several chunk lengths, run model_ema (x25), report 3-class acc + human recall.
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import accuracy_score, recall_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.path.join(HERE, "snn_v2_out"); REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, HOP, SCALE = 4, 1500, 25.4
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]; TH = json.load(open(os.path.join(OUT, "ERROR_ANALYSIS.json")))["thresholds"]

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
def probs(X):
    o = {"human": [], "vehicle": []}
    with torch.no_grad():
        for j in range(0, len(X), 4096):
            functional.reset_net(m); r = m(torch.as_tensor(X[j:j + 4096]).to(DEV))
            for nm in o: o[nm].append(torch.sigmoid(r[nm][:, 0]).cpu())
    return {k: np.array(torch.cat(v)) for k, v in o.items()}

REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
def feat(p, chunk):                                  # chunk in seconds; None = whole recording
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE
    seglen = len(a) if chunk is None else chunk * int(F.FS); fe = []
    for c0 in range(0, len(a), seglen):
        seg = a[c0:c0 + seglen]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)

print(f"{'chunk':>8s} {'acc':>6s} {'human_rec':>10s} {'veh_rec':>8s} {'noth_rec':>9s} {'n_win':>6s}")
res = {}
for chunk in (15, 30, 60, 120, None):
    rX, ry = [], []
    for fn, cls in REALS.items():
        Xr = feat(os.path.join(REAL_DIR, fn), chunk)
        rX.append(np.clip((Xr[:, fidx] - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)); ry += [cls] * len(rX[-1])
    rX = np.vstack(rX); ry = np.array(ry); pr = probs(rX)
    mh, mv = pr["human"] - TH["human"], pr["vehicle"] - TH["vehicle"]
    pred = np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
    acc = accuracy_score(ry, pred); rec = {c: recall_score(ry == c, pred == c, zero_division=0) for c in ("human", "vehicle", "nothing")}
    res[str(chunk)] = {"acc": float(acc), "recall": rec, "n": int(len(ry))}
    print(f"{str(chunk):>8s} {acc:6.3f} {rec['human']:10.3f} {rec['vehicle']:8.3f} {rec['nothing']:9.3f} {len(ry):6d}", flush=True)
json.dump(res, open(os.path.join(OUT, "CHUNK_SENSITIVITY.json"), "w"), indent=1)
accs = [v["acc"] for v in res.values()]
print(f"\nacc range across chunk lengths: {min(accs):.3f}-{max(accs):.3f} (spread {max(accs)-min(accs):.3f})")
print("If spread is large -> scene-length featurization matters -> fixing the bug could help.")
