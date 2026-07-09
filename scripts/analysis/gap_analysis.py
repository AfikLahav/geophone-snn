"""Is there a DATASET GAP behind the human faint-signal blindness? Locate the real missed-human
windows inside the SYNTHETIC feature distribution.

Synthetic groups (model z-space): faint-human (present, human_snr in [0,8] dB), clear-human (snr>8),
nothing (all-none). Real groups: missed-human (true human -> pred nothing), detected-human.
  - nearest-synthetic-neighbor class of each real window (faint-human / clear-human / nothing)
  - per-feature: where real-missed-human sits relative to the synth-faint-human distribution
    (|z| within synth-faint-human) -> features where real-missed-human is OUT of synthetic coverage
This says whether the simulator lacks examples like real faint human, and which physics to fix.
Writes GAP_ANALYSIS.json + a PCA plot. No tuning on real (diagnosis only).
"""
import os, sys, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate, layer, functional
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.path.join(HERE, "snn_v2_out"); PNG = os.path.join(OUT, "plots")
FEAT_DIR = r"G:/geophone_synth/features_v2"; REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, SCENE, HOP = 4, 30 * int(F.FS), 1500
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]; TH = json.load(open(os.path.join(OUT, "ERROR_ANALYSIS.json")))["thresholds"]
def Z(Xraw): return np.clip((Xraw - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)


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
m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m")
m.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); m.eval()
@torch.no_grad()
def hp(X):
    o = {"human": [], "vehicle": []}
    for j in range(0, len(X), 8192):
        functional.reset_net(m); m(torch.as_tensor(X[j:j + 8192]).to(DEV))
        for nm in o: o[nm].append(torch.sigmoid(getattr(m, nm)[-1].v_seq[-1][:, 0]).cpu())
    return {k: np.array(torch.cat(v)) for k, v in o.items()}

# ---- synthetic groups (sample) ----
cols = FEATURES + ["split", "human_level", "human_snr", "vehicle_level", "animal_level"]
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))[:3]],
               ignore_index=True)
LVL = {"none": 0, "single": 1, "multiple": 2}
hl = df["human_level"].map(LVL).fillna(df["human_level"]).to_numpy().astype(int)
vl = df["vehicle_level"].map(LVL).fillna(df["vehicle_level"]).to_numpy().astype(int)
al = df["animal_level"].map(LVL).fillna(df["animal_level"]).to_numpy().astype(int)
snr = df["human_snr"].to_numpy(float)
Xs = df[FEATURES].to_numpy(np.float32)
faint = (hl > 0) & (vl == 0) & (al == 0) & (snr >= 0) & (snr <= 8)
clear = (hl > 0) & (vl == 0) & (al == 0) & (snr > 8)
noth = (hl == 0) & (vl == 0) & (al == 0)
rng = np.random.default_rng(0)
def samp(mask, n): idx = np.where(mask)[0]; return idx[rng.choice(len(idx), min(n, len(idx)), replace=False)]
G = {"faint_human": samp(faint, 8000), "clear_human": samp(clear, 8000), "nothing": samp(noth, 8000)}
Zsyn = {k: Z(Xs[v]) for k, v in G.items()}
print("synthetic group sizes:", {k: len(v) for k, v in Zsyn.items()},
      "| faint-human SNR median %.1f dB" % np.median(snr[faint]))

# ---- real human: detected vs missed ----
def feat_real(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32); fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)
rawH = feat_real(os.path.join(REAL_DIR, "human.csv"))
Zr = Z(rawH[:, fidx]); pr = hp(Zr)
mh, mv = pr["human"] - TH["human"], pr["vehicle"] - TH["vehicle"]
predH = np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
real_missed = Zr[predH == "nothing"]; real_det = Zr[predH == "human"]
print(f"real human: detected {len(real_det)}  missed {len(real_missed)}")

# ---- nearest-synthetic-neighbor class of real windows ----
from sklearn.neighbors import KNeighborsClassifier
Xtr = np.vstack([Zsyn[k] for k in G]); ytr = np.concatenate([[k] * len(Zsyn[k]) for k in G])
knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr)
nn_missed = pd.Series(knn.predict(real_missed)).value_counts(normalize=True).round(3).to_dict()
nn_det = pd.Series(knn.predict(real_det)).value_counts(normalize=True).round(3).to_dict()

# ---- per-feature: is real-missed-human OUT of the synth-faint-human range? ----
fmu, fsd = Zsyn["faint_human"].mean(0), Zsyn["faint_human"].std(0) + 1e-9
z_real_missed = (real_missed.mean(0) - fmu) / fsd            # how many synth-faint-human std away
gap = sorted(zip(FEATURES, z_real_missed), key=lambda t: -abs(t[1]))
report = {"faint_human_snr_band_dB": [0, 8], "synth_faint_human_median_snr": float(np.median(snr[faint])),
          "real_missed_nearest_synth_class": nn_missed, "real_detected_nearest_synth_class": nn_det,
          "top_gap_features": [{"feature": f, "z_vs_synth_faint_human": float(z)} for f, z in gap[:15]]}
json.dump(report, open(os.path.join(OUT, "GAP_ANALYSIS.json"), "w"), indent=1)

# PCA plot: synth groups + real missed/detected
from sklearn.decomposition import PCA
pca = PCA(2).fit(Xtr)
plt.figure(figsize=(7.5, 6))
for k, c in [("nothing", "0.6"), ("faint_human", "tab:orange"), ("clear_human", "tab:green")]:
    p = pca.transform(Zsyn[k]); plt.scatter(p[:, 0], p[:, 1], s=4, alpha=.25, c=c, label="synth " + k)
pm = pca.transform(real_missed); pd_ = pca.transform(real_det)
plt.scatter(pm[:, 0], pm[:, 1], s=14, c="red", marker="x", label="REAL missed-human")
plt.scatter(pd_[:, 0], pd_[:, 1], s=14, c="blue", marker="+", label="REAL detected-human")
plt.legend(markerscale=2, fontsize=8); plt.title("Real faint human vs synthetic feature space (PCA)")
plt.tight_layout(); plt.savefig(os.path.join(PNG, "75_gap_pca.png"), dpi=120); plt.close()

print("\n=== GAP ANALYSIS ===")
print("real MISSED-human nearest synthetic class:", nn_missed)
print("real DETECTED-human nearest synthetic class:", nn_det)
print("top features where real-missed-human is OUT of synth-faint-human range (z vs synth-faint-human):")
for f, z in gap[:12]: print(f"   {f:22s} {z:+.2f}")
print("\nwrote GAP_ANALYSIS.json + 75_gap_pca.png")
