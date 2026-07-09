"""Characterize the MISSED real windows (true present -> predicted nothing) to decide whether
they are genuinely signal-free (label wrong) or carry signal the model misses (model blind).

For each present class:
  groups = detected (correct), missed (->nothing), and the matched BACKGROUND recording
           (human->human_nothing, vehicle->car_nothing).
  compare raw window RMS + the class-discriminative band feature; quantify the fraction of missed
  windows that are statistically indistinguishable from background (=> candidate mislabels).
Also: does the model's presence prob TRACK residual signal among missed windows (corr prob vs RMS)?
Writes MISSED_ANALYSIS.json + plots; lists flagged-likely-mislabeled window indices per recording.
"""
import os, sys, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate, layer, functional
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.path.join(HERE, "snn_v2_out"); PNG = os.path.join(OUT, "plots")
REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data"); DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, SCENE, HOP = 4, 30 * int(F.FS), 1500
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]
TH = json.load(open(os.path.join(OUT, "ERROR_ANALYSIS.json")))["thresholds"]   # per-head syn-cal thresholds


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
def probs(X):
    o = {"human": [], "vehicle": []}
    for j in range(0, len(X), 8192):
        functional.reset_net(m); m(torch.as_tensor(X[j:j + 8192]).to(DEV))
        for nm in o: o[nm].append(torch.sigmoid(getattr(m, nm)[-1].v_seq[-1][:, 0]).cpu())
    return {nm: np.array(torch.cat(o[nm])) for nm in o}

REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
def feat_real(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32); fe, rms = [], []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP):
            fe.append(F.window_features(pre, i0).astype(np.float32))
            w = seg[i0:i0 + F.NW]; rms.append(float(np.sqrt(np.mean(w ** 2))))
    X = np.stack(fe) if fe else np.empty((0, F.NFEAT), np.float32)
    return np.nan_to_num(X), np.array(rms)

rawX, RMS, ry, rsrc, ridx_local = [], [], [], [], []
for fn, cls in REALS.items():
    Xr, rms = feat_real(os.path.join(REAL_DIR, fn)); rawX.append(Xr); RMS.append(rms)
    ry += [cls] * len(Xr); rsrc += [fn] * len(Xr); ridx_local += list(range(len(Xr)))
rawX = np.vstack(rawX); RMS = np.concatenate(RMS); ry = np.array(ry); rsrc = np.array(rsrc); ridx_local = np.array(ridx_local)
Xz = np.clip((rawX[:, fidx] - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
pr = probs(Xz)
mh, mv = pr["human"] - TH["human"], pr["vehicle"] - TH["vehicle"]
pred = np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))

# raw class-band energies (NOT z-scored) for "is the class signal there" test
HB = F.FEATURE_NAMES.index("energy_human_peak"); CB = F.FEATURE_NAMES.index("energy_car_peak")
hum_e = rawX[:, HB]; car_e = rawX[:, CB]

from scipy.stats import mannwhitneyu
report = {"thresholds": TH}
flagged = {}
for cls, bgfile, band, bandname, headp in [("human", "human_nothing.csv", hum_e, "energy_human_peak", pr["human"]),
                                           ("vehicle", "car_nothing.csv", car_e, "energy_car_peak", pr["vehicle"])]:
    det = (ry == cls) & (pred == cls)
    mis = (ry == cls) & (pred == "nothing")
    bg = rsrc == bgfile
    def stat(mask, arr): return {"median": float(np.median(arr[mask])), "p90": float(np.percentile(arr[mask], 90)), "n": int(mask.sum())}
    bg90_rms = np.percentile(RMS[bg], 90); bg90_band = np.percentile(band[bg], 90)
    # fraction of missed indistinguishable from background (below bg 90th pct on BOTH rms and class band)
    below = mis & (RMS <= bg90_rms) & (band <= bg90_band)
    frac_bg = float(below[mis].sum() / max(1, mis.sum()))
    # does the model track residual signal among missed? corr(prob, rms) and corr(prob, band)
    corr_rms = float(np.corrcoef(headp[mis], RMS[mis])[0, 1]) if mis.sum() > 2 else None
    corr_band = float(np.corrcoef(headp[mis], band[mis])[0, 1]) if mis.sum() > 2 else None
    # is missed signal ABOVE background? Mann-Whitney missed-band vs bg-band
    mw = mannwhitneyu(band[mis], band[bg], alternative="greater") if mis.sum() > 5 else None
    report[cls] = {
        "rms": {"detected": stat(det, RMS), "missed": stat(mis, RMS), "background": stat(bg, RMS)},
        "class_band": {"feature": bandname, "detected": stat(det, band), "missed": stat(mis, band), "background": stat(bg, band)},
        "missed_frac_indistinguishable_from_bg": frac_bg,
        "missed_band_above_bg_pvalue": (float(mw.pvalue) if mw else None),
        "model_tracks_residual": {"corr_prob_rms": corr_rms, "corr_prob_band": corr_band,
                                  "missed_median_prob": float(np.median(headp[mis]))},
    }
    # flag candidate-mislabeled missed windows (indistinguishable from bg)
    fl = np.where(below)[0]
    flagged[cls] = [{"recording": rsrc[i], "window_idx": int(ridx_local[i]),
                     "rms": float(RMS[i]), bandname: float(band[i]), "prob": float(headp[i])} for i in fl]

report["flagged_likely_mislabeled_counts"] = {k: len(v) for k, v in flagged.items()}
json.dump(report, open(os.path.join(OUT, "MISSED_ANALYSIS.json"), "w"), indent=1)
json.dump(flagged, open(os.path.join(OUT, "flagged_mislabeled_windows.json"), "w"), indent=1)

# plots: RMS distributions per class (detected/missed/background)
for cls, bgfile in [("human", "human_nothing.csv"), ("vehicle", "car_nothing.csv")]:
    det = (ry == cls) & (pred == cls); mis = (ry == cls) & (pred == "nothing"); bg = rsrc == bgfile
    plt.figure(figsize=(7, 4))
    for mask, lab in [(det, "detected"), (mis, "missed"), (bg, f"bg ({bgfile})")]:
        plt.hist(np.log10(RMS[mask] + 1e-9), bins=40, alpha=.5, label=lab, density=True)
    plt.xlabel("log10 window RMS"); plt.ylabel("density"); plt.legend(); plt.title(f"{cls}: RMS of detected vs missed vs background")
    plt.tight_layout(); plt.savefig(os.path.join(PNG, f"74_missed_rms_{cls}.png"), dpi=120); plt.close()

print("=== MISSED-WINDOW ANALYSIS ===")
for cls in ("human", "vehicle"):
    r = report[cls]
    print(f"\n{cls.upper()}  (missed n={r['rms']['missed']['n']})")
    print(f"  RMS  median  detected {r['rms']['detected']['median']:.4f} | missed {r['rms']['missed']['median']:.4f} | bg {r['rms']['background']['median']:.4f}")
    print(f"  {r['class_band']['feature']} median  detected {r['class_band']['detected']['median']:.4f} | missed {r['class_band']['missed']['median']:.4f} | bg {r['class_band']['background']['median']:.4f}")
    print(f"  missed indistinguishable-from-bg: {r['missed_frac_indistinguishable_from_bg']*100:.0f}%  | missed-band>bg p={r['missed_band_above_bg_pvalue']}")
    print(f"  model tracks residual: corr(prob,RMS)={r['model_tracks_residual']['corr_prob_rms']}, "
          f"corr(prob,band)={r['model_tracks_residual']['corr_prob_band']}, missed median prob {r['model_tracks_residual']['missed_median_prob']:.3f}")
print("\nflagged likely-mislabeled:", report["flagged_likely_mislabeled_counts"])
print("wrote MISSED_ANALYSIS.json + flagged_mislabeled_windows.json")
