"""Where does the ~12% real-accuracy loss come from? Error analysis of the best config
(EMA + last-step readout + per-head thresholds calibrated on synthetic val) on the real 3-class task.
Breaks errors into: missed-detection / false-alarm / class-confusion; per recording; by confidence
(borderline vs confident-wrong); and which FEATURES most separate errors from correct windows
(cross-referenced with sim->real transfer_delta). Writes ERROR_ANALYSIS.json + plots.
"""
import os, sys, json, glob, sqlite3
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
OUT = os.path.join(HERE, "snn_v2_out"); PNG = os.path.join(OUT, "plots")
FEAT_DIR = os.path.join(_GEO_ROOT, "features_v2"); REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, SCENE, HOP = 4, 30 * int(F.FS), 1500
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]
# transfer_delta per feature (from the screening)
con = sqlite3.connect(os.path.join(HERE, "feature_analysis_v2.sqlite"))
TDELTA = {r[0]: (r[1] if r[1] is not None else 0.0) for r in con.execute("SELECT feature,transfer_delta FROM feature_scorecard")}; con.close()


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
def probs(X):                              # last-step readout presence probs
    o = {"human": [], "vehicle": [], "animal": []}
    for j in range(0, len(X), 8192):
        functional.reset_net(m); m(torch.as_tensor(X[j:j + 8192]).to(DEV))
        for nm in o: o[nm].append(torch.sigmoid(getattr(m, nm)[-1].v_seq[-1][:, 0]).cpu())
    return {nm: np.array(torch.cat(v)) for nm in o for v in [o[nm]]}

def best_thr(p, y):
    g = np.linspace(0.05, 0.95, 19); return float(g[np.argmax([balanced_accuracy_score(y, p > t) for t in g])])

# synthetic val -> per-head thresholds
LVL = {"none": 0, "single": 1, "multiple": 2}
dv = pd.concat([pd.read_parquet(s, columns=FEATURES + ["split", "human_level", "vehicle_level", "animal_level"])
                for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))], ignore_index=True)
dv = dv[dv.split == "val"].reset_index(drop=True)
Xv = np.clip((dv[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ)
def lv(c): return dv[c].map(LVL).fillna(dv[c]).to_numpy().astype(int)
vp = probs(Xv)
TH = {"human": best_thr(vp["human"], lv("human_level") > 0),
      "vehicle": best_thr(vp["vehicle"], lv("vehicle_level") > 0),
      "animal": best_thr(vp["animal"], lv("animal_level") > 0)}

# real
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
rX, ry, rsrc = [], [], []
for fn, cls in REALS.items():
    Xr = feat_real(os.path.join(REAL_DIR, fn)); rX.append(Xr); ry += [cls] * len(Xr); rsrc += [fn] * len(Xr)
rX = np.vstack(rX); ry = np.array(ry); rsrc = np.array(rsrc)
rp = probs(rX)
mh, mv = rp["human"] - TH["human"], rp["vehicle"] - TH["vehicle"]
pred = np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
err = pred != ry
acc = float((~err).mean())

labs = ["human", "vehicle", "nothing"]; cm = confusion_matrix(ry, pred, labels=labs)
# error taxonomy
present = ry != "nothing"; pred_present = pred != "nothing"
missed = (present & ~pred_present)                 # present -> nothing
falarm = (~present & pred_present)                 # nothing -> present
classconf = (present & pred_present & (pred != ry))  # human<->vehicle
report = {
    "config": "EMA + last-step + per-head syn-cal thresholds", "thresholds": TH,
    "real_accuracy": acc, "n": int(len(ry)),
    "confusion_rows_true": cm.tolist(), "labels": labs,
    "error_taxonomy": {"missed_detection": int(missed.sum()), "false_alarm": int(falarm.sum()),
                       "class_confusion_h_v": int(classconf.sum()), "total_errors": int(err.sum())},
    "per_class_report": classification_report(ry, pred, labels=labs, output_dict=True, zero_division=0),
    "per_recording_error_rate": {fn: float(err[rsrc == fn].mean()) for fn in REALS},
}
# confidence of errors: winning-head prob for present-predictions; for missed, the max(prob) vs threshold
win = np.maximum(rp["human"], rp["vehicle"])
report["error_confidence"] = {
    "missed_median_maxprob": float(np.median(win[missed])) if missed.any() else None,
    "missed_thr_human_vehicle": [TH["human"], TH["vehicle"]],
    "falarm_median_winprob": float(np.median(win[falarm])) if falarm.any() else None,
    "correct_median_winprob_present": float(np.median(win[(~err) & pred_present])) if ((~err) & pred_present).any() else None,
}
# which features most separate ERROR vs CORRECT real windows (standardized mean diff) + their transfer_delta
e, c = rX[err], rX[~err]
smd = np.abs(e.mean(0) - c.mean(0)) / (np.sqrt((e.var(0) + c.var(0)) / 2) + 1e-9)
ordr = np.argsort(-smd)
report["top_error_features"] = [{"feature": FEATURES[i], "std_mean_diff": float(smd[i]),
                                 "transfer_delta": float(TDELTA.get(FEATURES[i], 0.0))} for i in ordr[:15]]
report["corr_errorfeat_vs_transferdelta"] = float(np.corrcoef(
    smd, [TDELTA.get(f, 0.0) for f in FEATURES])[0, 1])
json.dump(report, open(os.path.join(OUT, "ERROR_ANALYSIS.json"), "w"), indent=1)

# plots
plt.figure(figsize=(4.6, 4)); plt.imshow(cm, cmap="Reds")
for (a, b), v in np.ndenumerate(cm): plt.text(b, a, int(v), ha="center", va="center")
plt.xticks(range(3), labs); plt.yticks(range(3), labs); plt.xlabel("pred"); plt.ylabel("true")
plt.title(f"Real errors (acc {acc:.3f})"); plt.tight_layout(); plt.savefig(os.path.join(PNG, "72_error_confusion.png"), dpi=120); plt.close()
plt.figure(figsize=(7, 4))
plt.hist(win[~err & pred_present], bins=30, alpha=.5, label="correct (present)")
plt.hist(win[missed], bins=30, alpha=.5, label="missed (true present->nothing)")
plt.hist(win[falarm], bins=30, alpha=.5, label="false alarm")
plt.axvline(min(TH["human"], TH["vehicle"]), color="k", ls="--", lw=.8, label="min threshold")
plt.legend(); plt.xlabel("winning head presence prob"); plt.title("Error confidence (real)")
plt.tight_layout(); plt.savefig(os.path.join(PNG, "73_error_confidence.png"), dpi=120); plt.close()

print(f"real acc {acc:.3f} | errors {int(err.sum())}/{len(ry)}")
print("taxonomy:", report["error_taxonomy"])
print("per-recording err rate:", {k: round(v, 3) for k, v in report["per_recording_error_rate"].items()})
print("confidence:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report["error_confidence"].items()})
print(f"corr(error-feature-separation, transfer_delta) = {report['corr_errorfeat_vs_transferdelta']:.3f}")
print("top error-associated features (smd | transfer_delta):")
for d in report["top_error_features"][:10]: print(f"   {d['feature']:22s} smd {d['std_mean_diff']:.2f}  Δtransfer {d['transfer_delta']:.2f}")
print("per-class:")
for cl in labs:
    r = report["per_class_report"][cl]; print(f"   {cl:8s} prec {r['precision']:.3f} recall {r['recall']:.3f} f1 {r['f1-score']:.3f} n {int(r['support'])}")
print("\nwrote", os.path.join(OUT, "ERROR_ANALYSIS.json"))
