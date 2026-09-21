"""v4 domain-gap analysis: WHERE do synthetic v4 and the real recordings differ, is the gap
task-relevant, did the D5 coupling re-anchor help, and is the human head the weak spot.

Parts:
  1. Domain-classifier AUROC (real vs synth) for v2/v3/v4 -> did v4 narrow the gap?
  2. Per-feature domain AUROC (real vs v4) -> which features separate the domains most.
  3. Task-relevance test: real human-vs-vehicle AUROC with ALL features vs after dropping the
     top-K domain separators -> is the gap on the decision axis or a nuisance axis?
  4. Human-band diagnostic: real vs v4 in-band energy fractions (20-90 Hz human band split into
     low 20-55 / high 55-90) -> did the coupling re-anchor over-attenuate the upper human band?
Writes DOMAIN_GAP_V4.json + plots.
"""
import os, sys, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4"))
import features as F

SCALE, SCENE, HOP = 25.4, 30 * int(F.FS), 1500
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
# 104 model features
import sqlite3
con = sqlite3.connect(os.path.join(ROOT, "feature_analysis_v2.sqlite"))
FEATS = [r[0] for r in con.execute("SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]; con.close()
fidx = [F.FEATURE_NAMES.index(f) for f in FEATS]


def feat_real(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE; fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.stack(fe)[:, fidx] if fe else np.empty((0, len(fidx)), np.float32)


def load_synth(feat_dir, n=6000, seed=0):
    sh = sorted(glob.glob(os.path.join(feat_dir, "features_shard_*.parquet")))[:6]
    df = pd.concat([pd.read_parquet(s, columns=FEATS + ["coarse"]) for s in sh], ignore_index=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), min(n, len(df)), replace=False)
    return df.iloc[idx][FEATS].to_numpy(np.float32), df.iloc[idx]["coarse"].to_numpy()


print("featurizing real...", flush=True)
Xr_by = {}
for fn, cls in REALS.items():
    X = feat_real(os.path.join(REAL_DIR, fn)); Xr_by.setdefault(cls, []).append(X)
Xr = {k: np.vstack(v) for k, v in Xr_by.items()}
Xreal_all = np.vstack(list(Xr.values()))
print(f"real windows: { {k: len(v) for k, v in Xr.items()} }", flush=True)

report = {"n_real": {k: int(len(v)) for k, v in Xr.items()}}

# ---- Part 1: domain-classifier AUROC for each synthetic version ----
print("\n=== Part 1: domain separability (real vs synth) per version ===", flush=True)
report["domain_auroc"] = {}
synth_dirs = {"v2": os.path.join(_GEO_ROOT, "features_v2"), "v3": os.path.join(_GEO_ROOT, "features_v3"),
              "v4": os.path.join(_GEO_ROOT, "features_v4_3s")}
Xsynth = {}
for ver, d in synth_dirs.items():
    if not glob.glob(os.path.join(d, "features_shard_*.parquet")):
        print(f"  {ver}: no shards, skip"); continue
    Xs, cs = load_synth(d); Xsynth[ver] = (Xs, cs)
    X = np.nan_to_num(np.vstack([Xreal_all, Xs])); y = np.r_[np.ones(len(Xreal_all)), np.zeros(len(Xs))]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    auc = float(np.mean(cross_val_score(clf, X, y, cv=4, scoring="roc_auc")))
    report["domain_auroc"][ver] = round(auc, 4)
    print(f"  {ver}: domain-classifier AUROC = {auc:.4f}  (1.0 = perfectly separable)", flush=True)

# ---- Part 2: per-feature domain AUROC (real vs v4) ----
print("\n=== Part 2: top per-feature domain separators (real vs v4) ===", flush=True)
Xv4, cv4 = Xsynth["v4"]
per_feat = []
for j, name in enumerate(FEATS):
    v = np.r_[Xreal_all[:, j], Xv4[:, j]]; y = np.r_[np.ones(len(Xreal_all)), np.zeros(len(Xv4))]
    v = np.nan_to_num(v)
    try:
        a = roc_auc_score(y, v); per_feat.append((name, float(max(a, 1 - a))))
    except Exception:
        pass
per_feat.sort(key=lambda t: -t[1])
report["top_domain_features"] = [{"feature": n, "domain_auroc": round(a, 3)} for n, a in per_feat[:20]]
for n, a in per_feat[:15]:
    print(f"  {n:28s} {a:.3f}", flush=True)

# ---- Part 3: task-relevance — real human-vs-vehicle, all feats vs drop top-K domain feats ----
print("\n=== Part 3: is the gap task-relevant? (real human vs vehicle) ===", flush=True)
Xh, Xv = Xr["human"], Xr["vehicle"]
Xtask = np.nan_to_num(np.vstack([Xh, Xv])); ytask = np.r_[np.ones(len(Xh)), np.zeros(len(Xv))]
def hv_auroc(cols):
    Xc = Xtask[:, cols]
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=2000), Xc, ytask, cv=5, scoring="roc_auc")))
all_cols = list(range(len(FEATS)))
topK = set(FEATS.index(d["feature"]) for d in report["top_domain_features"])
keep_cols = [c for c in all_cols if c not in topK]
auc_all = hv_auroc(all_cols); auc_drop = hv_auroc(keep_cols)
report["task_relevance"] = {"hv_auroc_all_feats": round(auc_all, 4),
                            "hv_auroc_drop_top20_domain": round(auc_drop, 4),
                            "n_human": int(len(Xh)), "n_vehicle": int(len(Xv))}
print(f"  real human-vs-vehicle AUROC: all feats {auc_all:.4f} | drop top-20 domain feats {auc_drop:.4f}")
print(f"  -> gap is {'TASK-IRRELEVANT (discrimination survives)' if auc_drop > auc_all - 0.03 else 'ON THE DECISION AXIS (discrimination drops)'}")

# ---- Part 4: human-band energy split (coupling-attenuation hypothesis) ----
print("\n=== Part 4: human-band energy (upper-band attenuation from D5 coupling re-anchor?) ===", flush=True)
def band_frac(x, lo, hi, tot=(20, 90)):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / F.FS)
    p = np.abs(X) ** 2
    return p[(f >= lo) & (f < hi)].sum() / (p[(f >= tot[0]) & (f < tot[1])].sum() + 1e-30)
# compare real human windows vs v4 synthetic human clean windows: upper/lower human-band ratio
rh = pd.read_csv(os.path.join(REAL_DIR, "human.csv"))["amplitude"].to_numpy(np.float32) * SCALE
real_hi = np.median([band_frac(rh[i:i+F.NW], 55, 90) for i in range(0, len(rh)-F.NW, HOP)])
# v4 synthetic human scene waveforms (reconstruct a few from the dataset)
import sqlite3 as sq

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
db = sq.connect(glob.glob(os.path.join(_GEO_ROOT, "dataset_v431/shard_0.sqlite"))[0])
rows = db.execute("SELECT w.clean_mv FROM scenes s JOIN waveforms w ON s.scene_id=w.scene_id "
                  "WHERE s.coarse='human' AND w.clean_mv IS NOT NULL LIMIT 60").fetchall()
db.close()
v4_his = []
for (cm,) in rows:
    x = np.frombuffer(cm, np.float32)
    for i in range(0, max(1, len(x)-F.NW), HOP*3):
        seg = x[i:i+F.NW]
        if len(seg) == F.NW and seg.std() > 1e-3: v4_his.append(band_frac(seg, 55, 90))
v4_hi = float(np.median(v4_his)) if v4_his else float("nan")
report["human_upper_band_frac"] = {"real": round(float(real_hi), 3), "v4_synth": round(v4_hi, 3),
                                   "note": "fraction of 20-90 Hz human-band energy above 55 Hz"}
print(f"  upper-human-band (55-90Hz) fraction of human band:  real {real_hi:.3f} | v4-synth {v4_hi:.3f}")
print(f"  -> {'v4 UNDER-represents the upper human band (coupling over-attenuates)' if v4_hi < real_hi - 0.05 else 'upper-band comparable'}")

json.dump(report, open(os.path.join(HERE, "DOMAIN_GAP_V4.json"), "w"), indent=1)
print("\nwrote DOMAIN_GAP_V4.json")
