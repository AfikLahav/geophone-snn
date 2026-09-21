"""Per-feature domain-AUC: which features are sim<->real DOMAIN TAGS (separate synthetic from real
regardless of class)? Cross-reference with task value (gbt_imp) and transfer_delta to propose a
drop-set = high domain-AUC AND low task value. Measured at x25 scale-alignment (so we find residual
band/shape tags, not just the level offset). Writes DOMAIN_FEATURE_AUDIT.json.
"""
import os, sys, json, glob, sqlite3
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
OUT = os.path.join(HERE, "snn_v2_out"); FEAT_DIR = os.path.join(_GEO_ROOT, "features_v2"); REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
SCENE, HOP, SCALE = 30 * int(F.FS), 1500, 25.4
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]
con = sqlite3.connect(os.path.join(HERE, "feature_analysis_v2.sqlite"))
sc = {r[0]: {"gbt": r[1] or 0.0, "tdelta": r[2] or 0.0, "class_sep": r[3] or 0.0}
      for r in con.execute("SELECT feature,gbt_imp,transfer_delta,class_sep FROM feature_scorecard")}; con.close()

# synthetic val sample
dv = pd.concat([pd.read_parquet(s, columns=FEATURES + ["split"]) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))[:2]], ignore_index=True)
dv = dv[dv.split == "val"]; Xs = np.clip((dv[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ)
Xs = Xs[np.random.default_rng(0).choice(len(Xs), min(40000, len(Xs)), replace=False)]
# real (x25)
REALS = ["car.csv", "human.csv", "car_nothing.csv", "human_nothing.csv"]
def feat(p):
    a = pd.read_csv(os.path.join(REAL_DIR, p))["amplitude"].to_numpy(np.float32) * SCALE; fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)
Xr = np.vstack([feat(p) for p in REALS]); Xr = np.clip((Xr[:, fidx] - mu) / sd, -CLIPZ, CLIPZ)
print(f"synth {len(Xs)} vs real {len(Xr)} windows; {len(FEATURES)} features")

def auc(x_s, x_r):
    z = np.r_[x_s, x_r]; y = np.r_[np.zeros(len(x_s)), np.ones(len(x_r))]
    r = rankdata(z); n1 = int(y.sum()); n0 = len(y) - n1
    a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0); return float(a)

rows = []
for i, f in enumerate(FEATURES):
    a = auc(Xs[:, i], Xr[:, i]); strg = abs(a - 0.5) * 2
    rows.append({"feature": f, "domain_auc": round(a, 3), "domain_strength": round(strg, 3),
                 "gbt_imp": round(sc[f]["gbt"], 4), "transfer_delta": round(sc[f]["tdelta"], 3),
                 "class_sep": round(sc[f]["class_sep"], 3)})
df = pd.DataFrame(rows).sort_values("domain_strength", ascending=False)
# drop-set: strong domain tag AND low task value
df["DROP"] = (df.domain_strength >= 0.7) & (df.gbt_imp <= 0.005)
keep = df[~df.DROP].feature.tolist(); drop = df[df.DROP].feature.tolist()
df.to_csv(os.path.join(OUT, "domain_feature_audit.csv"), index=False)
json.dump({"n_features": len(FEATURES), "n_keep": len(keep), "n_drop": len(drop),
           "drop_set": drop, "keep_set": keep,
           "frac_features_domain_strength_gt0.9": float((df.domain_strength > 0.9).mean()),
           "top15_domain_tags": df.head(15)[["feature", "domain_auc", "domain_strength", "gbt_imp", "class_sep"]].to_dict("records")},
          open(os.path.join(OUT, "DOMAIN_FEATURE_AUDIT.json"), "w"), indent=1)
print(f"\nfeatures with domain_strength>0.9 (near-perfect domain tags): {int((df.domain_strength>0.9).sum())}/{len(FEATURES)}")
print(f"proposed DROP (domain_strength>=0.7 AND gbt_imp<=0.005): {len(drop)}  ->  KEEP {len(keep)}")
print("\ntop domain tags (feature | domain_auc | strength | gbt_imp | class_sep):")
for r in df.head(18).itertuples():
    print(f"  {r.feature:22s} {r.domain_auc:5.2f}  {r.domain_strength:.2f}  {r.gbt_imp:.4f}  {r.class_sep:.2f}  {'DROP' if r.DROP else ''}")
print("\nwrote DOMAIN_FEATURE_AUDIT.json + domain_feature_audit.csv")
