"""P gate — DATASET-LEVEL leakage probe (no trained model in the loop, per project decision:
the CDI is a property of the dataset alone; the fixed 132-feature bank is deterministic
pipeline, not a learned model).

The check that matters: can a probe predict the scene CLASS from windows that contain NO
class signal (nothing windows + present-but-sub-floor windows)? If yes, something OTHER than
the class signal (weather, terrain, rig draw, rendering artifact) is leaking the label — a
shortcut the model would learn. Target: max one-vs-rest AUC <= 0.55.

(Complement note: predicting terrain/condition FROM features is expected — weather audibly
changes the signal; that is physics, not leakage. Leakage is only when nuisances predict the
LABEL absent the signal.)

Usage: python nuisance_probe.py <features_dir> <tag>
Writes NUISANCE_PROBE_<tag>.json next to this script.
"""
import os, sys, glob, json, sqlite3
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FEATS_DIR = sys.argv[1] if len(sys.argv) > 1 else r"G:/geophone_synth/features_v4_3s"
TAG = sys.argv[2] if len(sys.argv) > 2 else "v4"
CLASSES = ("human", "vehicle", "animal")
con = sqlite3.connect(os.path.join(ROOT, "feature_analysis_v2.sqlite"))
FEATURES = [r[0] for r in con.execute("SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]
con.close()

sh = sorted(glob.glob(os.path.join(FEATS_DIR, "features_shard_*.parquet")))
cols = FEATURES + ["coarse", "subkind"] + [f"{c}_occ" for c in CLASSES]
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sh], ignore_index=True)
# CONFUSER nothing-subkinds (machinery/overflight/traffic) carry deliberate signal content —
# distinguishing them is the intended task, not a shortcut. The P gate compares AMBIENT-ONLY:
# calm/wind/rain nothing vs subject-scene zero-occupancy windows.
is_confuser_nothing = (df["coarse"].astype(str) == "nothing") & \
    (~df["subkind"].astype(str).isin(["calm", "wind", "rain", "ambient"]))   # v4.2: ambient = pure bg
df = df[~is_confuser_nothing]

# TRUE ZERO-SIGNAL selection via OCCUPANCY: windows where NO class has any emission activity
# (occ == 0 for all classes). For subject/mixed scenes these are gap/pre-onset moments — a
# physically pure noise bed. Any class-predictability here is PURE shortcut (weather/terrain/
# rig draw leaking the label), never residual faint signal. (Sub-floor-but-occupied windows
# are deliberately EXCLUDED: below tau_lo there is still nonzero signal, so probes partially
# detecting it would inflate a false "leak".)
occ0 = np.ones(len(df), bool)
for c in CLASSES:
    occ0 &= (df[f"{c}_occ"].to_numpy() <= 0.0)
d = df[occ0]
rng = np.random.default_rng(0)
if len(d) > 120_000:
    d = d.iloc[rng.choice(len(d), 120_000, replace=False)]
X = np.nan_to_num(d[FEATURES].to_numpy(np.float32))
y = d["coarse"].astype(str).to_numpy()
print(f"zero-signal windows: {len(d):,} | class mix: {pd.Series(y).value_counts().to_dict()}")

clf = HistGradientBoostingClassifier(max_iter=150, random_state=0)
proba = cross_val_predict(clf, X, y, cv=3, method="predict_proba", n_jobs=3)
classes = sorted(set(y))
aucs = {}
for i, cl in enumerate(classes):
    yy = (y == cl).astype(int)
    if 0 < yy.sum() < len(yy):
        aucs[cl] = round(float(roc_auc_score(yy, proba[:, i])), 4)
mx = max(aucs.values())
rep = {"features_dir": FEATS_DIR, "n_zero_signal": int(len(d)),
       "class_auc_zero_signal": aucs, "max_class_auc_zero_signal": mx,
       "target": "<= 0.55 (chance = 0.5): above => label leaks through non-signal pathways"}
print("one-vs-rest AUC on ZERO-SIGNAL windows (target <= 0.55):")
for k, v in aucs.items():
    print(f"  {k:10s} {v:.4f}" + ("  <-- LEAK" if v > 0.55 else ""))
json.dump(rep, open(os.path.join(HERE, f"NUISANCE_PROBE_{TAG}.json"), "w"), indent=1)
print(f"wrote NUISANCE_PROBE_{TAG}.json")
