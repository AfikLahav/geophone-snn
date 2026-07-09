"""Real-data-only baseline with 5-fold CV — the "train directly on real" comparison point for
the synthetic-trained SNN, and the number to beat (cf. prior team's ~0.91).

HONESTY: one continuous recording per class → folds share their session with train, so accuracy
is session-confounded / optimistic (same regime as the prior work). Mitigations:
  - contiguous time-block folds (not random windows) so train/test aren't adjacent 50%-overlap views
  - report confound-robust views alongside: human-vs-vehicle (present-vs-present), and a
    robust-feature-only variant (the confound-clean features from REAL_FEATURE_ANALYSIS).

Models: LogisticRegression + HistGradientBoosting (strong tabular baselines) on the SAME 104
features the SNN uses. 3-class (human/vehicle/nothing). Saves real_baseline.json + a confusion PNG.
Usage: python real_baseline.py
"""
import os, sys, json, sqlite3
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F

REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
SQLITE = os.path.join(HERE, "feature_analysis_v2.sqlite")
REAL_DB = os.path.join(HERE, "real_feature_analysis.sqlite")
OUT = os.path.join(HERE, "snn_v2_out"); os.makedirs(OUT, exist_ok=True)
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
SCENE = 30 * int(F.FS); HOP = 1500; K = 5

con = sqlite3.connect(SQLITE)
FEATURES = [r[0] for r in con.execute(
    "SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]; con.close()
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]
# confound-clean robust features (from the real-data analysis)
rcon = sqlite3.connect(REAL_DB)
ROBUST = sorted({r[0] for r in rcon.execute("SELECT feature FROM real_robust")} & set(FEATURES)); rcon.close()
ridx = [FEATURES.index(f) for f in ROBUST]


def featurize(path):
    a = pd.read_csv(path)["amplitude"].to_numpy(np.float32); feats = []
    for c0 in range(0, len(a), SCENE):                       # chunk to synthetic scene length
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP):
            feats.append(F.window_features(pre, i0).astype(np.float32))
    X = np.stack(feats) if feats else np.empty((0, F.NFEAT), np.float32)
    return np.nan_to_num(X[:, fidx], nan=0.0, posinf=0.0, neginf=0.0)


# build dataset + contiguous block index per recording
X, y, block = [], [], []
for fn, cls in REALS.items():
    Xr = featurize(os.path.join(REAL_DIR, fn))
    n = len(Xr); blk = (np.arange(n) * K // max(n, 1))        # 0..K-1 contiguous blocks
    X.append(Xr); y += [cls] * n; block.append(blk)
    print(f"  {fn:18s} {n:5d} windows ({cls})", flush=True)
X = np.vstack(X); y = np.array(y); block = np.concatenate(block)
print(f"total {len(X)} windows | {len(FEATURES)} features | classes {dict(zip(*np.unique(y, return_counts=True)))}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report)


def make(kind):
    return (LogisticRegression(max_iter=500, C=1.0) if kind == "logreg"
            else HistGradientBoostingClassifier(max_iter=300, random_state=0))


def cv(Xd, yd, kind, robust=False, blk=None):
    if blk is None: blk = block
    Xd = Xd[:, ridx] if robust else Xd
    accs, baccs, f1s, cms = [], [], [], []
    labels = sorted(np.unique(yd))
    for k in range(K):
        te = blk == k; tr = ~te
        if te.sum() < 5 or len(np.unique(yd[tr])) < 2: continue
        sc = StandardScaler().fit(Xd[tr])
        clf = make(kind).fit(sc.transform(Xd[tr]), yd[tr])
        pred = clf.predict(sc.transform(Xd[te]))
        accs.append(accuracy_score(yd[te], pred)); baccs.append(balanced_accuracy_score(yd[te], pred))
        f1s.append(f1_score(yd[te], pred, average="macro", labels=labels))
        cms.append(confusion_matrix(yd[te], pred, labels=labels))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "bal_acc_mean": float(np.mean(baccs)), "macro_f1_mean": float(np.mean(f1s)),
            "per_fold_acc": [round(a, 4) for a in accs],
            "confusion_sum": np.sum(cms, 0).tolist(), "labels": labels}


report = {"n_windows": int(len(X)), "n_features": len(FEATURES), "folds": K,
          "fold_scheme": "contiguous time-blocks (session-confounded; optimistic)",
          "robust_features_used": ROBUST}
print("\n=== 3-class (human/vehicle/nothing) ===")
for kind in ("logreg", "hgb"):
    r = cv(X, y, kind); report[f"3class_{kind}"] = r
    print(f"  {kind:7s} acc {r['acc_mean']:.3f}±{r['acc_std']:.3f}  bal-acc {r['bal_acc_mean']:.3f}  macroF1 {r['macro_f1_mean']:.3f}")
    rr = cv(X, y, kind, robust=True); report[f"3class_{kind}_robustfeat"] = rr
    print(f"  {kind:7s} acc {rr['acc_mean']:.3f}  (robust-features-only, confound-clean view)")

print("\n=== human-vs-vehicle (present-vs-present, confound-robust) ===")
hv = np.isin(y, ["human", "vehicle"])
for kind in ("logreg", "hgb"):
    r = cv(X[hv], y[hv], kind, blk=block[hv]); report[f"hv_{kind}"] = r
    print(f"  {kind:7s} acc {r['acc_mean']:.3f}±{r['acc_std']:.3f}  bal-acc {r['bal_acc_mean']:.3f}")

# confusion PNG (3-class HGB)
cm = np.array(report["3class_hgb"]["confusion_sum"]); labels = report["3class_hgb"]["labels"]
plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues")
for (a, b), v in np.ndenumerate(cm): plt.text(b, a, int(v), ha="center", va="center")
plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
plt.title(f"Real 5-fold baseline (HGB) acc {report['3class_hgb']['acc_mean']:.3f}")
plt.xlabel("pred"); plt.ylabel("true"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "plots", "60_real_baseline_confusion.png"), dpi=120); plt.close()

json.dump(report, open(os.path.join(OUT, "real_baseline.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(OUT,'real_baseline.json')}")
print("NOTE: session-confounded (1 recording/class) -> optimistic; compare to the synthetic-trained "
      "model's run-once real test; read the robust-feature / human-vs-vehicle rows as the honest floor.")
