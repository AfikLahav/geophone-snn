"""Classical models trained and tested WITHIN the field recordings, on all windows and on the
cleaned subset, under three splits. Same 3-second windows and cached 132 features the campaign
uses; feature sets 77 (cepstral cut) and 132 (all).

Splits: one random 80/20 split (random_state 42), random 5-fold, and contiguous 5-fold with a
purge gap (each test fold is one consecutive fifth of every recording; training windows within
two windows of a test block are dropped, since windows overlap by half).

Rows go to `within_dataset` with the feature label suffixed ", cleaned" for the cleaned runs.

    python campaign/classical_field.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simgeo", "simgeo_v42"))
import features as F                                  # noqa: E402
from training import db
from evaluation import review
from training.data import select_features

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))


def contiguous_folds(src, pos, k=5, purge=2):
    folds = []
    for f in range(k):
        test = np.zeros(len(src), bool); drop = np.zeros(len(src), bool)
        for name in np.unique(src):
            m = np.where(src == name)[0]
            m = m[np.argsort(pos[m])]
            n = len(m); a, b = f * n // k, (f + 1) * n // k
            test[m[a:b]] = True
            drop[m[max(0, a - purge):a]] = True
            drop[m[b:min(n, b + purge)]] = True
        folds.append((np.where(~test & ~drop)[0], np.where(test)[0]))
    return folds


def main():
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    con = db.connect(os.path.join(OUT, "campaign.sqlite"))
    X_all = np.load(os.path.join(ROOT, "campaign", "real_cache", "elbit_X.npy"))
    rows = json.load(open(os.path.join(ROOT, "campaign", "real_cache", "elbit_rows.json")))
    y_all = np.array([{"human": "person", "vehicle": "vehicle", "nothing": "nothing"}[r["label"]] for r in rows])
    src = np.array([r["recording"] for r in rows]); pos = np.array([r["pos"] for r in rows])
    W = {"recording": src, "t0_s": np.array([r["t0_s"] for r in rows]), "label": y_all}
    keep = review.cleaning_mask(W)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e9, neginf=-1e9)
    sets = {"77 cepstral cut": select_features("cumulative:cep16"), "132 all": list(F.FEATURE_NAMES)}
    models = {
        "SVM rbf C=100": lambda: SVC(kernel="rbf", C=100, gamma="scale"),
        "logistic regression": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "random forest": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=8),
        "boosted trees": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    t0 = time.time()
    for subset, mask in (("all", np.ones(len(y_all), bool)), ("cleaned", keep)):
        idx = np.where(mask)[0]
        y, s, p = y_all[idx], src[idx], pos[idx]
        splits = {}
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
        splits["notebook 80/20, random_state 42"] = [(tr, te)]
        splits["random 5-fold"] = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
        splits["contiguous 5-fold, purged"] = contiguous_folds(s, p)
        for fname, names in sets.items():
            cols = [F.FEATURE_NAMES.index(n) for n in names]
            X = X_all[idx][:, cols]
            label = f"{fname}, 3 s windows, 50% overlap" + (", cleaned" if subset == "cleaned" else "")
            for mname, factory in models.items():
                for sname, folds in splits.items():
                    accs = []
                    for fi, (a, b) in enumerate(folds):
                        pipe = Pipeline([("scaler", StandardScaler()), ("model", factory())])
                        pipe.fit(X[a], y[a]); pred = pipe.predict(X[b])
                        acc = accuracy_score(y[b], pred)
                        per = {c: float(((pred == c) & (y[b] == c)).sum() / max(1, (y[b] == c).sum())) for c in np.unique(y)}
                        db._write(con, "INSERT OR REPLACE INTO within_dataset VALUES (?,?,?,?,?,?,?,?,?,?)",
                                  (mname, label, "three-way", sname, fi, int(len(a)), int(len(b)), float(acc), json.dumps(per), None))
                        accs.append(acc)
                    db._write(con, "INSERT OR REPLACE INTO within_dataset VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (mname, label, "three-way", sname, -1, None, None, float(np.mean(accs)),
                               json.dumps({"std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0}), "pooled mean over folds"))
                    db._commit(con)
                    print(f"  {subset:8s} {fname:16s} {mname:20s} {sname:32s} {np.mean(accs):.4f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
