"""Reproduce the original notebook's within-dataset result, then re-split it honestly.

The notebook (Goephone-Project, commit 26f1c2e, Training.ipynb) did this:
  band-pass 1-200 Hz, clip at the 1st/99th percentile, RMS-normalize each file;
  1,000-sample windows at 80% overlap (step 200); 11 hand features per window;
  a standard scaler + RBF support-vector machine, grid-searched with 3-fold CV;
  one stratified random 80/20 split, random_state 42.
Recorded: SVM 0.9751 (person vs its quiet file), 1.0000 (car vs its quiet file), 0.9912
(four-way: person, person-quiet, car, car-quiet); one-neighbour KNN 1.0000 on all three.

This script re-runs exactly that, then the same models under two other splits:
  random 5-fold      -- what "5-fold" usually means; neighbouring windows share 80% of their
                        samples, so every test window has near-copies in training
  contiguous 5-fold  -- each file cut into five consecutive blocks; fold k tests block k of
                        every file; training windows within 5 windows of a test block are
                        dropped so no sample is shared across the split
Every row is written to the `within_dataset` table of the campaign database.

    python -m campaign.svm_repro
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from training import db

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))
DATA = os.path.join(ROOT, "Goephone-Project", "geophone_data")
CAR_FIX = os.environ.get("GEO_CAR_FIX", "")          # the notebook's car file, if recovered

SCHEMA = """
CREATE TABLE IF NOT EXISTS within_dataset (
    model TEXT NOT NULL, features TEXT NOT NULL, task TEXT NOT NULL, split TEXT NOT NULL,
    fold INTEGER NOT NULL, n_train INTEGER, n_test INTEGER, accuracy REAL, per_class_json TEXT,
    note TEXT, PRIMARY KEY (model, features, task, split, fold));
"""


# ---------------------------------------------------------------- the notebook, verbatim in effect

def bandpass(x, lo=1, hi=200, fs=1000, order=4):
    b, a = butter(order, [lo / (0.5 * fs), hi / (0.5 * fs)], btype="band")
    return filtfilt(b, a, x)


def preprocess(x):
    x = bandpass(x)
    x = np.clip(x, np.quantile(x, 0.01), np.quantile(x, 0.99))
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-8)


def segment(x, window=1000, overlap=0.8):
    step = int(window * (1 - overlap))
    return np.array([x[s:s + window] for s in range(0, len(x) - window, step)])


def features(w):
    f = np.abs(np.fft.rfft(w))
    fr = np.fft.rfftfreq(len(w), d=1 / 1000)
    p = f / (f.sum() + 1e-12)
    be = lambda lo, hi: float((f[(fr >= lo) & (fr <= hi)] ** 2).sum())
    return [w.mean(), w.std(), np.sqrt(np.mean(w ** 2)), w.max() - w.min(),
            float(((w[:-1] * w[1:]) < 0).sum()), float((fr * f).sum() / (f.sum() + 1e-12)),
            float(-(p * np.log(p + 1e-12)).sum()), be(0, 10), be(10, 50), be(50, 100), be(100, 200)]


def load():
    files = {
        "car": CAR_FIX if CAR_FIX and os.path.exists(CAR_FIX) else os.path.join(DATA, "car.csv"),
        "car_nothing": os.path.join(DATA, "car_nothing.csv"),
        "human": os.path.join(DATA, "human.csv"),
        "human_nothing": os.path.join(DATA, "human_nothing.csv"),
    }
    X, y, pos, src = [], [], [], []
    for name, path in files.items():
        a = pd.read_csv(path).iloc[:, -1].to_numpy(np.float64)
        segs = segment(preprocess(a))
        for i, w in enumerate(segs):
            X.append(features(w)); y.append(name); pos.append(i); src.append(name)
        print(f"  {name}: {len(segs)} windows from {os.path.basename(path)}", flush=True)
    return np.array(X), np.array(y), np.array(pos), np.array(src)


# ---------------------------------------------------------------- splits

def contiguous_folds(src, pos, k=5, purge=5):
    """Fold k = the k-th consecutive fifth of every file. Training rows within `purge` windows
    of a test block are dropped: at 80% overlap, windows 1-4 apart share samples."""
    folds = []
    for f in range(k):
        test = np.zeros(len(src), bool); drop = np.zeros(len(src), bool)
        for name in np.unique(src):
            m = np.where(src == name)[0]
            n = len(m); a, b = f * n // k, (f + 1) * n // k
            test[m[a:b]] = True
            drop[m[max(0, a - purge):a]] = True
            drop[m[b:min(n, b + purge)]] = True
        folds.append((np.where(~test & ~drop)[0], np.where(test)[0]))
    return folds


def run_task(con, task, X, y, src, pos, models, verbose=True):
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    classes = np.unique(y)
    splits = {}
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    splits["notebook 80/20, random_state 42"] = [(tr, te)]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    splits["random 5-fold"] = list(skf.split(X, y))
    splits["contiguous 5-fold, purged"] = contiguous_folds(src, pos)
    for mname, factory in models.items():
        for sname, folds in splits.items():
            accs = []
            for fi, (a, b) in enumerate(folds):
                pipe = Pipeline([("scaler", StandardScaler()), ("model", factory())])
                pipe.fit(X[a], y[a])
                pred = pipe.predict(X[b])
                acc = accuracy_score(y[b], pred)
                per = {str(c): float(((pred == c) & (y[b] == c)).sum() / max(1, (y[b] == c).sum())) for c in classes}
                con.execute("INSERT OR REPLACE INTO within_dataset VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (mname, "notebook 11 features, 1 s windows, 80% overlap", task, sname, fi,
                             int(len(a)), int(len(b)), float(acc), json.dumps(per), None))
                accs.append(acc)
            con.commit()
            if verbose:
                print(f"    {task:<14} {mname:<12} {sname:<32} accuracy {np.mean(accs):.4f}"
                      + (f" ± {np.std(accs, ddof=1):.4f} over {len(accs)} folds" if len(accs) > 1 else ""), flush=True)


def main():
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    con = db.connect(os.path.join(OUT, "campaign.sqlite"))
    con.executescript(SCHEMA); con.commit()
    t0 = time.time()
    print("[svm] building the notebook's feature table", flush=True)
    X, y, pos, src = load()
    print(f"      {X.shape[0]:,} windows x {X.shape[1]} features in {time.time() - t0:.0f}s", flush=True)
    tasks = {
        "person vs quiet": (np.isin(y, ["human", "human_nothing"]), lambda yy: (yy == "human").astype(int),
                            dict(C=1, gamma=0.1)),
        "car vs quiet": (np.isin(y, ["car", "car_nothing"]), lambda yy: (yy == "car").astype(int),
                         dict(C=100, gamma=0.001)),
        "four-way": (np.ones(len(y), bool), lambda yy: yy, dict(C=100, gamma="scale")),
        "three-way": (np.ones(len(y), bool), lambda yy: np.where(yy == "human", "person",
                                                                    np.where(yy == "car", "vehicle", "nothing")),
                      dict(C=100, gamma="scale")),
    }
    for task, (mask, relabel, svm_params) in tasks.items():
        models = {
            "SVM rbf, notebook params": (lambda p=svm_params: SVC(kernel="rbf", **p)),
            "1-NN": (lambda: KNeighborsClassifier(n_neighbors=1)),
        }
        run_task(con, task, X[mask], relabel(y[mask]), src[mask], pos[mask], models)
    print(f"[svm] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
