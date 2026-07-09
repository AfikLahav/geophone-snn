"""Same feature analysis as screen_features.py, run on the REAL geophone validation
CSVs (Goephone-Project/geophone_data). Thin CSV adapter only — the 132 features come
from features.window_features and the rank-AUC from screen_features.rank_auc (both reused
verbatim, nothing reimplemented).

Real data = actual geophone OUTPUT @ 1000 Hz (cols: time_s,amplitude), so it is directly
comparable to the synthetic out_mv domain (no geophone render needed). Windowed NW=3000,
hop=1500 EXACTLY like the synthetic corpus.

Labels present in the real set: human / vehicle / nothing only. There is NO animal, NO
single-vs-multiple, and NO per-class SNR -> those synthetic boundaries are N/A here.

CAVEAT (reported in the MD): one continuous recording per class, so GBT holdout windows
share their source session with train -> GBT numbers are optimistic/indicative. The
per-feature rank-AUC (pure distributional separation, no train/test) is the trustworthy
real metric and the right thing to compare against the synthetic AUC table.

Outputs: REAL_FEATURE_ANALYSIS.md  +  real_feature_analysis.sqlite (real_auc/real_gbt/meta)
Usage: python analyze_real.py
"""
import os, sys, json, sqlite3, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from screen_features import rank_auc

DATA = os.path.join(HERE, "..", "Goephone-Project", "geophone_data")
FILES = {"car.csv": "vehicle", "human.csv": "human",
         "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
HOP = 1500                       # 1.5 s hop, NW=3.0 s -> 50% overlap (matches synthetic)
PROJ = os.path.join(HERE, "..")
MD_OUT = os.path.join(PROJ, "REAL_FEATURE_ANALYSIS.md")
DB_OUT = os.path.join(PROJ, "real_feature_analysis.sqlite")


def featurize(path):
    """Slide NW/HOP windows over one recording, return (X[n,132], n_samples)."""
    a = pd.read_csv(path)["amplitude"].to_numpy(np.float32)
    pre = F.scene_precompute(a)
    feats = [F.window_features(pre, i0).astype(np.float32)
             for i0 in range(0, len(a) - F.NW + 1, HOP)]
    X = np.stack(feats) if feats else np.empty((0, F.NFEAT), np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), len(a)


def main():
    t00 = time.time()
    allX, ally, allsrc = [], [], []
    n_samp = {}
    for fn, cls in FILES.items():
        X, ns = featurize(os.path.join(DATA, fn))
        n_samp[fn] = ns
        allX.append(X); ally += [cls] * len(X); allsrc += [fn] * len(X)
        print(f"  {fn:20s} -> {len(X):5d} windows ({ns/F.FS:.0f}s, class={cls})", flush=True)
    X = np.vstack(allX); y = np.array(ally); src = np.array(allsrc)
    print(f"total {len(X)} windows, {(time.time()-t00):.1f}s", flush=True)

    m = {c: y == c for c in ("human", "vehicle", "nothing")}
    bounds = {                                            # only boundaries the real labels support
        "human_vs_vehicle":   (m["human"], m["vehicle"]),
        "human_vs_nothing":   (m["human"], m["nothing"]),
        "vehicle_vs_nothing": (m["vehicle"], m["nothing"]),
        # sanity control: two independent nothing recordings should give ~0.5 (no class signal)
        "nothing_ctrl_car_vs_human": (src == "car_nothing.csv", src == "human_nothing.csv"),
    }

    # ---------------- per-feature rank AUC (the trustworthy real metric) ----------------
    auc_rows = []
    for bname, (mp, mn) in bounds.items():
        ip = np.where(mp)[0]; iq = np.where(mn)[0]
        if len(ip) < 20 or len(iq) < 20:
            print(f"  SKIP {bname}: n={len(ip)}/{len(iq)}", flush=True)
            continue
        idx = np.concatenate([ip, iq])
        yy = np.concatenate([np.ones(len(ip)), np.zeros(len(iq))])
        sub = X[idx]
        for fi, nm in enumerate(F.FEATURE_NAMES):
            a = rank_auc(sub[:, fi], yy)
            auc_rows.append((bname, nm, float(a), float(max(a, 1 - a)),
                             int(len(ip)), int(len(iq))))
        print(f"  AUC {bname}: n={len(ip)}/{len(iq)}", flush=True)

    # ---------------- GBT (indicative only; one session per class) ----------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    rng = np.random.default_rng(0)
    # contiguous per-source 70/30 split (windows are appended in i0 order) to limit
    # adjacent-window leakage across the split boundary
    test = np.zeros(len(X), bool)
    for fn in FILES:
        s = np.where(src == fn)[0]
        test[s[int(len(s) * 0.7):]] = True

    gbt_rows = []

    def fit(mask, yy, name, binary):
        tr = mask & ~test; te = mask & test
        if tr.sum() < 30 or te.sum() < 10 or len(np.unique(yy[tr])) < 2:
            print(f"  SKIP GBT {name}", flush=True); return
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
        clf.fit(X[tr], yy[tr]); pred = clf.predict(X[te])
        ba = balanced_accuracy_score(yy[te], pred)
        labels = sorted(np.unique(yy[mask]).tolist())
        cm = confusion_matrix(yy[te], pred, labels=labels)
        auc = None
        if binary:
            auc = float(roc_auc_score(yy[te], clf.predict_proba(X[te])[:, 1]))
        # permutation importance on the test fold
        base = balanced_accuracy_score(yy[te], clf.predict(X[te]))
        Xte = X[te].copy(); imp = []
        idx_te = np.where(te)[0]
        for fi, nm in enumerate(F.FEATURE_NAMES):
            sv = Xte[:, fi].copy(); Xte[:, fi] = rng.permutation(sv)
            imp.append((nm, float(base - balanced_accuracy_score(yy[te], clf.predict(Xte)))))
            Xte[:, fi] = sv
        imp.sort(key=lambda t: -t[1])
        gbt_rows.append((name, float(ba), json.dumps({"labels": labels, "matrix": cm.tolist()}),
                         auc, int(tr.sum()), int(te.sum()), json.dumps(imp[:15])))
        print(f"  GBT {name}: bal-acc {ba:.3f}" + (f" auc {auc:.3f}" if auc else ""), flush=True)

    fit(np.ones(len(X), bool), y, "hva_3class", binary=False)
    for c in ("human", "vehicle"):
        fit(np.ones(len(X), bool), (y == c).astype(int), f"{c}_present", binary=True)

    # ---------------- persist: sqlite ----------------
    db = sqlite3.connect(DB_OUT)
    db.executescript("""
        DROP TABLE IF EXISTS real_auc; DROP TABLE IF EXISTS real_gbt; DROP TABLE IF EXISTS meta;
        CREATE TABLE real_auc(boundary TEXT, feature TEXT, auc REAL, auc_abs REAL,
                              n_pos INT, n_neg INT);
        CREATE TABLE real_gbt(name TEXT, balanced_acc REAL, confusion TEXT, auc REAL,
                              n_train INT, n_test INT, top_importance TEXT);
        CREATE TABLE meta(key TEXT, value TEXT);
    """)
    db.executemany("INSERT INTO real_auc VALUES (?,?,?,?,?,?)", auc_rows)
    db.executemany("INSERT INTO real_gbt VALUES (?,?,?,?,?,?,?)", gbt_rows)
    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("n_windows_total", str(len(X))), ("hop", str(HOP)), ("nw", str(F.NW)),
        ("nfeat", str(F.NFEAT)), ("files", json.dumps(n_samp)),
        ("class_window_counts", json.dumps({c: int(m[c].sum()) for c in m})),
        ("note", "one continuous recording per class; GBT indicative only, AUC trustworthy"),
    ])
    db.commit(); db.close()

    # ---------------- persist: markdown ----------------
    auc_df = pd.DataFrame(auc_rows, columns=["boundary", "feature", "auc", "auc_abs",
                                             "n_pos", "n_neg"])
    L = []
    L.append("# Real-data feature analysis (geophone validation CSVs)\n")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · same 132-feature set and rank-AUC "
             "methodology as the synthetic v2 screening (features.py + screen_features.rank_auc, "
             "reused verbatim)._\n")
    L.append("## Dataset\n")
    L.append("Source: `Goephone-Project/geophone_data/*.csv` — real geophone OUTPUT @ 1000 Hz "
             "(`time_s,amplitude`). Windowed NW=3000 (3 s), hop=1500 (1.5 s, 50% overlap), "
             "identical to the synthetic corpus.\n")
    L.append("| file | class | seconds | windows |")
    L.append("|---|---|---:|---:|")
    for fn, cls in FILES.items():
        L.append(f"| `{fn}` | {cls} | {n_samp[fn]/F.FS:.0f} | {int((src==fn).sum())} |")
    L.append("")
    L.append("**Labels available:** human / vehicle / nothing. **Absent in real:** animal, "
             "single-vs-multiple, per-class SNR — those synthetic boundaries cannot be evaluated here.\n")
    L.append("> **Caveat.** One continuous recording per class, so GBT holdout windows share "
             "their source session with training windows — the GBT balanced-accuracy below is "
             "**optimistic / indicative only**. The per-feature rank-AUC (pure distributional "
             "separation, no train/test) is the trustworthy metric and the one to compare against "
             "the synthetic AUC table. Absolute-level features may differ in scale between real "
             "and synthetic; rank-AUC separation is scale-robust per domain.\n")

    L.append("## Per-feature rank-AUC — top 20 separators per boundary\n")
    for b in bounds:
        d = auc_df[auc_df.boundary == b].sort_values("auc_abs", ascending=False)
        if d.empty:
            continue
        np_, nn_ = int(d.n_pos.iloc[0]), int(d.n_neg.iloc[0])
        L.append(f"### {b}  (n={np_} vs {nn_})\n")
        if b.startswith("nothing_ctrl"):
            L.append("_Control: two independent nothing recordings. Features near 0.5 = no "
                     "spurious source separation; high values = recording/site artifact to be "
                     "wary of._\n")
        L.append("| feature | AUC | |AUC-0.5|·2 |")
        L.append("|---|---:|---:|")
        for _, r in d.head(20).iterrows():
            L.append(f"| {r.feature} | {r.auc:.3f} | {abs(r.auc_abs-0.5)*2:.3f} |")
        L.append("")

    L.append("## Multivariate GBT (indicative)\n")
    for name, ba, cm, auc, ntr, nte, top in gbt_rows:
        cmj = json.loads(cm)
        L.append(f"### {name}\n")
        L.append(f"- balanced accuracy: **{ba:.3f}**" + (f" · ROC-AUC {auc:.3f}" if auc else ""))
        L.append(f"- train/test windows: {ntr}/{nte}")
        L.append(f"- labels: {cmj['labels']}")
        L.append(f"- confusion (rows=true): `{cmj['matrix']}`")
        L.append("- top features (permutation importance): " +
                 ", ".join(f"{n} ({v:+.3f})" for n, v in json.loads(top)[:10]))
        L.append("")

    L.append("## Cross-domain comparison\n")
    L.append("The synthetic v2 screening writes the same per-feature AUC for `human_vs_vehicle`, "
             "`human_vs_nothing`, `vehicle_vs_nothing`. The synth-vs-real delta per feature "
             "(does a feature that separates classes in synthetic also separate them in real?) "
             "is added to `FEATURE_ANALYSIS_V2.md` once the synthetic screening completes.\n")
    open(MD_OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"\nWROTE {MD_OUT}\nWROTE {DB_OUT}\nDONE in {(time.time()-t00):.1f}s", flush=True)


if __name__ == "__main__":
    main()
