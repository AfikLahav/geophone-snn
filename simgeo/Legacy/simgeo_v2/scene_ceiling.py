"""Phase A1 — scene-level ceiling for vehicle single-vs-multiple.

Per-window screening put vehicle single-vs-multi at bal-acc 0.605 (near chance). This asks:
is the signal recoverable at the SCENE level (full temporal context)?
  scene-level >> 0.605  -> signal exists but is temporal -> counting belongs in a temporal head
  scene-level  ~ 0.605  -> no recoverable signal even with full context -> a wall

Method: aggregate the existing per-window features over each VEHICLE scene
(mean/std/min/max/p90 + n_windows), label = subkind in {convoy,two_vehicle} (ground-truth
scene intent, clean by construction), train HistGBT with a PROFILE-grouped holdout.

Sharded: scenes never span shards, so each worker aggregates its own parquet independently;
we concat the small scene tables and train once. Usage: python scene_ceiling.py
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F

PARQ = sorted(glob.glob(r"G:/geophone_synth/features_v2/features_shard_*.parquet"))
MULTI = {"convoy", "two_vehicle"}
OUT = os.path.join(HERE, "..", "scene_ceiling_result.json")


def agg_shard(p):
    cols = ["scene_id", "profile_id", "subkind", "coarse", "split"] + F.FEATURE_NAMES
    df = pd.read_parquet(p, columns=cols)
    df = df[df.coarse == "vehicle"]
    if not len(df):
        return None
    g = df.groupby("scene_id")
    a = g[F.FEATURE_NAMES].agg(["mean", "std", "min", "max"])
    a.columns = [f"{c}_{s}" for c, s in a.columns]
    p90 = g[F.FEATURE_NAMES].quantile(0.9)
    p90.columns = [f"{c}_p90" for c in p90.columns]
    meta = g.agg(profile_id=("profile_id", "first"), subkind=("subkind", "first"),
                 split=("split", "first"), n_win=("scene_id", "size"))
    return a.join(p90).join(meta).reset_index()


def main():
    t0 = time.time()
    print(f"aggregating {len(PARQ)} shards (vehicle scenes only) ...", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=7) as ex:
        for r in ex.map(agg_shard, PARQ):
            if r is not None:
                parts.append(r)
    S = pd.concat(parts, ignore_index=True)
    S = S[S.split == "train"].reset_index(drop=True)            # train profiles only (val untouched)
    y = S.subkind.isin(MULTI).astype(int).to_numpy()
    feat_cols = [c for c in S.columns if c not in
                 ("scene_id", "profile_id", "subkind", "split")]
    X = np.nan_to_num(S[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {len(S)} vehicle scenes | multiple={int(y.sum())} single={int((1-y).sum())} "
          f"| {len(feat_cols)} agg features | {(time.time()-t0)/60:.1f} min", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
    rng = np.random.default_rng(0)
    profs = S.profile_id.astype(str).to_numpy()
    uprof = np.unique(profs)
    hold = set(rng.choice(uprof, max(2, len(uprof) // 5), replace=False))   # 20% profiles held out
    te = np.isin(profs, list(hold))
    tr = ~te
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         class_weight="balanced", random_state=0)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    pred = clf.predict(X[te])
    ba = balanced_accuracy_score(y[te], pred)
    auc = roc_auc_score(y[te], proba)
    cm = confusion_matrix(y[te], pred)

    # quick permutation importance (top agg features) on the holdout
    base = ba
    Xte = X[te].copy(); imp = []
    for fi, nm in enumerate(feat_cols):
        sv = Xte[:, fi].copy(); Xte[:, fi] = rng.permutation(sv)
        imp.append((nm, float(base - balanced_accuracy_score(y[te], clf.predict(Xte)))))
        Xte[:, fi] = sv
    imp.sort(key=lambda t: -t[1])

    res = {"scene_bal_acc": float(ba), "scene_auc": float(auc),
           "per_window_baseline_bal_acc": 0.605,
           "n_scenes": int(len(S)), "n_multiple": int(y.sum()), "n_single": int((1 - y).sum()),
           "n_train_scenes": int(tr.sum()), "n_hold_scenes": int(te.sum()),
           "n_hold_profiles": len(hold), "confusion_rows_true": cm.tolist(),
           "top_agg_importance": imp[:20], "n_agg_features": len(feat_cols),
           "minutes": (time.time() - t0) / 60}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nSCENE-LEVEL single-vs-multi: bal-acc {ba:.3f}  AUC {auc:.3f}  "
          f"(per-window baseline 0.605)", flush=True)
    print(f"confusion (rows=true [single,multiple]): {cm.tolist()}", flush=True)
    print("top scene-agg features:", ", ".join(f"{n}({v:+.3f})" for n, v in imp[:10]), flush=True)
    print(f"VERDICT: {'TEMPORAL signal exists (scene >> window)' if ba > 0.70 else 'likely a WALL (scene ~ window)'}", flush=True)
    print(f"wrote {OUT} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
