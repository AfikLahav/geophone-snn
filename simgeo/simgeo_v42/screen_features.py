"""Feature screening on the full-corpus feature table — produces the data behind
FEATURE_ANALYSIS.md.

Methodology:
  - TRAIN split only (90/10 by profile; the 29 val profiles are never touched).
  - Window class = the class whose ordinal level != none (scenes are single-coarse);
    'nothing' = all-none windows. Class windows are detectable by construction
    (level requires SNR >= 0 dB).
  - Per-feature AUC (Mann-Whitney rank) per boundary:
      class-vs-class (3), class-vs-nothing (3), single-vs-multiple per class (3),
      faint-class-vs-noise (3): windows in subject scenes with class SNR in [-20,0)
      (labeled none, signal faintly present) vs windows from nothing scenes.
  - Robustness: per-terrain-family AUC for the 6 class boundaries (a feature that
    only works on one family is a shortcut risk).
  - Multivariate: HistGradientBoosting 4-class on detectable+nothing windows with a
    PROFILE-GROUPED internal holdout (10% of train profiles) -> confusion matrix,
    balanced accuracy, permutation importance. Same for single-vs-multi per class.
  - Sides capped at CAP windows (AUC error ~ +-0.002); caps logged, nothing silent.

Outputs to N:\\geophone_synth\\features\\screening\\:
  auc_global.csv, auc_family.csv, auc_faint.csv, gbt_results.json, meta.json
Usage: python screen_features.py [features_dir]
"""
import os, sys, json, glob, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import FEATURE_NAMES

CAP = 600_000          # max windows per boundary side for rank-AUC
SEED = 0


def rank_auc(x, y01):
    """AUC of feature x for labels y01 (1=positive) via Mann-Whitney ranks."""
    from scipy.stats import rankdata
    r = rankdata(x)
    n1 = int(y01.sum()); n0 = len(y01) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y01 == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cap_idx(idx, rng):
    return rng.choice(idx, CAP, replace=False) if len(idx) > CAP else idx


def main():
    fdir = sys.argv[1] if len(sys.argv) > 1 else r"G:/geophone_synth/features_v2"
    odir = os.path.join(fdir, "screening")
    os.makedirs(odir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t00 = time.time()

    files = sorted(glob.glob(os.path.join(fdir, "features_shard_*.parquet")))
    print(f"loading {len(files)} parquet files ...", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df)} windows, {(time.time()-t00)/60:.1f} min", flush=True)

    tr = df[df["split"] == "train"].reset_index(drop=True)
    del df
    meta = {"n_train_windows": int(len(tr)), "cap_per_side": CAP, "seed": SEED}

    hl = tr["human_level"].to_numpy(); vl = tr["vehicle_level"].to_numpy()
    al = tr["animal_level"].to_numpy()
    coarse = tr["coarse"].astype(str).to_numpy()
    fam = tr["family"].astype(str).to_numpy()
    noth = (hl == 0) & (vl == 0) & (al == 0)
    cls_w = {"human": hl > 0, "vehicle": vl > 0, "animal": al > 0}
    # faint: subject-scene windows with class SNR in [-20,0) (labeled none)
    faint = {c: (coarse == c) & (tr[f"{c}_snr"].to_numpy() >= -20)
                 & (tr[f"{c}_snr"].to_numpy() < 0) & ~cls_w[c] for c in cls_w}
    noise_w = coarse == "nothing"          # true-noise windows (nothing scenes)

    bounds = {}
    for a, b in (("human", "animal"), ("human", "vehicle"), ("animal", "vehicle")):
        # a-only vs b-only — exclude co-occurring (mixed) windows so the pairwise is clean
        bounds[f"{a}_vs_{b}"] = (cls_w[a] & ~cls_w[b], cls_w[b] & ~cls_w[a])
    for c in cls_w:
        bounds[f"{c}_vs_nothing"] = (cls_w[c], noth)
    lvl = {"human": hl, "vehicle": vl, "animal": al}
    for c in cls_w:
        bounds[f"{c}_multi_vs_single"] = (lvl[c] == 2, lvl[c] == 1)
    faint_bounds = {f"{c}_faint_vs_noise": (faint[c], noise_w) for c in cls_w}

    X = tr[FEATURE_NAMES].to_numpy(np.float32)

    def screen(bdict, fname, with_family):
        rows = []
        for bname, (mp, mn) in bdict.items():
            ip = cap_idx(np.where(mp)[0], rng); im = cap_idx(np.where(mn)[0], rng)
            if len(ip) < 50 or len(im) < 50:
                print(f"  SKIP {bname}: n={len(ip)}/{len(im)}", flush=True)
                continue
            idx = np.concatenate([ip, im])
            y = np.concatenate([np.ones(len(ip)), np.zeros(len(im))])
            sub = X[idx]
            t0 = time.time()
            for fi, nm in enumerate(FEATURE_NAMES):
                rows.append((bname, "ALL", nm, rank_auc(sub[:, fi], y),
                             len(ip), len(im)))
            if with_family:
                bf = fam[idx]
                for fm in np.unique(bf):
                    m = bf == fm
                    if y[m].sum() < 50 or (1 - y[m]).sum() < 50:
                        continue
                    sm = sub[m]; ym = y[m]
                    for fi, nm in enumerate(FEATURE_NAMES):
                        rows.append((bname, fm, nm, rank_auc(sm[:, fi], ym),
                                     int(ym.sum()), int((1 - ym).sum())))
            print(f"  {bname}: n={len(ip)}/{len(im)} {(time.time()-t0):.0f}s", flush=True)
        out = pd.DataFrame(rows, columns=["boundary", "family", "feature", "auc",
                                          "n_pos", "n_neg"])
        out.to_csv(os.path.join(odir, fname), index=False)
        return out

    print("AUC screening (global + per-family) ...", flush=True)
    g = screen(bounds, "auc_global_and_family.csv", with_family=True)
    print("AUC screening (faint vs noise) ...", flush=True)
    screen(faint_bounds, "auc_faint.csv", with_family=False)

    # ----------------------------------------------------------- multivariate
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    prof = tr["profile_id"].astype(str).to_numpy()
    profs = np.unique(prof)
    hold_p = set(rng.choice(profs, max(2, len(profs) // 10), replace=False))
    hold = np.isin(prof, list(hold_p))
    meta["internal_holdout_profiles"] = sorted(hold_p)
    gbt_out = {}

    n_act = (hl > 0).astype(int) + (vl > 0).astype(int) + (al > 0).astype(int)
    ycls = np.full(len(tr), 3)             # 0 h, 1 v, 2 a, 3 nothing (single-class windows only)
    ycls[cls_w["human"]] = 0; ycls[cls_w["vehicle"]] = 1; ycls[cls_w["animal"]] = 2

    def fit_gbt(mask, y, name, cap_per_class=300_000):
        idx = np.where(mask)[0]
        keep = []
        for c in np.unique(y[idx]):
            ic = idx[y[idx] == c]
            keep.append(rng.choice(ic, cap_per_class, replace=False)
                        if len(ic) > cap_per_class else ic)
        idx = np.concatenate(keep)
        itr = idx[~hold[idx]]; ite = idx[hold[idx]]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        t0 = time.time()
        clf.fit(X[itr], y[itr])
        pred = clf.predict(X[ite])
        ba = balanced_accuracy_score(y[ite], pred)
        cm = confusion_matrix(y[ite], pred)
        res = {"balanced_acc": float(ba), "confusion": cm.tolist(),
               "n_train": int(len(itr)), "n_hold": int(len(ite)),
               "fit_min": (time.time() - t0) / 60}
        if len(np.unique(y)) == 2:
            res["auc"] = float(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
        # permutation importance on a holdout subsample
        sub = rng.choice(ite, 80_000, replace=False) if len(ite) > 80_000 else ite
        base = balanced_accuracy_score(y[sub], clf.predict(X[sub]))
        imp = []
        Xs = X[sub].copy()
        for fi, nm in enumerate(FEATURE_NAMES):
            sv = Xs[:, fi].copy()
            Xs[:, fi] = rng.permutation(sv)
            imp.append((nm, float(base - balanced_accuracy_score(
                y[sub], clf.predict(Xs)))))
            Xs[:, fi] = sv
        imp.sort(key=lambda t: -t[1])
        res["perm_importance_top25"] = imp[:25]
        res["base_holdout_ba"] = float(base)
        gbt_out[name] = res
        print(f"  GBT {name}: bal-acc {ba:.3f} ({res['fit_min']:.1f} min fit)", flush=True)

    # per-class PRESENT binary (the multi-head analog; robust to co-occurrence / mixed scenes,
    # since each head detects its class regardless of what else is firing in the window)
    allm = np.ones(len(tr), bool)
    for c in cls_w:
        print(f"multivariate GBT ({c}-present) ...", flush=True)
        fit_gbt(allm, cls_w[c].astype(int), f"{c}_present")
    # 4-class coarse on single-class + nothing windows only (mixed excluded; v1-comparable, 0.963)
    print("multivariate GBT (4-class, single-class windows) ...", flush=True)
    fit_gbt((n_act == 1) | noth, ycls, "coarse_4class_singleonly")
    for c in cls_w:
        m = (lvl[c] == 1) | (lvl[c] == 2)
        print(f"multivariate GBT ({c} single-vs-multi) ...", flush=True)
        fit_gbt(m, (lvl[c] == 2).astype(int), f"{c}_multi")

    json.dump(gbt_out, open(os.path.join(odir, "gbt_results.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(odir, "meta.json"), "w"), indent=1)
    print(f"SCREENING DONE in {(time.time()-t00)/60:.1f} min -> {odir}", flush=True)


if __name__ == "__main__":
    main()
