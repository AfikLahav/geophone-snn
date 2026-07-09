"""
ft_03_classconditional.py — Class-conditional domain AUROC.

Repeats domain AUROC restricted to matched class pairs:
  - real human   vs synth human
  - real vehicle vs synth vehicle
  - real nothing vs synth nothing

High AUROC within a class pair → the discrepancy is in the class signal itself
Low AUROC within pairs but high overall → the discrepancy is in background/noise

Synth class map: 'coarse' column in synthetic df
  0 = nothing, 1 = human, 2 = vehicle  (from simgeo label convention)
Real class map: labels from ft_00_load.CSV_LABEL_MAP
"""
import numpy as np
from sklearn.metrics import roc_auc_score

# synth coarse labels are strings: 'human', 'vehicle', 'nothing'
PAIRS = [("human", "human"), ("vehicle", "vehicle"), ("nothing", "nothing")]


def conditional_auroc(synth_df, synth_X, real_X, real_labels, feat_names):
    """
    Returns dict:
      class_pair -> {feature -> auroc}
    """
    results = {}
    for rname, sc in PAIRS:
        # synth mask (coarse is a string column e.g. 'human', 'vehicle', 'nothing')
        if "coarse" not in synth_df.columns:
            print(f"[WARN] 'coarse' column not in synth_df; skipping class-conditional")
            break
        smask = np.array(synth_df["coarse"].astype(str) == str(sc))
        sx = synth_X[smask]
        # real mask
        rmask = real_labels == rname
        rx = real_X[rmask]
        if len(sx) < 10 or len(rx) < 10:
            print(f"[WARN] insufficient data for pair {rname}: synth={len(sx)}, real={len(rx)}")
            results[rname] = {}
            continue
        print(f"[class-cond] {rname}: synth={len(sx)}, real={len(rx)}")
        y = np.concatenate([np.zeros(len(sx)), np.ones(len(rx))])
        pair_auroc = {}
        for i, fn in enumerate(feat_names):
            scores = np.concatenate([sx[:, i], rx[:, i]])
            finite = np.isfinite(scores)
            if finite.sum() < 10 or y[finite].sum() < 3 or (1-y[finite]).sum() < 3:
                pair_auroc[fn] = float("nan")
                continue
            try:
                pair_auroc[fn] = float(roc_auc_score(y[finite], scores[finite]))
            except Exception:
                pair_auroc[fn] = float("nan")
        results[rname] = pair_auroc
    return results


def class_conditional_summary(cond_results, feat_names):
    """
    For each class pair, median AUROC and top-5 most-separable features.
    Also flags: is overall_auroc >> max(class_auroc)?
    """
    summary = {}
    for cls, auroc_dict in cond_results.items():
        vals = [v for v in auroc_dict.values() if np.isfinite(v)]
        devs = [(fn, auc, abs(auc-0.5)) for fn, auc in auroc_dict.items() if np.isfinite(auc)]
        devs.sort(key=lambda x: x[2], reverse=True)
        summary[cls] = {
            "median_auroc":  float(np.median(vals)) if vals else None,
            "mean_auroc":    float(np.mean(vals))   if vals else None,
            "frac_above_0p7": float(np.mean(np.array(vals) > 0.7)) if vals else None,
            "top5_separable": [{"feature": fn, "auroc": auc} for fn, auc, _ in devs[:5]],
        }
    return summary
