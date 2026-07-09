"""
ft_01_auroc.py — Per-feature domain AUROC (real=1 vs synth=0, all windows pooled).

For every one of the 132 features:
  AUROC = probability that a randomly chosen real window scores higher than a
          randomly chosen synth window.  AUROC=0.5 → indistinguishable.
          AUROC→1 or →0 → highly separable (domain discrepancy).

We report:
  - Full ranked table of all 132 features (by |AUROC - 0.5| descending)
  - Per-family median AUROC
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_domain_auroc(synth_X, real_X, feat_names):
    """
    synth_X : (N_synth, 132)
    real_X  : (N_real,  132)
    Returns : dict feature_name -> auroc (float)
    """
    # Labels: synth=0, real=1
    y = np.concatenate([np.zeros(len(synth_X)), np.ones(len(real_X))])
    results = {}
    for i, fn in enumerate(feat_names):
        scores = np.concatenate([synth_X[:, i], real_X[:, i]])
        # guard NaN/inf
        finite = np.isfinite(scores)
        if finite.sum() < 10 or y[finite].sum() < 5 or (1 - y[finite]).sum() < 5:
            results[fn] = float("nan")
            continue
        try:
            auc = roc_auc_score(y[finite], scores[finite])
        except Exception:
            auc = float("nan")
        results[fn] = float(auc)
    return results


def family_auroc(auroc_dict, families, feat_names):
    """Per-family median AUROC."""
    out = {}
    for fam, idxs in families.items():
        vals = [auroc_dict[feat_names[i]] for i in idxs
                if feat_names[i] in auroc_dict and
                np.isfinite(auroc_dict[feat_names[i]])]
        if vals:
            out[fam] = {
                "median_auroc": float(np.median(vals)),
                "max_auroc":    float(np.max(vals)),
                "n_features":   len(vals),
                "max_deviation_from_half": float(np.max(np.abs(np.array(vals) - 0.5))),
            }
    return out


def ranked_table(auroc_dict):
    """List of (feature, auroc, deviation) sorted by |auroc-0.5| descending."""
    rows = []
    for fn, auc in auroc_dict.items():
        if np.isfinite(auc):
            rows.append((fn, auc, abs(auc - 0.5)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows
