"""
ft_02_ood.py — Per-feature shift in synthetic-scaler z-units.

The model was trained with features z-scored using scaler.json (mean/std computed
over the 104 discriminative+is_rep features).  We apply those same z-scores to
the REAL windows and measure how far they land from the training distribution.

Metric: median |z| per feature over all real windows.
  |z| > 2 → out-of-distribution (the model sees this feature as shifted)
  |z| > 4 → severely out of distribution (beyond 2 sigma from the ±2 OOD boundary)

The scaler covers only the 104 model features.  We also z-score the full 132 via
the synth distribution (mean/std from synth_X) for completeness.
"""
import numpy as np


def apply_scaler(X_real, feat_names_real, scaler):
    """
    Apply scaler.json z-score to real windows for the 104 model features.

    X_real        : (N, 132) real feature matrix
    feat_names_real: list of 132 feature names (FEATURE_NAMES)
    scaler        : dict with keys 'mean', 'std', 'features' (104 names)

    Returns:
      z_real      : (N, 104) z-scored real windows
      feat_order  : list[str] of the 104 features in scaler order
    """
    feat_order = scaler["features"]
    mean = np.array(scaler["mean"], dtype=np.float64)
    std  = np.array(scaler["std"],  dtype=np.float64)
    clip = float(scaler.get("clip", 8.0))

    # index into real feature matrix
    name2idx = {n: i for i, n in enumerate(feat_names_real)}
    z_rows = []
    for i, fn in enumerate(feat_order):
        col = X_real[:, name2idx[fn]]
        z   = (col - mean[i]) / (std[i] + 1e-10)
        z_rows.append(z)
    z_real = np.column_stack(z_rows)   # (N, 104)
    return z_real, feat_order, clip


def ood_stats(z_real, feat_order, clip=8.0):
    """
    Per-feature OOD stats over all real windows.

    Returns list of dicts:
      feature, median_abs_z, p90_abs_z, frac_ood_2, frac_ood_4
    """
    rows = []
    for i, fn in enumerate(feat_order):
        absz = np.abs(z_real[:, i])
        rows.append({
            "feature":      fn,
            "median_abs_z": float(np.median(absz)),
            "p90_abs_z":    float(np.percentile(absz, 90)),
            "mean_abs_z":   float(np.mean(absz)),
            "frac_ood_2":   float(np.mean(absz > 2.0)),
            "frac_ood_4":   float(np.mean(absz > 4.0)),
        })
    rows.sort(key=lambda x: x["median_abs_z"], reverse=True)
    return rows


def family_ood(ood_rows, families, feat_names):
    """Per-family median of median_abs_z."""
    name2row = {r["feature"]: r for r in ood_rows}
    out = {}
    for fam, idxs in families.items():
        fnames = [feat_names[i] for i in idxs if feat_names[i] in name2row]
        if not fnames:
            continue
        vals = [name2row[fn]["median_abs_z"] for fn in fnames]
        frac2 = [name2row[fn]["frac_ood_2"] for fn in fnames]
        out[fam] = {
            "median_median_abs_z": float(np.median(vals)),
            "max_median_abs_z":    float(np.max(vals)),
            "mean_frac_ood_2":     float(np.mean(frac2)),
            "n_features_in_scaler": len(fnames),
        }
    return out
