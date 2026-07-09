"""
ft_04_clipz.py — CLIPZ saturation analysis.

The model clips z-scores at ±8 (scaler.json 'clip'=8.0).
For each of the 104 model features, what fraction of REAL windows hit the ±8 rail?

A feature that clips frequently is one where the model always sees the same
(railed) value regardless of actual signal content → discriminative information is
thrown away.

Also reports the same for synth windows as a baseline (synth shouldn't clip much
since the scaler was fit on synth).
"""
import numpy as np


def clip_saturation(z_real, z_synth, feat_order, clip=8.0):
    """
    z_real  : (N_real,  104) z-scored real windows
    z_synth : (N_synth, 104) z-scored synth windows (using same scaler)
    feat_order: list[str] of 104 feature names

    Returns list of dicts per feature sorted by real clip fraction.
    """
    rows = []
    for i, fn in enumerate(feat_order):
        zr = z_real[:, i]
        zs = z_synth[:, i]
        rows.append({
            "feature":          fn,
            "real_frac_clipped":   float(np.mean(np.abs(zr) >= clip)),
            "real_frac_clip_pos":  float(np.mean(zr >=  clip)),
            "real_frac_clip_neg":  float(np.mean(zr <= -clip)),
            "synth_frac_clipped":  float(np.mean(np.abs(zs) >= clip)),
            "real_p99_abs_z":      float(np.percentile(np.abs(zr), 99)),
        })
    rows.sort(key=lambda x: x["real_frac_clipped"], reverse=True)
    return rows


def z_score_synth(synth_X, feat_names_synth, scaler):
    """Z-score synth_X using the scaler (same transform as applied to real)."""
    feat_order = scaler["features"]
    mean = np.array(scaler["mean"], dtype=np.float64)
    std  = np.array(scaler["std"],  dtype=np.float64)
    name2idx = {n: i for i, n in enumerate(feat_names_synth)}
    z_rows = []
    for i, fn in enumerate(feat_order):
        if fn not in name2idx:
            z_rows.append(np.zeros(len(synth_X)))
            continue
        col = synth_X[:, name2idx[fn]]
        z   = (col - mean[i]) / (std[i] + 1e-10)
        z_rows.append(z)
    return np.column_stack(z_rows)
