"""Rebuild the label-noise cleaning for the field recordings.

The exclusion list produced by the original campaign was lost with a deleted temporary folder,
but the RULE is written down in full, so the cleaning is reproducible. The half that survives
(66 absence windows) is used here as a check on the implementation before the half that does not
(79 contamination windows) is trusted.

The rule, verbatim from the campaign write-up:

  Contamination (background files): exclude runs of >=2 consecutive windows with broadband
  loudness above a robust 99th percentile -- the 99th percentile of the values left after
  trimming anything beyond the median plus five median-absolute-deviations, computed per file.

  Absence (person / vehicle files): exclude windows that are BOTH below twice the paired
  background file's median class-band loudness AND have fewer than two structure features above
  the background's 95th percentile.
      footstep structure: cadence salience, envelope periodicity, impulse density, peakiness
      vehicle  structure: 5-25 Hz fraction, wheel-hop band, engine comb, beat depth

The model is never in the loop. This is analysis of the recordings, not of any prediction.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simgeo", "simgeo_v42"))
import features as F                                        # noqa: E402
from evaluation import prepare_real

PAIR = {"human.csv": "human_nothing.csv", "car.csv": "car_nothing.csv"}
BAND = {"human.csv": "frac_foot_20_90", "car.csv": "frac_veh_5_25"}
STRUCTURE = {
    "human.csv": ["cad_salience", "env_ac_strength", "impulse_density", "kurtosis"],
    "car.csv": ["frac_veh_5_25", "hop_band_frac", "comb_salience", "beat_depth"],
}


def robust_p99(x):
    """99th percentile of what is left after trimming beyond median + 5 median-absolute-deviations."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-12
    keep = x[x <= med + 5.0 * mad]
    return float(np.percentile(keep if len(keep) > 10 else x, 99))


def runs_of_true(mask, min_len=2):
    """Indices belonging to a run of at least `min_len` consecutive True values."""
    out = np.zeros(len(mask), bool)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def build(verbose=True):
    got = prepare_real.cached("elbit")
    if got is None:
        raise SystemExit("prepare the field recordings first")
    X, rows, _ = got
    names = list(F.FEATURE_NAMES)
    rec = np.array([r["recording"] for r in rows], dtype=object)
    t0 = np.array([r["t0_s"] for r in rows])
    col = {n: X[:, names.index(n)] for n in set(
        ["rms_total", "frac_foot_20_90", "frac_veh_5_25"]
        + [c for v in STRUCTURE.values() for c in v])}

    excluded = {}

    # --- contamination: background files that contain real activity ------------------
    for f in ("human_nothing.csv", "car_nothing.csv"):
        m = rec == f
        broad = col["rms_total"][m]
        thr = robust_p99(broad)
        order = np.argsort(t0[m])
        hot = runs_of_true(broad[order] > thr, min_len=2)
        idx = np.where(m)[0][order][hot]
        excluded[f] = [{"t0_s": float(t0[i]), "reason": "contamination"} for i in idx]

    # --- absence: subject files where the subject is not actually there ---------------
    for f in ("human.csv", "car.csv"):
        m = rec == f
        nm = rec == PAIR[f]
        band = BAND[f]
        # class-band loudness, as an absolute level rather than a fraction
        lvl = col[band] * col["rms_total"]
        quiet_med = float(np.median((col[band] * col["rms_total"])[nm]))
        faint = lvl[m] < 2.0 * quiet_med
        # structure: how many of this class's shape features rise above the background's 95th
        votes = np.zeros(int(m.sum()), int)
        for c in STRUCTURE[f]:
            p95 = float(np.percentile(col[c][nm], 95))
            votes += (col[c][m] > p95).astype(int)
        drop = faint & (votes < 2)
        idx = np.where(m)[0][drop]
        excluded[f] = [{"t0_s": float(t0[i]), "reason": "absence"} for i in idx]

    if verbose:
        tot = sum(len(v) for v in excluded.values())
        print(f"excluded {tot} of {len(rows)} windows ({100*tot/len(rows):.1f}%)")
        for f, v in excluded.items():
            print(f"  {f:<20} {len(v):>4}")
    return excluded, rows


def mask_for(excluded, rows):
    """Boolean keep-mask over the window list."""
    keep = np.ones(len(rows), bool)
    by = {}
    for f, items in excluded.items():
        by[f] = np.array([i["t0_s"] for i in items])
    for k, r in enumerate(rows):
        t = by.get(r["recording"])
        if t is not None and len(t) and np.min(np.abs(t - r["t0_s"])) < 1e-6:
            keep[k] = False
    return keep


if __name__ == "__main__":
    excluded, rows = build()
    # check against the surviving half of the original list
    ref = json.load(open(os.path.join(ROOT, "research", "outdoor_dataset", "elbit_cleaned",
                                      "absence_structure.json"), encoding="utf-8"))
    print("\ncheck against the original absence list that survives on disk:")
    for f, items in ref.items():
        want = {round(i["chunk"] * 30.0 + i["t_s"], 3) for i in items}
        got = {round(i["t0_s"], 3) for i in excluded.get(f, [])}
        print(f"  {f:<12} original {len(want):>3}   rebuilt {len(got):>3}   "
              f"matching {len(want & got):>3}")
    out = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out")) + "/elbit_cleaning.json"
    json.dump(excluded, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwritten to {out}")
