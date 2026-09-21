# wf_common.py
# Shared loaders + windowed metrics for the REAL-vs-SYNTHETIC v4 time-domain
# discrepancy study. Everything here is deterministic and pointed at read-only data.
#
# Conventions (from mission spec):
#   REAL csv 'amplitude' is in VOLTS.
#     mV       = volts * 1000        <- primary analysis unit
#     pipeline = volts * 25.4        <- the ML pipeline's (units_check) convention
#   SYNTH model input (mV) = clip(clean_mv + clean_mv2? + noise_mv, -256, +256)
#
# fs assumed 1000 Hz unless the file's own time column says otherwise.
import os, sqlite3
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))

BASE = os.environ.get("PROJECT_ROOT", ".")
SYNTH_DB = os.path.join(_GEO_ROOT, "dataset_v431/shard_0.sqlite")
CLIP_MV = 256.0
WIN_S = 3.0  # 3-second windows

# ---------------------------------------------------------------- real loaders
def load_real_volts(path):
    """Return (time_s, amp_volts) float64 arrays from a real geophone CSV."""
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(dtype=np.float64)
    a = df["amplitude"].to_numpy(dtype=np.float64)
    return t, a

def measure_fs(t):
    """Median sample rate + jitter stats from a time column."""
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    med = float(np.median(dt))
    fs = 1.0 / med if med > 0 else float("nan")
    return {
        "fs_hz": fs,
        "dt_median_s": med,
        "dt_std_s": float(np.std(dt)),
        "dt_min_s": float(np.min(dt)),
        "dt_max_s": float(np.max(dt)),
        "jitter_ms": float(np.std(dt) * 1000.0),
        "frac_gap_gt_1p5x": float(np.mean(dt > 1.5 * med)),
        "n_samples": int(len(t) + 1),
    }

# ------------------------------------------------------------- synth loaders
def _blob(b):
    return None if b is None else np.frombuffer(b, dtype=np.float32).astype(np.float64)

def synth_model_input(cur, scene_id):
    """Reconstruct the clipped model-input waveform (mV) for one scene."""
    w = cur.execute(
        "SELECT noise_mv, clean_mv, clean_mv2 FROM waveforms WHERE scene_id=?",
        (scene_id,),
    ).fetchone()
    if w is None:
        return None
    n, cl, cl2 = _blob(w[0]), _blob(w[1]), _blob(w[2])
    if n is None:
        return None
    L = len(n)
    tot = n.copy()
    if cl is not None:
        tot = tot + cl[:L]
    if cl2 is not None:
        tot = tot + cl2[:L]
    unclipped = tot
    clipped = np.clip(tot, -CLIP_MV, CLIP_MV)
    return clipped, unclipped

def synth_scene_ids(cur, coarse, n, seed=0):
    ids = [r[0] for r in cur.execute(
        "SELECT scene_id FROM scenes WHERE coarse=?", (coarse,)).fetchall()]
    rng = np.random.default_rng(seed)
    if len(ids) > n:
        ids = list(rng.choice(ids, size=n, replace=False))
    return [int(x) for x in ids]

# ------------------------------------------------------------- windowing/metrics
def windows(x, win_len):
    """Non-overlapping windows as a 2D array (n_win, win_len); drops remainder."""
    n = (len(x) // win_len) * win_len
    if n == 0:
        return np.empty((0, win_len))
    return x[:n].reshape(-1, win_len)

def window_metrics(x, fs, win_s=WIN_S):
    """Per-window rms(mV-scale of input), excess kurtosis, crest factor, distinct
    levels. x is already in the target amplitude unit. Returns dict of arrays."""
    win_len = int(round(fs * win_s))
    if win_len < 8:
        win_len = 8
    W = windows(x, win_len)
    if W.shape[0] == 0:
        return None
    rms = np.sqrt(np.mean(W * W, axis=1))
    # crest: peak / rms about zero (model sees zero-referenced amplitude)
    peak = np.max(np.abs(W), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 0, peak / rms, np.nan)
    kurt = sp_kurtosis(W, axis=1, fisher=True, bias=False)
    # distinct levels per window
    ndist = np.array([len(np.unique(row)) for row in W], dtype=np.float64)
    return {
        "win_len": win_len,
        "n_win": int(W.shape[0]),
        "rms": rms,
        "crest": crest[np.isfinite(crest)],
        "kurt": kurt[np.isfinite(kurt)],
        "ndist": ndist,
    }

def pct(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}

def cv_of_rms(rms):
    rms = np.asarray(rms, dtype=np.float64)
    m = np.mean(rms)
    return float(np.std(rms) / m) if m > 0 else float("nan")

# ---------------------------------------------------------------- quantization
def estimate_lsb(a):
    """Empirical LSB (smallest recurring amplitude step) and effective bits.
    a in the file's native unit (volts for real)."""
    u = np.unique(np.round(a, 9))
    if len(u) < 3:
        return {"lsb": None, "n_distinct": int(len(u)), "eff_bits": None,
                "range": float(np.ptp(a))}
    d = np.diff(np.sort(u))
    d = d[d > 1e-9]
    if d.size == 0:
        return {"lsb": None, "n_distinct": int(len(u)), "eff_bits": None,
                "range": float(np.ptp(a))}
    lsb = float(np.min(d))
    rng = float(np.ptp(a))
    eff_bits = float(np.log2(rng / lsb)) if lsb > 0 and rng > 0 else None
    return {"lsb": lsb, "n_distinct": int(len(u)), "eff_bits": eff_bits, "range": rng}

# ---------------------------------------------------------------- clipping/rail
def real_rail_fraction(a):
    """Fraction of samples pinned at the observed extreme code (both rails)."""
    amax, amin = float(np.max(a)), float(np.min(a))
    rail = max(abs(amax), abs(amin))
    n = len(a)
    frac_exact = float((np.sum(a == amax) + np.sum(a == amin)) / n)
    frac_near = float(np.mean(np.abs(a) >= 0.999 * rail)) if rail > 0 else 0.0
    return {"rail_abs": rail, "pos_rail": amax, "neg_rail": amin,
            "frac_at_exact_rail": frac_exact, "frac_ge_999pct_rail": frac_near}

def synth_clip_fraction(unclipped):
    """Fraction of model-input samples that hit the +-256 mV clip."""
    return float(np.mean(np.abs(unclipped) >= CLIP_MV))
