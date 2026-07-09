"""PHYSICAL-PRESENCE labeling with SNR as a difficulty covariate — GENERATION_PLAN v1.2 Part B.

GROUND TRUTH = PHYSICAL PRESENCE. `present`/`level` follow the TRUE source count: a window
in which the source is actually emitting (count >= 1) is labeled with that count's ordinal
level EVEN IF the realized SNR is faint or negative; a window with no active source is `none`.
Presence is NOT gated on an SNR threshold.

SNR (computed at the HIGH-PRECISION geophone output, signal vs analog noise — the SAME point
the model trains on, NO quantizer) is kept as a CONTINUOUS DIFFICULTY covariate, NOT a gate:
  - `snr_db`  : measured in-band SNR (the raw difficulty number).
  - `soft`    : soft_target = sigmoid(SNR/6), a graded loss weight that fades to ~0 only when a
                truly buried (very negative SNR) source is present — easy windows weight ~1,
                buried windows weight ~0. Independent of the presence/level label.
  - `zone`    : descriptive SNR band (detectable/marginal/nothing) for analysis. NOTE: `zone`
                describes the SNR difficulty, it does NOT null the level.

Per-class detection floors (car -12, human -8, animal -9 dB; see detection_thresholds.md) are
EVALUATION/analysis thresholds (for Pd-vs-SNR curves), exposed as DET_FLOOR_DB / eval_zone();
they are NOT used to gate the present/level label.
"""
import numpy as np

FS = 1000.0
WIN, HOP = 3.0, 1.5
BANDS = {"common": (5, 90), "human": (20, 90), "animal": (20, 90), "vehicle": (5, 25)}

# Per-class seismic detection floors in the project SNR convention (3 s in-band RMS at the
# sensor, no processing gain), from gap_study/v3_verify/detection_thresholds.md §2/§6. These
# are EVALUATION thresholds for downstream Pd-vs-SNR analysis ONLY — they do NOT gate labels.
DET_FLOOR_DB = {"vehicle": -12.0, "human": -8.0, "animal": -9.0}


def _band_rms(x, lo, hi, fs=FS):
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    X = np.fft.rfft(x)
    X[(f < lo) | (f > hi)] = 0
    return np.sqrt(np.mean(np.fft.irfft(X, n=n) ** 2))


def windows(clean_mv, noise_mv, fs=FS, win=WIN, hop=HOP):
    """Yield (t0, common_snr_db) per window. clean = signal-only at geophone output (mV,
    high precision); noise = analog noise (Johnson+amp+ambient) at output."""
    nw, nh = int(win * fs), int(hop * fs)
    lo, hi = BANDS["common"]
    for i0 in range(0, max(1, len(clean_mv) - nw + 1), nh):
        cs = _band_rms(clean_mv[i0:i0 + nw], lo, hi)
        ns = _band_rms(noise_mv[i0:i0 + nw], lo, hi) + 1e-12
        yield round(i0 / fs, 3), 20 * np.log10(max(cs, 1e-12) / ns)


def zone(snr_db):
    """Descriptive SNR-difficulty band (uniform floor). Analysis label only — it does NOT
    gate presence/level. detectable (>=0) / marginal (-20..0) / nothing (<-20)."""
    if snr_db >= 0:
        return "detectable"
    if snr_db >= -20:
        return "marginal"
    return "nothing"


def eval_zone(snr_db, cls):
    """Per-class detectability zone for DOWNSTREAM EVALUATION (Pd-vs-SNR), using the
    class-dependent detection floor DET_FLOOR_DB. above_floor (SNR >= class floor, a real
    detector is expected to pull it out) / below_floor (faint, physically present but
    sub-floor). Returns 'above_floor'/'below_floor'. NEVER used to null the level."""
    floor = DET_FLOOR_DB.get(cls, -20.0)
    return "above_floor" if snr_db >= floor else "below_floor"


def soft_target(snr_db):
    """sigmoid(SNR/6); 0 below the nothing floor."""
    if snr_db < -20:
        return 0.0
    return float(1.0 / (1.0 + np.exp(-snr_db / 6.0)))


def ordinal(count):
    """Per-class true count -> ordinal level."""
    return "none" if count == 0 else "single" if count == 1 else "multiple"


def per_class_window(clean_by_class, noise_mv, counts, i0, nw, fs=FS):
    """For one window: per-class SNR (in that class's diagnostic band), zone, soft target,
    ordinal level. clean_by_class: {class: signal_mv}; counts: {class: n_sources}.

    GROUND TRUTH = PHYSICAL PRESENCE: `level` is the ordinal of the TRUE source count and is
    INDEPENDENT of SNR. A faint/negative-SNR window with an active source still gets its true
    level (single/multiple); only a window with no active source (count == 0) is `none`. The
    measured `snr_db` and `soft` (sigmoid(SNR/6)) are stored as CONTINUOUS DIFFICULTY fields,
    not gates: `soft` fades to ~0 for a truly buried but present source, giving graded loss
    weighting without erasing the presence label."""
    out = {}
    for cls, sig in clean_by_class.items():
        lo, hi = BANDS[cls]
        cs = _band_rms(sig[i0:i0 + nw], lo, hi)
        ns = _band_rms(noise_mv[i0:i0 + nw], lo, hi) + 1e-12
        snr = 20 * np.log10(max(cs, 1e-12) / ns)
        # PHYSICAL PRESENCE: level follows the true source count, NOT the SNR. SNR is kept
        # only as the difficulty covariate (snr_db / soft / eval_zone), never as a gate.
        n = int(counts.get(cls, 0))
        lvl = ordinal(n)
        out[cls] = dict(snr_db=round(float(snr), 2), zone=zone(snr),
                        soft=round(soft_target(snr), 4), level=lvl,
                        n_sources=n)
    return out
