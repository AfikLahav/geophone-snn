"""Three-zone analog-floored labeling — GENERATION_PLAN v1.2 Part B.

SNR computed at the HIGH-PRECISION geophone output (signal vs analog noise), the SAME
point the model trains on (NO quantizer). Floored by the analog noise so an undetectable
window contains no signal AND is labeled nothing, by construction.

zone: detectable (>=0 dB) / marginal (-20..0 dB, soft target sigmoid(SNR/6)) / nothing.
Per class -> ordinal {none, single, multiple} from the true source count.
"""
import numpy as np

FS = 1000.0
WIN, HOP = 3.0, 1.5
BANDS = {"common": (5, 90), "human": (20, 90), "animal": (20, 90), "vehicle": (5, 25)}


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
    if snr_db >= 0:
        return "detectable"
    if snr_db >= -20:
        return "marginal"
    return "nothing"


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
    ordinal level. clean_by_class: {class: signal_mv}; counts: {class: n_sources}."""
    out = {}
    for cls, sig in clean_by_class.items():
        lo, hi = BANDS[cls]
        cs = _band_rms(sig[i0:i0 + nw], lo, hi)
        ns = _band_rms(noise_mv[i0:i0 + nw], lo, hi) + 1e-12
        snr = 20 * np.log10(max(cs, 1e-12) / ns)
        present = snr >= 0
        lvl = ordinal(counts.get(cls, 0)) if present else "none"
        out[cls] = dict(snr_db=round(float(snr), 2), zone=zone(snr),
                        soft=round(soft_target(snr), 4), level=lvl,
                        n_sources=int(counts.get(cls, 0)))
    return out
