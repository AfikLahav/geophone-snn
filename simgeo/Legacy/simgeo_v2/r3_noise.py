"""R3-anchored ambient/weather noise — uses the REAL fitted PSDs from the 55 GB fetch
(datasets/noise_atlas/r3_fits.npz) instead of literature placeholders.

For each scene's condition (calm / wind tier / rain), draw a fitted real PSD SHAPE (pooled
across the YW/ZG/LASSO/IS sites = site randomization), generate ground-motion noise with that
shape, scale by the R3-measured per-condition level ratio anchored to the rig floor, and (wind)
modulate with the R3-measured gust-AM intensity sigma/mu 0.5-0.9. Returns ground velocity (m/s)
added to v_ground BEFORE the sensor chain.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_FITS = None
_POOLS = None          # condition -> list of (f, psd_normalized)
_LEVELS = None         # condition -> level ratio vs calm (from R3 median band power)
COND_BINS = ["calm", "wind_low", "wind_mid", "wind_high", "rain_light", "rain_heavy"]
# targets with usable in-band coverage (Nyquist): zg/lasso 250 Hz, yw 125, is 20
TARGETS = ["yw", "zg", "lasso", "is_il"]


def _load():
    global _FITS, _POOLS, _LEVELS
    if _POOLS is not None:
        return
    p = os.path.join(HERE, "..", "..", "..", "datasets", "noise_atlas", "r3_fits.npz")
    _FITS = dict(np.load(p))
    _POOLS = {c: [] for c in COND_BINS}
    band_pow = {c: [] for c in COND_BINS}
    for tgt in TARGETS:
        for c in COND_BINS:
            kf, km = f"{tgt}__{c}__f", f"{tgt}__{c}__med"
            if kf in _FITS and km in _FITS:
                f = _FITS[kf]; psd = _FITS[km]
                if f.max() < 15 or not np.all(np.isfinite(psd)) or psd.max() <= 0:
                    continue                                  # too low-band (IS) for shape use
                m = (f >= 1) & (f <= 20)                       # common band all sites share
                bp = psd[m].mean()
                # normalize shape to unit power in 1-20 Hz
                _POOLS[c].append((f.copy(), psd / (bp + 1e-30)))
                band_pow[c].append(bp)
    # per-condition level ratio vs calm (median band power), capped sane
    base = np.median([np.median(band_pow[c]) for c in ("calm",) if band_pow[c]] or [1.0])
    _LEVELS = {}
    for c in COND_BINS:
        if band_pow[c]:
            _LEVELS[c] = float(np.clip(np.median(band_pow[c]) / (base + 1e-30), 0.5, 30.0))
        else:
            _LEVELS[c] = 1.0
    # fallback pools: if a condition had no usable site, borrow calm shape
    for c in COND_BINS:
        if not _POOLS[c]:
            _POOLS[c] = _POOLS["calm"] or [(np.linspace(1, 250, 200), np.ones(200))]


def condition_of(wind_ms, rain_mmh):
    if rain_mmh > 5:
        return "rain_heavy"
    if rain_mmh > 0.5:
        return "rain_light"
    if wind_ms >= 10:
        return "wind_high"
    if wind_ms >= 6:
        return "wind_mid"
    if wind_ms >= 3:
        return "wind_low"
    return "calm"


def _psd_on_grid(f_src, psd_src, n, fs):
    """Interpolate fitted PSD (log-log) onto rfft grid; extrapolate >f_src.max() with -2 slope."""
    f = np.fft.rfftfreq(n, 1 / fs)
    out = np.zeros_like(f)
    m = f >= 1
    lf = np.log10(np.clip(f[m], 1, None))
    out[m] = 10 ** np.interp(lf, np.log10(np.clip(f_src, 1, None)),
                             np.log10(np.clip(psd_src, 1e-30, None)),
                             left=np.log10(psd_src[0] + 1e-30),
                             right=np.log10(psd_src[-1] + 1e-30))
    # HF extrapolation: roll off as f^-2 above the source Nyquist
    hi = f > f_src.max()
    if hi.any():
        out[hi] = out[f <= f_src.max()][-1] * (f[hi] / f_src.max()) ** -2
    return out


def ground_noise(n, wind_ms, rain_mmh, rng, base_level=None, fs=1000.0):
    """R3-anchored ambient/weather ground-velocity noise (m/s), length n."""
    _load()
    cond = condition_of(wind_ms, rain_mmh)
    f_src, psd_src = _POOLS[cond][rng.integers(len(_POOLS[cond]))]
    shape = np.sqrt(_psd_on_grid(f_src, psd_src, n, fs))      # amplitude spectrum
    spec = shape * (rng.standard_normal(len(shape)) + 1j * rng.standard_normal(len(shape)))
    x = np.fft.irfft(spec, n=n)
    x /= (x.std() + 1e-30)
    # level: rig-anchored base x R3 condition ratio x per-scene jitter
    base = base_level if base_level is not None else 10 ** rng.uniform(-9.0, -8.2)
    level = base * np.sqrt(_LEVELS[cond]) * 10 ** rng.uniform(-0.25, 0.25)
    x = x * level
    # wind: R3-measured gust AM (sigma/mu 0.5-0.9), timescale T_L ~ 100/U
    if cond.startswith("wind"):
        TL = 100.0 / max(wind_ms, 1.0)
        kf = np.fft.rfftfreq(n, 1 / fs)
        E = np.fft.rfft(rng.standard_normal(n))
        E[kf > 1 / (2 * np.pi * TL) * 6] = 0
        env = np.fft.irfft(E, n=n)
        sig_mu = rng.uniform(0.5, 0.9)                        # R3-measured
        env = 1 + sig_mu * env / (env.std() + 1e-12)
        x = x * np.clip(env, 0.05, None)
    return x.astype(np.float64)
