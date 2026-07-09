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
    p = os.path.join(HERE, "..", "datasets", "noise_atlas", "r3_fits.npz")
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


# ---------------------------------------------------------------- C1/C2 helpers
def _lowpass_env(n, fc_hz, rng, fs):
    """Unit-std low-passed Gaussian (slow random envelope), corner fc_hz."""
    kf = np.fft.rfftfreq(n, 1 / fs)
    E = np.fft.rfft(rng.standard_normal(n))
    E[kf > fc_hz] = 0.0
    e = np.fft.irfft(E, n=n)
    return e / (e.std() + 1e-12)


def _impulsive_transients(n, rng, fs, dens_per_s, amp_scale, band=(15.0, 90.0)):
    """C1a: micro-transients -> non-Gaussian band-passed velocity (Groos & Ritter 2009:
    ~40-50% of windows transient/positive-kurtosis dominated, crest factor 1.3-3.1). Each
    transient is a short damped HF burst at a random time; the whole train is band-limited to
    the soil micro-tremor band. Density is set so ~40-50% of 3 s windows catch a transient;
    amplitudes are moderately heavy-tailed so windows are non-Gaussian but DON'T blow the crest
    factor past ~3. Returned series has UNIT std (scaled by caller)."""
    n_imp = rng.poisson(max(dens_per_s, 0.0) * n / fs)
    x = np.zeros(n)
    if n_imp == 0:
        return x
    t_idx = rng.integers(0, n, n_imp)
    # moderately heavy-tailed amplitudes (lognormal, capped) -> positive kurtosis without a
    # runaway crest factor. amp_scale ~0.4-0.7 keeps the tail bounded.
    amps = np.clip(rng.lognormal(0.0, amp_scale, n_imp), None, 2.5)
    for t, a in zip(t_idx, amps):
        L = int(rng.integers(8, 40))                          # 8-40 ms burst
        if t + L >= n:
            continue
        fk = rng.uniform(*band)                               # micro-transient carrier
        tau = rng.uniform(2.0, 8.0)                           # ms decay
        k = np.arange(L)
        x[t:t + L] += a * np.exp(-k / tau) * np.sin(2 * np.pi * fk * k / fs +
                                                    rng.uniform(0, 6.28))
    # band-limit the whole impulse train (soil strips the very-HF content)
    f = np.fft.rfftfreq(n, 1 / fs)
    gate = ((f >= band[0] * 0.5) & (f <= band[1] * 1.5)).astype(float)
    x = np.fft.irfft(np.fft.rfft(x) * gate, n=n)
    s = x.std()
    return x / s if s > 1e-30 else x


def cultural_lines(n, rng, fs, n_extra=None):
    """C2: parametric cultural/narrowband vocabulary on the GROUND-velocity noise (m/s,
    UNIT std). Adds mains harmonics (50/100/150 Hz, grid-freq wander) AND site-idiosyncratic
    machinery tones (a few narrow lines at random RPM-driven frequencies + harmonics) so the
    'nothing' PSD-shape pool stops collapsing to ~8 smooth templates (V3 §C2). Each scene
    draws a different line set, so the PSD-shape PDF broadens with site-idiosyncratic peaks."""
    t = np.arange(n) / fs
    x = np.zeros(n)
    # mains family (most sites): 50 Hz + harmonics with grid-frequency wander
    if rng.random() < 0.7:
        f0 = rng.normal(50.0, 0.05)
        a0 = 10 ** rng.uniform(-1.0, 0.0)
        for k in (1, 2, 3):
            x += (a0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
    # machinery / cultural tones: 0-3 site-idiosyncratic narrowband lines (RPM-driven,
    # NORESS sawmill 6/12/17 Hz, pump bands, traffic) + a couple of harmonics each
    n_mach = rng.integers(0, 4) if n_extra is None else n_extra
    for _ in range(int(n_mach)):
        fm = float(rng.uniform(4.0, 80.0))                    # motor/pump fundamental
        am = 10 ** rng.uniform(-1.3, -0.3)
        nh = int(rng.integers(1, 4))
        for k in range(1, nh + 1):
            if fm * k < fs / 2:
                x += (am / k) * np.sin(2 * np.pi * fm * k * t + rng.uniform(0, 6.28))
    s = x.std()
    return x / s if s > 1e-30 else x


def ground_noise(n, wind_ms, rain_mmh, rng, base_level=None, fs=1000.0):
    """R3-anchored ambient/weather ground-velocity noise (m/s), length n.

    v3 (C1/C2): the colored r3 background is now (a) made NON-STATIONARY by a slow RMS
    drift (RMS-CoV 0.2-0.6), (b) made NON-GAUSSIAN/impulsive by sparse micro-transients
    (~40-50% of windows non-Gaussian), and (c) broadened by parametric cultural lines
    (mains + machinery tones). The wind gust AM is re-enabled for all wind tiers. Color
    (slope) is unchanged (N5: r3 is already colored; do NOT re-color)."""
    _load()
    cond = condition_of(wind_ms, rain_mmh)
    f_src, psd_src = _POOLS[cond][rng.integers(len(_POOLS[cond]))]
    shape = np.sqrt(_psd_on_grid(f_src, psd_src, n, fs))      # amplitude spectrum
    spec = shape * (rng.standard_normal(len(shape)) + 1j * rng.standard_normal(len(shape)))
    x = np.fft.irfft(spec, n=n)
    x /= (x.std() + 1e-30)                                    # unit-std colored r3 base

    # C2: parametric cultural lines (mains + site machinery), small relative to background
    x = x + rng.uniform(0.05, 0.30) * cultural_lines(n, rng, fs)

    # C1a: impulsive micro-transients on a DISTRIBUTION (not every window). Density set so
    # ~40-50% of the 3 s windows catch a transient; capped lognormal amps -> positive kurtosis
    # with crest factor in the 1.3-3.1 band. Weight kept MODEST so the transients lift kurtosis
    # without making the peak/RMS ratio (crest) blow past ~3 (a band-passed Gaussian already
    # sits near crest 3.4, so the headroom is small).
    dens = rng.uniform(0.5, 1.3)                              # transients/s
    timp = _impulsive_transients(n, rng, fs, dens, amp_scale=rng.uniform(0.3, 0.55))
    x = x + rng.uniform(0.28, 0.55) * timp

    x /= (x.std() + 1e-30)                                    # renormalize to unit std

    # C1b: slow RMS drift (non-stationarity) applied LAST (multiplicative on the combined
    # series) so it is not diluted by the additive transient/line terms. Diurnal/episodic
    # envelope; sig/mu drawn so RMS-CoV lands in the 0.2-0.6 band (Groos & Ritter 6-25 dB swing).
    drift_fc = rng.uniform(0.04, 0.14)                        # Hz (7-25 s timescale)
    cov_drive = rng.uniform(0.45, 0.8)
    denv = _lowpass_env(n, drift_fc, rng, fs)
    drift = np.clip(1.0 + cov_drive * denv, 0.1, None)
    x = x * drift
    x /= (x.std() + 1e-30)

    # level: PHYSICALLY-ANCHORED base x R3 condition ratio x per-scene jitter.
    # CALIBRATION (2026-06-16, SNR-absolute-scale fix): the prior base 10**U(-9.0,-8.2)
    # (median ~6e-9 m/s ground-velocity RMS) was ~37 dB BELOW real near-surface ambient.
    # Measured: that floor left synthetic targets +10..+70 dB at 250 m where a real human
    # is gone past ~50 m (the SNR-vs-distance crossings never landed at the field ranges).
    # The off quantity is the AMBIENT floor (geophone GAIN cancels in SNR; the only ambient
    # DOF is this level). Re-anchored to a moderately-cultural near-surface site: realized
    # broadband (1-120 Hz) ground-velocity RMS ~2.3e-6 m/s, in-band(20-90 Hz) ~1e-6 m/s.
    # PUBLISHED near-surface vertical ambient velocity RMS: ~1e-7..1e-6 m/s at quiet-moderate
    # land sites, rising to ~1e-5 m/s at culturally noisy sites (Peterson 1993 NLNM/NHNM
    # extended to 100 Hz for very-quiet desert; Bormann NMSOP cultural-noise band; the R3
    # in-band weather corpus). This base lands the SNR-distance crossings at the published
    # field detection ranges (human ~15-50 m, vehicle hundreds of m, animal tens of m;
    # Koc & Yegin 2013; Pakhomov SPIE 5071/5417). Median 10**-5.6 = 2.5e-6 m/s.
    CALIB_BASE_LOG10 = -5.6                                # median 2.5e-6 m/s (was -8.6)
    base = base_level if base_level is not None else 10 ** rng.uniform(
        CALIB_BASE_LOG10 - 0.2, CALIB_BASE_LOG10 + 0.2)
    level = base * np.sqrt(_LEVELS[cond]) * 10 ** rng.uniform(-0.25, 0.25)
    x = x * level

    # wind: R3-measured gust AM (sigma/mu 0.5-0.9), timescale T_L ~ 100/U (re-enabled,
    # F19). Adds gusty non-stationarity on top of the diurnal drift for wind tiers.
    if cond.startswith("wind"):
        TL = 100.0 / max(wind_ms, 1.0)
        env = _lowpass_env(n, 1 / (2 * np.pi * TL) * 6, rng, fs)
        sig_mu = rng.uniform(0.5, 0.9)                        # R3-measured
        env = 1 + sig_mu * env
        x = x * np.clip(env, 0.05, None)
    return x.astype(np.float64)
