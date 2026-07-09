"""Sensor chain + noise generators (PLAN Phase 1, NOISE_MODEL.md).

Chain (spec §3): v_ground [m/s] (+ ground-borne noise) -> coupling resonance ->
geophone H(iw) -> volts (+ electronic noise + lines) -> preamp gain -> ADS1015
quantize/clip -> amplitude [mV] @ 1000 Hz. All stages exact frequency-domain ops.
"""
import numpy as np

FS = 1000.0


def _w(n, fs=FS):
    return 2 * np.pi * np.fft.rfftfreq(n, 1 / fs)


def geophone_response(n, f0=4.5, h=0.6, G=28.8, fs=FS):
    """2nd-order high-pass velocity transducer, V per (m/s)."""
    w = _w(n, fs); w0 = 2 * np.pi * f0
    s = 1j * w
    return G * s**2 / (s**2 + 2 * h * w0 * s + w0**2)


def coupling_response(n, fc, qc, fs=FS):
    """2nd-order low-pass ground-coupling resonance ahead of the geophone."""
    w = _w(n, fs); wc = 2 * np.pi * fc
    s = 1j * w
    return wc**2 / (s**2 + (wc / qc) * s + wc**2)


def colored_noise(n, rng, slope=-1.0, fs=FS):
    """Unit-RMS colored noise, PSD ~ f^slope over 1 Hz..Nyquist."""
    f = np.fft.rfftfreq(n, 1 / fs)
    shape = np.zeros_like(f)
    m = f >= 0.5
    shape[m] = (np.maximum(f[m], 1.0)) ** (slope / 2)
    x = np.fft.irfft(shape * (rng.standard_normal(len(f)) +
                              1j * rng.standard_normal(len(f))), n=n)
    s = x.std()
    return x / s if s > 0 else x


def wind_noise(n, rng, wind_ms, veg=0.3, fs=FS):
    """NOISE_MODEL §1: shaped noise + von-Karman AM envelope. Returns ground m/s."""
    if wind_ms < 3.0:
        return np.zeros(n)
    base = colored_noise(n, rng, slope=rng.uniform(-2.0, -1.0), fs=fs)
    # level: bilinear slope, vegetation-steepened; anchored to a soft-site scale
    k = 0.4 + veg * rng.uniform(3.0, 4.6)               # dB per m/s
    level = 10 ** ((k * (wind_ms - 3.0)) / 20) * 2e-7    # m/s RMS scale (randomized below)
    level *= 10 ** rng.uniform(-0.5, 0.5)
    # gust AM envelope: low-passed noise, timescale T_L ~ 100/U
    TL = 100.0 / max(wind_ms, 1.0)
    env = colored_noise(n, rng, slope=0.0, fs=fs)
    kf = np.fft.rfftfreq(n, 1 / fs)
    E = np.fft.rfft(env); E[kf > 1 / (2 * np.pi * TL) * 6] = 0
    env = np.fft.irfft(E, n=n)
    env = np.exp(0.25 * env / max(env.std(), 1e-12))
    env /= env.mean()
    return base * env * level


def rain_noise(n, rng, rate_mmh, fs=FS):
    """NOISE_MODEL §2: >60 Hz Poisson impact shot noise, PSD ~ R^alpha."""
    if rate_mmh <= 0.5:
        return np.zeros(n)
    rate = 30.0 * rate_mmh                               # impacts/s reaching the sensor area
    n_imp = rng.poisson(rate * n / fs)
    x = np.zeros(n)
    t_idx = rng.integers(0, n, n_imp)
    amp = rng.lognormal(0, 0.5, n_imp)
    for t, a in zip(t_idx, amp):                         # ~10 ms 100 Hz wavelets
        L = 12
        if t + L < n:
            x[t:t + L] += a * np.exp(-np.arange(L) / 4.0) * np.sin(2 * np.pi * 100 *
                                                                    np.arange(L) / fs)
    # band-limit onset ~60 Hz (sigmoid gate) + level ~ R^alpha
    f = np.fft.rfftfreq(n, 1 / fs)
    gate = 1 / (1 + np.exp(-(f - 60.0) / 10.0))
    x = np.fft.irfft(np.fft.rfft(x) * gate, n=n)
    alpha = rng.uniform(1.2, 1.9)
    lvl = 1.5e-8 * (rate_mmh ** (alpha / 2)) * 10 ** rng.uniform(-0.3, 0.3)
    s = x.std()
    return x * (lvl / s) if s > 0 else x


def site_lines(n, rng, p_lines=1.0, fs=FS):
    """v2: terrain-gated 50 Hz mains (Israel grid). With prob p_lines: 50/100/150 Hz + grid-freq
    jitter + random phase/amplitude; off-grid (1-p_lines): zeros. The v1 44-47/55 Hz rig lines
    (confirmed PC-fan contamination) are removed; the 75 Hz pump is now machinery (sources_v2)."""
    if rng.random() >= p_lines:
        return np.zeros(n)                               # off-grid: clean floor only
    t = np.arange(n) / fs
    x = np.zeros(n)
    f0 = rng.normal(50.0, 0.05)                          # grid frequency wander
    a50 = 0.05 * 10 ** rng.uniform(-0.5, 1.0)
    for k in (1, 2, 3):                                  # 50 / 100 / 150 Hz
        x += (a50 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
    return x


def spurious_bump(n, f_spur=150.0, q_spur=20.0, spur_db=12.0, fs=FS):
    """v2: geophone parasitic-mode resonance (>=140 Hz, review B1). Lorentzian peak of
    magnitude A=10^(dB/20) at f_spur, ->1 away. Multiplies H (shapes signal AND ground noise)."""
    f = np.fft.rfftfreq(n, 1 / fs)
    A = 10 ** (spur_db / 20.0); fr = np.maximum(f, 1e-9)
    detune = q_spur * (fr / f_spur - f_spur / fr)
    return 1.0 + (A - 1.0) / np.sqrt(1.0 + detune ** 2)


def electronic_floor(n, rng, gain, e_white=12e-9, f_c=10.0, f_rc=175.0, jitter_dec=0.2, fs=FS):
    """v2 physics electronic floor at OUTPUT (mV): 1/f rise <f_c, flat plateau, gentle 1-pole
    roll-off >f_rc (NO ADC cap). Level = input-referred e_white [V/sqrt Hz] x sqrt(eff noise BW)
    x gain. Replaces v1's flat white+0.3*pink (the 'too white' floor)."""
    f = np.fft.rfftfreq(n, 1 / fs); fr = np.maximum(f, 1e-9)
    asd = np.sqrt(1.0 + f_c / fr) / np.sqrt(1.0 + (f / f_rc) ** 2); asd[0] = asd[1]
    spec = asd * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    x = np.fft.irfft(spec, n=n); x /= (x.std() + 1e-30)
    eff_bw = float(np.sum(asd ** 2) * (fs / n))
    rms_out_V = e_white * np.sqrt(eff_bw) * gain * 10 ** rng.uniform(-jitter_dec, jitter_dec)
    return (x * rms_out_V * 1e3).astype(np.float64)      # mV


def render_hp(v_signal, v_ground_noise, rng, profile_fc, profile_q, p_lines=0.5, sensor=None):
    """v2 HIGH-PRECISION render — coupling x 4.5 Hz geophone x spurious resonance (>=140 Hz);
    physics electronic floor + terrain-gated 50 Hz lines; NO ADC quantizer. Returns
    (out_mv, clean_mv, noise_mv, params). The p=dict(...) block stays the FIRST rng consumer so
    clean_mv stays reproducible; all floor/line draws come strictly after it (review A2)."""
    n = len(v_signal)
    p = dict(f0=rng.normal(4.5, 0.09), h=rng.normal(0.6, 0.05),
             G=rng.normal(28.8, 1.4), gain=10 ** rng.uniform(2.3, 2.9),
             fc=profile_fc * 10 ** rng.uniform(-0.15, 0.15), qc=profile_q,
             f_spur=max(float(rng.normal(150, 15)), 140.0),
             q_spur=float(rng.uniform(10, 40)), spur_db=float(rng.uniform(6, 18)),
             **(sensor or {}))
    H = (coupling_response(n, p["fc"], p["qc"]) *
         geophone_response(n, p["f0"], p["h"], p["G"]) *
         spurious_bump(n, p["f_spur"], p["q_spur"], p["spur_db"]))
    sig_v = np.fft.irfft(np.fft.rfft(v_signal) * H, n=n)
    noi_v = np.fft.irfft(np.fft.rfft(v_ground_noise) * H, n=n)
    clean_mv = (sig_v * p["gain"] * 1e3).astype(np.float32)
    floor_mv = electronic_floor(n, rng, p["gain"])         # physics floor at output mV
    lines_mv = site_lines(n, rng, p_lines)                 # terrain-gated 50 Hz
    p["lines_present"] = int(np.any(lines_mv != 0))        # realized flag (for leakage QA)
    noise_mv = (noi_v * p["gain"] * 1e3 + floor_mv + lines_mv).astype(np.float32)
    return (clean_mv + noise_mv).astype(np.float32), clean_mv, noise_mv, p


def render(v_signal, v_ground_noise, rng, profile_fc, profile_q, sensor=None):
    """Full chain: ground velocity (m/s) -> amplitude [mV] @ 1000 Hz.
    Signal and ground noise pass through the SAME sampled chain separately, so the
    per-window SNR downstream is exact. Returns (amplitude_mV, clean_mV, params)."""
    n = len(v_signal)
    p = dict(f0=rng.normal(4.5, 0.09), h=rng.normal(0.6, 0.05),
             G=rng.normal(28.8, 1.4), gain=10 ** rng.uniform(2.3, 2.9),
             fc=profile_fc * 10 ** rng.uniform(-0.15, 0.15),
             qc=profile_q, **(sensor or {}))
    H = (coupling_response(n, p["fc"], p["qc"]) *
         geophone_response(n, p["f0"], p["h"], p["G"]))
    sig_v = np.fft.irfft(np.fft.rfft(v_signal) * H, n=n)
    noi_v = np.fft.irfft(np.fft.rfft(v_ground_noise) * H, n=n)
    # electronic noise at volts (pre-gain), fitted to rig floor 0.08 mV RMS at output
    floor_v = 0.08e-3 / p["gain"] * 10 ** rng.uniform(-0.2, 0.2)
    noi_v = noi_v + floor_v * (colored_noise(n, rng, 0.0) +
                               0.3 * colored_noise(n, rng, -1.0))
    clean_mv = sig_v * p["gain"] * 1e3
    noise_mv = noi_v * p["gain"] * 1e3 + site_lines(n, rng)
    out_mv = clean_mv + noise_mv
    # ADS1015: 0.125 mV LSB, clip +-256 mV
    out_mv = np.clip(np.round(out_mv / 0.125) * 0.125, -256.0, 256.0)
    return out_mv.astype(np.float32), clean_mv.astype(np.float32), p
