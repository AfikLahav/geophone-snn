"""Sensor chain + noise generators (PLAN Phase 1, NOISE_MODEL.md).

Chain (spec §3): v_ground [m/s] (+ ground-borne noise) -> coupling resonance ->
geophone H(iw) -> volts (+ electronic noise + lines) -> preamp gain -> ADS1015
quantize/clip -> amplitude [mV] @ 1000 Hz. All stages exact frequency-domain ops.
"""
import os
import numpy as np

FS = 1000.0
# Coupling model: "lowpass" (v4 baseline, 2nd-order low-pass -> rolls off above fc) or
# "bump" (resonance PEAK on flat unity response -> keeps energy above fc). The domain-gap
# analysis showed the low-pass anchored to the measured ~53 Hz over-attenuates the 55-90 Hz
# human band (synth 0.16 vs real 0.47 upper-band fraction); the bump form matches real (0.51)
# while keeping the C0-measured resonance frequency. See dataset_validation/exp_coupling.json.
COUPLING_MODE = os.environ.get("GEO_COUPLING_MODE", "lowpass")
COUPLING_BUMP_DB = float(os.environ.get("GEO_COUPLING_BUMP_DB", "8.0"))
# v4.1 amplitude/line-fidelity flags (DISCREPANCY_REPORT §7-A). Defaults reproduce v4.
GAIN_LOG10 = tuple(float(x) for x in os.environ.get("GEO_GAIN_LOG10", "2.3,2.9").split(","))
SPUR_P = float(os.environ.get("GEO_SPUR_P", "1.0"))          # P(spurious 150 Hz resonance active)
LINES_MODE = os.environ.get("GEO_LINES_MODE", "v4")          # "v41" = mains level in dB ABOVE FLOOR

# v4.2 sensor-chain domain-randomization spans (Change 3). Nuisances become RANDOMIZED axes the
# model must be invariant to (not calibrated away). Defaults reproduce v4 with NO extra rng draws
# (the draws are conditional -> bit-identical stream when the flags are unset). All are drawn in
# render_hp's per-scene p dict, class-independent by construction.
#   GEO_COUPLING_MODE="mix" -> per-scene lowpass OR bump (50/50) on top of anchor_fc.
#   GEO_QUANT_MIX="0,0.125,0.2,0.25" -> per-scene ADC LSB (mV); 0 = continuous. Applied to the
#     MODEL-INPUT SUM only (never the per-class clean blobs -> D2 + SNR labels stay exact).
#   GEO_RAIL_MIX="inf,512,256" -> per-scene ADC full-scale rail (mV); inf = never clips.
_QUANT_MIX = os.environ.get("GEO_QUANT_MIX")
_RAIL_MIX = os.environ.get("GEO_RAIL_MIX")
QUANT_LEVELS = ([float(x) for x in _QUANT_MIX.split(",")] if _QUANT_MIX else None)
RAIL_LEVELS = ([(float("inf") if x.strip() in ("inf", "0", "") else float(x))
                for x in _RAIL_MIX.split(",")] if _RAIL_MIX else None)


def _w(n, fs=FS):
    return 2 * np.pi * np.fft.rfftfreq(n, 1 / fs)


def geophone_response(n, f0=4.5, h=0.6, G=28.8, fs=FS):
    """2nd-order high-pass velocity transducer, V per (m/s)."""
    w = _w(n, fs); w0 = 2 * np.pi * f0
    s = 1j * w
    return G * s**2 / (s**2 + 2 * h * w0 * s + w0**2)


def coupling_response(n, fc, qc, fs=FS, form=None):
    """Ground-coupling resonance ahead of the geophone. Mode set by GEO_COUPLING_MODE, or the
    per-scene `form` override ('lowpass'/'bump') when GEO_COUPLING_MODE=='mix' (v4.2 Change 3).
    lowpass: 2nd-order low-pass (unity DC, -40 dB/dec above fc).
    bump:    resonance peak on flat unity (=1 away from fc) -> preserves the upper band.
    The corpus render AND the per-class H-rebuild in gen_scene MUST pass the SAME form."""
    mode = form if form is not None else COUPLING_MODE
    if mode == "bump":
        f = np.fft.rfftfreq(n, 1 / fs); fr = np.maximum(f, 1e-9)
        A = 10 ** (COUPLING_BUMP_DB / 20.0)
        detune = qc * (fr / fc - fc / fr)
        return 1.0 + (A - 1.0) / np.sqrt(1.0 + detune ** 2)
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


def site_lines(n, rng, p_lines=1.0, fs=FS, floor_rms=None):
    """v2: terrain-gated 50 Hz mains (Israel grid). With prob p_lines: 50/100/150 Hz + grid-freq
    jitter + random phase/amplitude; off-grid (1-p_lines): zeros. The v1 44-47/55 Hz rig lines
    (confirmed PC-fan contamination) are removed; the 75 Hz pump is now machinery (sources_v2).

    v4.1 (LINES_MODE=='v41'): mains level defined RELATIVE to the realized electronic floor
    (dB-above-floor draw, real rigs measured ~19-25 dB) instead of fixed mV. Keeps the whole
    noise stack proportional to gain -> per-window SNR stays gain-invariant."""
    if rng.random() >= p_lines:
        return np.zeros(n)                               # off-grid: clean floor only
    t = np.arange(n) / fs
    x = np.zeros(n)
    f0 = rng.normal(50.0, 0.05)                          # grid frequency wander
    if LINES_MODE == "v41" and floor_rms is not None:
        a50 = float(floor_rms) * 10 ** (rng.uniform(8.0, 22.0) / 20.0) * np.sqrt(2)
    else:
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


def electronic_floor(n, rng, gain, e_white=1.5e-9, f_c=10.0, f_rc=175.0, jitter_dec=0.2, fs=FS):
    """v2 physics electronic floor at OUTPUT (mV): 1/f rise <f_c, flat plateau, gentle 1-pole
    roll-off >f_rc (NO ADC cap). Level = input-referred e_white [V/sqrt Hz] x sqrt(eff noise BW)
    x gain. Replaces v1's flat white+0.3*pink (the 'too white' floor).

    v3 (C1f): e_white LOWERED 12e-9 -> 1.5e-9 V/sqrtHz so the colored r3 ground shows through on a
    typical (median) draw instead of a near-white instrument floor carrying ~6x the in-band power
    (S4). Lowering the floor — NOT re-coloring r3 (N5) — restores the reddened calm slope.
    CAVEAT: this RAISES the realized ground:floor ratio and therefore lowers the realized calm
    noise floor — it changes the realized SNR baseline; flag for the scene/SNR agent."""
    f = np.fft.rfftfreq(n, 1 / fs); fr = np.maximum(f, 1e-9)
    asd = np.sqrt(1.0 + f_c / fr) / np.sqrt(1.0 + (f / f_rc) ** 2); asd[0] = asd[1]
    spec = asd * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    x = np.fft.irfft(spec, n=n); x /= (x.std() + 1e-30)
    eff_bw = float(np.sum(asd ** 2) * (fs / n))
    rms_out_V = e_white * np.sqrt(eff_bw) * gain * 10 ** rng.uniform(-jitter_dec, jitter_dec)
    return (x * rms_out_V * 1e3).astype(np.float64)      # mV


# ADS1015 full-scale rail at the recorded gain stage (D5). Hard clip the SUMMED signal at
# +-256 mV (NO 0.125 mV quantize per project decision). Component blobs stay UNCLIPPED so the
# downstream SNR labels (clean_mv vs noise_mv) remain exact.
ADC_CLIP_MV = 256.0


def render_hp(v_signal, v_ground_noise, rng, profile_fc, profile_q, p_lines=0.5, sensor=None):
    """v2 HIGH-PRECISION render — coupling x 4.5 Hz geophone x spurious resonance (>=140 Hz);
    physics electronic floor + terrain-gated 50 Hz lines. v3 (D5): ADS1015 hard rail restored —
    the RETURNED out_mv is clipped to +-256 mV (no quantize); clean_mv/noise_mv blobs stay
    unclipped so SNR labels stay exact. Returns (out_mv, clean_mv, noise_mv, params). The
    p=dict(...) block stays the FIRST rng consumer so clean_mv stays reproducible; all floor/line
    draws come strictly after it (review A2)."""
    n = len(v_signal)
    p = dict(f0=rng.normal(4.5, 0.09), h=rng.normal(0.6, 0.05),
             G=rng.normal(28.8, 1.4), gain=10 ** rng.uniform(*GAIN_LOG10),
             fc=profile_fc * 10 ** rng.uniform(-0.15, 0.15), qc=profile_q,
             f_spur=max(float(rng.normal(150, 15)), 140.0),
             q_spur=float(rng.uniform(10, 40)), spur_db=float(rng.uniform(6, 18)),
             **(sensor or {}))
    if SPUR_P < 1.0 and rng.random() >= SPUR_P:            # v4.1: spurious mode probabilistic
        p["spur_db"] = 0.0
    # v4.2 Q_j draws (conditional -> no rng shift at v4 defaults). coupling_form MUST be drawn
    # before H (it selects the coupling filter); quant/rail affect only the recorded model-input
    # SUM (out_mv), never the per-class clean blobs. All independent of scene class.
    p["coupling_form"] = (("bump" if rng.random() < 0.5 else "lowpass")
                          if COUPLING_MODE == "mix" else COUPLING_MODE)
    p["quant_lsb"] = float(rng.choice(QUANT_LEVELS)) if QUANT_LEVELS is not None else 0.0
    p["rail_mv"] = float(rng.choice(RAIL_LEVELS)) if RAIL_LEVELS is not None else float(ADC_CLIP_MV)
    H = (coupling_response(n, p["fc"], p["qc"], form=p["coupling_form"]) *
         geophone_response(n, p["f0"], p["h"], p["G"]) *
         spurious_bump(n, p["f_spur"], p["q_spur"], p["spur_db"]))
    sig_v = np.fft.irfft(np.fft.rfft(v_signal) * H, n=n)
    noi_v = np.fft.irfft(np.fft.rfft(v_ground_noise) * H, n=n)
    clean_mv = (sig_v * p["gain"] * 1e3).astype(np.float32)
    floor_mv = electronic_floor(n, rng, p["gain"])         # physics floor at output mV
    lines_mv = site_lines(n, rng, p_lines, floor_rms=float(np.std(floor_mv)))  # terrain-gated 50 Hz
    p["lines_present"] = int(np.any(lines_mv != 0))        # realized flag (for leakage QA)
    noise_mv = (noi_v * p["gain"] * 1e3 + floor_mv + lines_mv).astype(np.float32)
    # v4.2: per-scene ADS rail + optional quantization on the recorded SUM (component blobs stay
    # unclipped/unquantized -> exact SNR + D2). rail_mv=inf -> no clip; quant_lsb=0 -> continuous.
    # (out_mv is not stored by the generator; the SAME per-scene rail/quant are re-applied to the
    # reconstructed model input at feature time via the recorded rail_mv/quant_lsb columns.)
    out = clean_mv.astype(np.float64) + noise_mv.astype(np.float64)
    if np.isfinite(p["rail_mv"]):
        out = np.clip(out, -p["rail_mv"], p["rail_mv"])
    if p["quant_lsb"] > 0:
        out = np.round(out / p["quant_lsb"]) * p["quant_lsb"]
    return out.astype(np.float32), clean_mv, noise_mv, p


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
