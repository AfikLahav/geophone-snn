"""Footstep force wavelet — v3 (A1 + A2', pilot-validated).

CHANGES vs v1 (dataset_validation/v3_harness/A1_A2prime_spec.md):
  A1  radiate the DYNAMIC impact only. The quasi-static weight W(t) is computed (it shapes Fx /
      is available for reference) but is NOT summed into the radiated normal force Fz — a quasi-static
      load does not radiate far-field seismic energy, and including it leaked a spurious sub-10 Hz lobe
      on stiff / HF-passing terrain (rock, clay).
  A2' heel-strike rise shifted to the seismic-transient range ~4-15 ms (FOOTWEAR-modulated via `sharp`,
      slow end CAPPED at 15 ms), replacing the old 5-50 ms. The old slow end (>15 ms) is a ~3-8 Hz bump,
      not a heel strike, and recorded below the real 20-90 Hz footstep band. Footwear variation is
      preserved: boot -> ~4-9 ms (HF-rich), cushioned -> ~10-15 ms (softer) — NOT a single constant.
"""
import numpy as np

FS = 1000.0


def _lobe(t, center, width, amp):
    x = (t - center) / width
    y = np.where(np.abs(x) <= 1, 0.5 * (1 + np.cos(np.pi * x)), 0.0)
    return amp * y


def _rise_a2p(rng, sharp):
    """A2' (PHYSICS-based; dataset_validation/v3_verify/human_source_HF_physics.md): heel-strike modeled as a
    Hertzian contact impact, shock-front contact time tau_c ~ U(2-8 ms), footwear-modulated. Source
    corner f_c ~ 0.34/tau_c (~40-170 Hz); the RECORDED HF then EMERGES via soil Q-attenuation x distance
    x geophone H(s) (high near/stiff, stripped far/soft) -- NOT curve-fit to human.csv (a near-field
    stiff-surface recording, dropped as the soil target)."""
    return float(np.clip(rng.uniform(0.002, 0.008) * (1.2 - 0.4 * sharp), 0.002, 0.010))


def walk_footfall(rng, mass, Tc=None, sharp=None):
    """One walking footfall. Returns (Fz, Fx) in Newtons. Fz = dynamic impact only (A1)."""
    bw = mass * 9.81
    Tc = Tc if Tc else rng.uniform(0.55, 0.65)
    sharp = sharp if sharp is not None else rng.uniform(0, 1)
    n = int(Tc * FS); t = np.arange(n) / FS

    # W(t): quasi-static weight curve (computed for reference; NOT radiated — A1)
    p1 = rng.uniform(1.00, 1.12) * bw; p2 = rng.uniform(0.90, 1.00) * bw
    W = (_lobe(t, 0.22 * Tc, 0.20 * Tc, p1) + _lobe(t, 0.75 * Tc, 0.22 * Tc, p2))
    trough = 0.75 * bw * np.exp(-((t - 0.5 * Tc) / (0.18 * Tc)) ** 2)
    W = np.maximum(W, trough * (t > 0.1 * Tc) * (t < 0.9 * Tc))

    # I(t): sharp heel-strike impact — A2' rise
    rise = _rise_a2p(rng, sharp)
    imp_amp = rng.uniform(0.3, 1.0) * (0.4 + 0.6 * sharp) * bw
    k = int(max(rise * 3, 0.006) * FS); ki = np.arange(min(k, n)); tau = rise * FS
    I = np.zeros(n); I[:len(ki)] = imp_amp * (ki / tau) * np.exp(1 - ki / tau)

    Fz = I                                          # A1: radiate the dynamic impact only

    # Fx(t): tangential friction (unchanged), true magnitude 0.10-0.20 BW
    fx_amp = rng.uniform(0.10, 0.20) * bw
    Fx = fx_amp * (np.exp(-((t - 0.20 * Tc) / (0.10 * Tc)) ** 2)
                   - np.exp(-((t - 0.78 * Tc) / (0.12 * Tc)) ** 2))
    Fx[:len(ki)] += rng.uniform(0.05, 0.12) * bw * (0.4 + 0.6 * sharp) * (ki / tau) * np.exp(1 - ki / tau)
    return Fz, Fx


def run_footfall(rng, mass, Tc=None, sharp=None):
    """One running footfall. Fz = sharp dynamic impact only (A1+A2'); running hits 1.6-3.1 BW."""
    bw = mass * 9.81
    Tc = Tc if Tc else rng.uniform(0.16, 0.30)
    sharp = sharp if sharp is not None else rng.uniform(0, 1)
    n = int(Tc * FS); t = np.arange(n) / FS
    rise = _rise_a2p(rng, sharp)
    amp = rng.uniform(1.6, 3.1) * bw
    k = int(max(rise * 3, 0.006) * FS); ki = np.arange(min(k, n)); tau = rise * FS
    I = np.zeros(n); I[:len(ki)] = amp * (ki / tau) * np.exp(1 - ki / tau)
    Fz = I
    fx_amp = rng.uniform(0.2, 0.5) * bw
    Fx = fx_amp * (np.exp(-((t - 0.25 * Tc) / (0.10 * Tc)) ** 2)
                   - np.exp(-((t - 0.75 * Tc) / (0.10 * Tc)) ** 2))
    return Fz, Fx
