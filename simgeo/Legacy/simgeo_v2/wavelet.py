"""Footstep force wavelet — GENERATION_PLAN v1.2 Part H (corrected mechanism).

Two force components at TRUE magnitudes (NO inflation); the terrain GF bank's site
response shapes the recorded 20-90 Hz band:
  Fz(t) = W(t) weight curve (sub-10 Hz)  +  I(t) sharp heel-strike impact (the band driver)
  Fx(t) = tangential/friction (braking->propulsion + stick-slip), true magnitude 0.10-0.20 BW

The 20-90 Hz seismic band comes from the NORMAL-force impact I(t) amplified by the site
transfer function (Ekimov & Sabatier 2006) — NOT from inflating friction.
"""
import numpy as np

FS = 1000.0


def _lobe(t, center, width, amp):
    """Raised-cosine lobe (smooth, sub-10 Hz)."""
    x = (t - center) / width
    y = np.where(np.abs(x) <= 1, 0.5 * (1 + np.cos(np.pi * x)), 0.0)
    return amp * y


def walk_footfall(rng, mass, Tc=None, sharp=None):
    """One walking footfall. Returns (Fz, Fx) arrays in Newtons, length = Tc*FS.
    sharp in [0,1]: footwear sharpness (1=boot fast rise, 0=cushioned slow)."""
    bw = mass * 9.81
    Tc = Tc if Tc else rng.uniform(0.55, 0.65)
    sharp = sharp if sharp is not None else rng.uniform(0, 1)
    n = int(Tc * FS)
    t = np.arange(n) / FS

    # W(t): asymmetric double-hump weight curve (1st ~1.04, 2nd ~0.94 BW, trough ~0.75)
    p1 = rng.uniform(1.00, 1.12) * bw
    p2 = rng.uniform(0.90, 1.00) * bw
    W = (_lobe(t, 0.22 * Tc, 0.20 * Tc, p1) +
         _lobe(t, 0.75 * Tc, 0.22 * Tc, p2))
    # fill the mid-stance trough so it doesn't go to zero (~0.75 BW)
    W = np.maximum(W, _lobe(t, 0.5 * Tc, 0.5 * Tc, 0.75 * bw) * 0.0 + 0.0)
    trough = 0.75 * bw * np.exp(-((t - 0.5 * Tc) / (0.18 * Tc)) ** 2)
    W = np.maximum(W, trough * (t > 0.1 * Tc) * (t < 0.9 * Tc))

    # I(t): sharp heel-strike impact at touchdown — THE 20-90 Hz driver
    rise = rng.uniform(0.005, 0.050) * (1.4 - 0.8 * sharp)      # boot faster
    rise = max(rise, 0.003)
    imp_amp = rng.uniform(0.3, 1.0) * (0.4 + 0.6 * sharp) * bw
    k = int(max(rise * 3, 0.006) * FS)
    ki = np.arange(min(k, n))
    # critically-damped impulse: fast rise, exponential decay
    tau = rise * FS
    I = np.zeros(n)
    I[:len(ki)] = imp_amp * (ki / tau) * np.exp(1 - ki / tau)

    Fz = W + I

    # Fx(t): tangential friction, TRUE magnitude 0.10-0.20 BW, braking->propulsion + stick-slip
    fx_amp = rng.uniform(0.10, 0.20) * bw
    Fx = fx_amp * (np.exp(-((t - 0.20 * Tc) / (0.10 * Tc)) ** 2)
                   - np.exp(-((t - 0.78 * Tc) / (0.12 * Tc)) ** 2))
    # sharp stick-slip transient co-located with heel-strike (true magnitude, small)
    Fx[:len(ki)] += rng.uniform(0.05, 0.12) * bw * (0.4 + 0.6 * sharp) * \
        (ki / tau) * np.exp(1 - ki / tau)
    return Fz, Fx


def run_footfall(rng, mass, Tc=None, sharp=None):
    bw = mass * 9.81
    Tc = Tc if Tc else rng.uniform(0.16, 0.30)
    sharp = sharp if sharp is not None else rng.uniform(0, 1)
    n = int(Tc * FS); t = np.arange(n) / FS
    rearfoot = rng.random() < 0.7
    if rearfoot:                                                # double-hump: impact + active
        imp = rng.uniform(1.4, 1.9) * bw
        act = rng.uniform(2.2, 3.1) * bw
        rise = max(rng.uniform(0.004, 0.020) * (1.4 - 0.8 * sharp), 0.003)
        tau = rise * FS; ki = np.arange(min(int(rise * 4 * FS), n))
        I = np.zeros(n); I[:len(ki)] = imp * (ki / tau) * np.exp(1 - ki / tau)
        W = _lobe(t, 0.45 * Tc, 0.45 * Tc, act)
        Fz = W + I
    else:                                                       # forefoot single-hump
        Fz = _lobe(t, 0.45 * Tc, 0.45 * Tc, rng.uniform(2.0, 3.1) * bw)
    fx_amp = rng.uniform(0.2, 0.5) * bw
    Fx = fx_amp * (np.exp(-((t - 0.25 * Tc) / (0.10 * Tc)) ** 2)
                   - np.exp(-((t - 0.75 * Tc) / (0.10 * Tc)) ** 2))
    return Fz, Fx
