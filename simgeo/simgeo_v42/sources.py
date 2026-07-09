"""Source force models F(t) in Newtons (PLAN Phase 3, spec §4-5).

BINDING (likeness probe + FootprintID): every footfall wavelet includes
(a) the sharp heel-strike impact transient and (b) a tangential/friction
component (radiated via the Fx bank) — these create the 10-40 Hz band.
Each source returns a list of "emissions": (t_start, x, y, Fz(t), Fx(t)),
each emission radiated from its own position (per-footfall GF / retarded time).
"""
import numpy as np

FS = 1000.0


def _grf_walk(mass, contact_s, rng, sharp):
    """One walking footfall: double-hump weight curve + heel-strike impact transient.
    sharp in [0,1] = footwear sharpness (boot 1, barefoot/cushioned 0)."""
    n = int(contact_s * FS)
    t = np.arange(n) / n
    bw = mass * 9.81
    hump = (np.exp(-((t - 0.25) / 0.13) ** 2) +
            rng.uniform(0.9, 1.0) * np.exp(-((t - 0.75) / 0.15) ** 2))
    fz = bw * rng.uniform(1.0, 1.2) * hump / hump.max()
    # heel-strike impact: ms-scale transient at touchdown (THE 10-40+ Hz source)
    imp_len = max(int(rng.uniform(0.008, 0.030) * (1.5 - sharp) * FS), 4)
    k = np.arange(imp_len)
    impact = np.exp(-k / (imp_len / 3)) * np.sin(np.pi * k / imp_len)
    fz[:imp_len] += bw * rng.uniform(0.3, 1.0) * (0.4 + 0.6 * sharp) * impact
    # tangential (friction) force: braking then propulsion, sharper than the hump
    fx = bw * rng.uniform(0.15, 0.25) * (np.exp(-((t - 0.18) / 0.08) ** 2) -
                                         np.exp(-((t - 0.82) / 0.10) ** 2))
    fx[:imp_len] += bw * rng.uniform(0.1, 0.3) * (0.4 + 0.6 * sharp) * impact
    return fz, fx


def _grf_run(mass, contact_s, rng, sharp):
    n = int(contact_s * FS)
    t = np.arange(n) / n
    bw = mass * 9.81
    fz = bw * rng.uniform(2.0, 2.9) * np.exp(-((t - 0.45) / 0.22) ** 2)
    imp_len = max(int(rng.uniform(0.006, 0.020) * (1.5 - sharp) * FS), 4)
    k = np.arange(imp_len)
    impact = np.exp(-k / (imp_len / 3)) * np.sin(np.pi * k / imp_len)
    fz[:imp_len] += bw * rng.uniform(0.8, 1.6) * (0.4 + 0.6 * sharp) * impact
    fx = bw * rng.uniform(0.2, 0.35) * (np.exp(-((t - 0.25) / 0.1) ** 2) -
                                        np.exp(-((t - 0.75) / 0.1) ** 2))
    fx[:imp_len] += bw * rng.uniform(0.15, 0.4) * impact
    return fz, fx


def human(path_xy, t_path, rng, gait="walk"):
    """Footfall train along a path. path_xy: (npts,2) m; t_path: (npts,) s."""
    mass = rng.uniform(50, 100)
    sharp = rng.uniform(0, 1)                            # footwear knob
    cadence = rng.uniform(1.6, 2.2) if gait == "walk" else rng.uniform(2.6, 3.2)
    contact = rng.uniform(0.6, 0.7) if gait == "walk" else rng.uniform(0.15, 0.3)
    grf = _grf_walk if gait == "walk" else _grf_run
    emissions = []
    t = t_path[0] + rng.uniform(0, 0.5)
    while t < t_path[-1]:
        i = np.searchsorted(t_path, t)
        x, y = path_xy[min(i, len(path_xy) - 1)]
        fz, fx = grf(mass, contact, rng, sharp)
        emissions.append((t, x, y, fz, fx))
        t += 1.0 / cadence * rng.normal(1.0, 0.03)       # cadence jitter
    return emissions, dict(mass=mass, cadence=cadence, sharp=sharp, gait=gait)


def animal(path_xy, t_path, rng, kind="dog"):
    """Quadruped: 4-beat footfall sequence, allometric scaling (spec §5)."""
    mass = {"dog": rng.uniform(10, 30), "boar": rng.uniform(50, 120),
            "horse": rng.uniform(400, 700)}[kind]
    stride = {"dog": rng.uniform(2.0, 3.5), "boar": rng.uniform(1.5, 2.5),
              "horse": rng.uniform(0.8, 1.5)}[kind]
    contact = float(np.clip(0.10 * (mass / 20) ** 0.14, 0.08, 0.5))
    per_foot = mass * 9.81 / 4 * rng.uniform(1.0, 1.5)
    emissions = []
    t = t_path[0] + rng.uniform(0, 0.5)
    beat = 1.0 / (stride * 4)                            # 4 footfalls per stride
    while t < t_path[-1]:
        i = np.searchsorted(t_path, t)
        x, y = path_xy[min(i, len(path_xy) - 1)]
        n = int(contact * FS)
        tt = np.arange(n) / n
        fz = per_foot * 4 * np.exp(-((tt - 0.4) / 0.2) ** 2)
        imp = max(int(0.010 * FS), 4)
        k = np.arange(imp)
        fz[:imp] += per_foot * 2 * np.exp(-k / (imp / 3)) * np.sin(np.pi * k / imp)
        fx = 0.2 * fz * rng.choice([-1, 1])
        emissions.append((t, x, y, fz, fx))
        t += beat * rng.normal(1.0, 0.05)
    return emissions, dict(mass=mass, stride=stride, kind=kind)


def vehicle(path_xy, t_path, rng, kind="car"):
    """Moving load chunks: quasi-static axle weight + quarter-car roughness +
    engine harmonics, emitted from successive positions (retarded-time superposition
    -> bell envelope + Doppler emerge). Chunk = 0.5 s of force at one position."""
    mass = {"car": rng.uniform(1000, 2500), "truck": rng.uniform(3500, 20000),
            "motorbike": rng.uniform(200, 400)}[kind]
    rpm = rng.uniform(1500, 4000)
    # hann-windowed chunks at 50% overlap (COLA): the summed force is continuous —
    # un-windowed chunks have DC axle-weight edges that wrap in the FFT convolution.
    chunk = 0.5
    hop = chunk / 2
    n = int(chunk * FS)
    win = np.hanning(n)
    t_rel = np.arange(n) / FS
    f_hop = rng.uniform(10, 17)                          # quarter-car wheel-hop
    from scipy.signal import lfilter
    b, a_ = _biquad_bp(f_hop, 0.3 * f_hop)
    emissions = []
    t = t_path[0]
    while t < t_path[-1] - chunk:
        i = np.searchsorted(t_path, t)
        x, y = path_xy[min(i, len(path_xy) - 1)]
        static = mass * 9.81
        rough = np.convolve(rng.standard_normal(n), np.ones(8) / 8, "same")
        dyn = lfilter(b, a_, rough)
        dyn *= 0.10 * static / max(dyn.std(), 1e-9)
        fz = static + dyn
        if kind in ("car", "truck", "motorbike"):
            eng = np.zeros(n)
            amp = {"car": 0.01, "truck": 0.02, "motorbike": 0.05}[kind] * static
            for order in (1, 2, 3):                      # absolute time -> coherent phase
                eng += (amp / order) * np.sin(2 * np.pi * (rpm / 60) * order * (t_rel + t))
            fz = fz + eng
        fx = 0.05 * (fz - static)
        emissions.append((t, x, y, (fz * win).astype(np.float64), fx * win))
        t += hop
    return emissions, dict(mass=mass, kind=kind, rpm=rpm, f_hop=f_hop)


def _biquad_bp(f0, bw, fs=FS):
    from scipy.signal import iirpeak
    return iirpeak(f0 / (fs / 2), f0 / bw)
