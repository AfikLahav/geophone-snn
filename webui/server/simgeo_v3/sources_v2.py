"""Source emission generators for the slice — GENERATION_PLAN v1.2.
Each returns a list of emissions (t, x, y, Fz[bank.n], Fx[bank.n]) to convolve with the bank.
Human via wavelet.py; vehicle = moving load + roughness + engine; animal = quadruped
allometric (CONFOUND class — breadth over the mass/gait envelope, 4-footfall pattern)."""
import numpy as np
import wavelet
from scipy.signal import iirpeak, lfilter

FS = 1000.0


def _pad(F, n):
    out = np.zeros(n); out[:min(len(F), n)] = F[:n]; return out


def human(path_xy, t_grid, rng, bank_n, gait="walk"):
    """gait: walk / run / stealth / child. Stealth = reduced GRF + slow cadence (v1.2 0.5-0.7x)."""
    if gait == "child":
        mass = rng.uniform(18, 42); gait_w = "walk"; scale = 1.0; cad = rng.uniform(1.8, 2.2)
    elif gait == "stealth":
        mass = rng.uniform(55, 95); gait_w = "walk"; scale = rng.uniform(0.5, 0.7)
        cad = rng.uniform(1.0, 1.5)
    elif gait == "run":
        mass = rng.uniform(55, 95); gait_w = "run"; scale = 1.0; cad = rng.uniform(2.6, 3.3)
    else:
        mass = rng.uniform(50, 100); gait_w = "walk"; scale = 1.0; cad = rng.uniform(1.3, 2.2)
    ff = wavelet.walk_footfall if gait_w == "walk" else wavelet.run_footfall
    em = []; t = t_grid[0] + rng.uniform(0, 0.5)
    while t < t_grid[-1]:
        Fz, Fx = ff(rng, mass)
        em.append((t, *path_xy[min(np.searchsorted(t_grid, t), len(path_xy) - 1)],
                   _pad(Fz * scale, bank_n), _pad(Fx * scale, bank_n)))
        t += 1.0 / cad * rng.normal(1, 0.03)
    return em, dict(kind="human", subkind=gait, mass=round(mass, 1), cadence=round(cad, 2))


def vehicle(path_xy, t_grid, rng, bank_n, kind="car"):
    # bicycle broadened to the light-wheeled family (bike/e-bike/e-scooter — seismically alike);
    # tractor = heavy/slow with a low-RPM diesel tonal (10-25 Hz fundamental).
    # tracked = pad-impact comb f=V/pitch + road-wheel line V/axle-spacing (Krylov 2011,
    #   EuroDyn: Leopard-1 f_tr=V/a=23.1 Hz @ a=0.169 m, f_wb=V/E=5.9 Hz @ E=0.665 m).
    mass = {"car": rng.uniform(900, 2500), "truck": rng.uniform(3500, 20000),
            "motorbike": rng.uniform(200, 400), "bicycle": rng.uniform(75, 130),
            "tractor": rng.uniform(3000, 8000),
            "tracked": rng.uniform(10000, 60000)}[kind]
    has_engine = kind in ("car", "truck", "motorbike", "tractor", "tracked")
    eng_amp = {"car": 0.01, "truck": 0.02, "motorbike": 0.05, "bicycle": 0.0,
               "tractor": 0.04, "tracked": 0.03}[kind]
    rough_mul = {"car": 1.0, "truck": 1.0, "motorbike": 0.8, "bicycle": 1.5,
                 "tractor": 1.2, "tracked": 1.3}[kind]
    # 4-stroke diesel/petrol; tracked = multi-cyl diesel idling/low-rev
    n_cyl = {"car": 4, "truck": 6, "motorbike": 2, "bicycle": 0,
             "tractor": 4, "tracked": 12}[kind]
    rpm = (rng.uniform(600, 1500) if kind in ("tractor", "tracked")
           else rng.uniform(1500, 3500))
    f_hop = rng.uniform(10, 15)                      # unsprung-mass / axle-hop line, peak ~12 Hz
    # tracked pad-impact geometry (per emission speed -> f = V/pitch); pitch 0.12-0.19 m
    track_pitch = rng.uniform(0.12, 0.19)
    axle_spacing = rng.uniform(0.55, 0.75)          # road-wheel base line V/E
    chunk = 0.5; n = int(chunk * FS); win = np.hanning(n); t_rel = np.arange(n) / FS
    b, a = iirpeak(f_hop / (FS / 2), f_hop / (0.3 * f_hop))
    em = []; t = float(t_grid[0])
    while t < t_grid[-1] - chunk:
        i = np.searchsorted(t_grid, t); x, y = path_xy[min(i, len(path_xy) - 1)]
        static = mass * 9.81
        dyn = lfilter(b, a, np.convolve(rng.standard_normal(n), np.ones(8) / 8, "same"))
        dyn *= 0.10 * rough_mul * static / max(dyn.std(), 1e-9)
        eng = np.zeros(n)
        if has_engine:
            # 4-stroke engine-firing comb: f_fire = RPM*N_cyl/120 (= RPM/2 * N_cyl per s).
            f_fire = rpm * n_cyl / 120.0
            for order in (1, 2, 3):
                eng += (eng_amp * static / order) * np.sin(
                    2 * np.pi * f_fire * order * (t_rel + t))
        track = np.zeros(n)
        if kind == "tracked":
            # local speed from path displacement over this chunk (fallback to nominal)
            j = min(i + 1, len(path_xy) - 1)
            dx = path_xy[j][0] - x; dy = path_xy[j][1] - y
            dt_step = max(float(t_grid[min(j, len(t_grid) - 1)] - t_grid[i]), 1e-3) \
                if i < len(t_grid) else 0.1
            v_loc = float(np.hypot(dx, dy) / max(dt_step, 1e-3))
            v_loc = float(np.clip(v_loc, 1.5, 18.0))     # field tracked speed band
            f_tr = v_loc / track_pitch                   # pad-impact fundamental V/pitch
            f_wb = v_loc / axle_spacing                  # road-wheel line V/axle-spacing
            # pad comb = MULTIPLES of f_tr (Krylov: harmonics of V/a, not fixed 4/16/32 Hz)
            for h in range(1, 7):
                fk = f_tr * h
                if fk >= FS / 2:
                    break
                track += (0.06 * static / h) * np.sin(
                    2 * np.pi * fk * (t_rel + t) + rng.uniform(0, 6.28))
            # road-wheel base line + its 2nd harmonic
            for h in (1, 2):
                fk = f_wb * h
                if fk < FS / 2:
                    track += (0.04 * static / h) * np.sin(
                        2 * np.pi * fk * (t_rel + t) + rng.uniform(0, 6.28))
        Fz = (static + dyn + eng + track) * win
        Fx = 0.05 * (Fz - static * win)
        em.append((t, x, y, _pad(Fz, bank_n), _pad(Fx, bank_n)))
        t += chunk / 2
    info = dict(kind="vehicle", subkind=kind, mass=round(mass),
                rpm=round(rpm) if has_engine else 0, f_hop=round(f_hop, 1))
    if kind == "tracked":
        info.update(track_pitch=round(track_pitch, 3), axle_spacing=round(axle_spacing, 3))
    return em, info


# Foot type per species — drives the impact-shock wavelet (B2). HOOF = a keratinous
# rigid hoof striking the ground: high-amplitude, HF-rich short shock (galloping-horse
# hoof impact 80-100 g, HF energy >400 Hz; PMC9454475). PAW = a soft, viscoelastically
# damped pad: HF-poor, longer/softer onset (musculoskeletal/pad attenuation). NOTE:
# physically motivated source realism only — the distinctive hoof HF content is largely
# stripped by soil propagation at standoff, so this is NOT claimed as a standoff
# discriminator (V2 §B2; near-surface HF attenuation lengths "a few feet").
_FOOT = {"dog": "paw", "jackal": "paw",
         "horse": "hoof", "boar": "hoof", "sheep": "hoof"}


def animal(path_xy, t_grid, rng, bank_n, kind="dog"):
    """Quadruped, allometric (M^-0.148), 4-footfall gait, fore/hind asymmetry.
    Confound class: breadth over mass envelope matters, not per-species precision.
    Hoof (horse/boar/sheep) vs paw (dog/jackal) impact wavelets differ in HF content."""
    mass = {"dog": rng.uniform(10, 30), "boar": rng.uniform(50, 120),
            "jackal": rng.uniform(6, 15), "horse": rng.uniform(400, 700),
            "sheep": rng.uniform(45, 90)}[kind]
    foot = _FOOT[kind]
    stride = 1.8 * (mass / 30) ** (-0.148) * rng.uniform(0.85, 1.15)   # Hz
    contact = float(np.clip(0.18 * (mass / 30) ** 0.148, 0.05, 0.5))
    per_foot = mass * 9.81 * rng.uniform(1.0, 1.3)
    gait = rng.choice(["walk", "trot", "gallop"])
    em = []; t = t_grid[0] + rng.uniform(0, 0.5)
    # 4-footfall sequence per stride (the biped-vs-quadruped discriminator)
    if gait == "walk":
        offs = [0, 0.25, 0.5, 0.75]
    elif gait == "trot":
        offs = [0, 0.0, 0.5, 0.5]            # diagonal pairs
    else:
        offs = [0, 0.1, 0.55, 0.6]           # gallop asymmetric
    fore = [True, False, True, False]        # forelimbs ~30-40% higher
    n = int(contact * FS); tt = np.arange(n) / n
    # hoof = sharp HF-rich short shock (~6 ms rise, damped HF ring-down);
    # paw = soft/damped impact (~18 ms rise, no HF ring) — see _FOOT note.
    if foot == "hoof":
        imp_ms, shock_amp, ring_f, ring_zeta = 0.006, 0.9, rng.uniform(180, 320), 0.18
    else:
        imp_ms, shock_amp, ring_f, ring_zeta = 0.018, 0.35, rng.uniform(40, 80), 0.45
    while t < t_grid[-1]:
        i = np.searchsorted(t_grid, t); x, y = path_xy[min(i, len(path_xy) - 1)]
        for off, isf in zip(offs, fore):
            amp = per_foot * (1.0 if isf else 0.7)
            shape = np.exp(-((tt - 0.4) / 0.22) ** 2)            # slow GRF lobe (unchanged)
            Fz = amp * shape
            # impact-shock transient at heel/hoof strike (foot-type dependent)
            imp = max(int(imp_ms * FS), 4); k = np.arange(imp)
            rise = (k / (imp / 3.0)) * np.exp(1 - k / (imp / 3.0))
            Fz[:imp] += amp * shock_amp * rise
            # damped sinusoid ring-down riding on the impact (hoof: HF; paw: LF, heavy damp)
            kr = np.arange(n)
            ring = np.exp(-ring_zeta * 2 * np.pi * ring_f * kr / FS) * \
                np.sin(2 * np.pi * ring_f * kr / FS)
            Fz += amp * shock_amp * 0.5 * ring
            Fx = 0.2 * Fz * rng.choice([-1, 1])
            em.append((t + off / stride, x, y, _pad(Fz, bank_n), _pad(Fx, bank_n)))
        t += 1.0 / stride * rng.normal(1, 0.05)
    return em, dict(kind="animal", subkind=kind, mass=round(mass, 1),
                    stride=round(float(stride), 2), gait=gait, foot=foot)


# ============ v2 stationary machinery (ground-borne; return ground velocity m/s) ============
def pump(n, rng, fs=FS, level=None):
    """Irrigation pump (stationary). BPF=(RPM/60)*N_vanes ~= 75-175 Hz + harmonics (per v1.2
    spec, NOT fixed 75), duty-cycled. Returns (ground_vel m/s, info)."""
    rpm = rng.uniform(1000, 1500); vanes = int(rng.integers(4, 8)); f0 = rpm / 60.0 * vanes
    t = np.arange(n) / fs; x = np.zeros(n)
    for k in (1, 2, 3):
        if f0 * k < fs / 2:
            x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
    env = np.ones(n); ramp = int(min(2 * fs, n // 10))
    if ramp > 0:
        env[:ramp] = np.linspace(0, 1, ramp); env[-ramp:] = np.linspace(1, 0, ramp)
    if n / fs > 120 and rng.random() < 0.5:                # long scene: an OFF stretch
        a = int(rng.uniform(0.3, 0.6) * n); b = min(a + int(rng.uniform(10, 40) * fs), n); env[a:b] = 0.0
    x *= env
    lvl = level if level is not None else 10 ** rng.uniform(-8.6, -7.6)
    return (x / (x.std() + 1e-30) * lvl).astype(np.float64), dict(kind="pump", rpm=round(rpm), vanes=vanes, bpf=round(f0, 1))


def generator_set(n, rng, fs=FS, level=None):
    """Stationary genset: engine-firing tonal + harmonics + mild broadband. Ground vel m/s."""
    rpm = rng.uniform(1500, 3000); f0 = rpm / 60.0
    t = np.arange(n) / fs; x = np.zeros(n)
    for k in (1, 2, 3, 4):
        if f0 * k < fs / 2:
            x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
    x += 0.15 * rng.standard_normal(n)
    lvl = level if level is not None else 10 ** rng.uniform(-8.6, -7.6)
    return (x / (x.std() + 1e-30) * lvl).astype(np.float64), dict(kind="generator", rpm=round(rpm), f0=round(f0, 1))


# ============ v2 aircraft overflight (air-to-ground coupled; ground velocity m/s) ============
AIRCRAFT = {
    "apache":     dict(bpf=19.3, tail=93.0, nh=10, kind="heli"),
    "blackhawk":  dict(bpf=17.2, tail=81.0, nh=10, kind="heli"),
    "ch53":       dict(bpf=20.0, tail=50.0, nh=12, kind="heli"),
    "light_heli": dict(bpf=13.0, tail=78.0, nh=8,  kind="heli"),
    "cessna":     dict(bpf=80.0, tail=40.0, nh=5,  kind="prop"),
    "fighter":    dict(bpf=0.0,  tail=0.0,  nh=0,  kind="jet"),     # subsonic, NO sonic boom
    "commercial": dict(bpf=0.0,  tail=0.0,  nh=0,  kind="jet_hi"),  # ~absent (<1 Hz)
}
C_COUPLE = 2.4e-6           # m/s per Pa (Novoselov 2020); angle-modulated below


def overflight(n, rng, kind=None, fs=FS):
    """Aircraft flyover -> ground velocity (m/s). Comb (heli/prop) or broadband (jet) x Doppler
    glide x incidence-angle coupling (peaks overhead) x 1/R x air absorption."""
    kind = kind or str(rng.choice(list(AIRCRAFT)))
    a = AIRCRAFT[kind]; t = np.arange(n) / fs; dur = n / fs
    v = rng.uniform(40, 120) if a["kind"] != "jet_hi" else rng.uniform(200, 260)
    h = rng.uniform(80, 600) if a["kind"] == "heli" else rng.uniform(150, 4000)
    if a["kind"] == "jet_hi":
        h = rng.uniform(8000, 11000)
    t0 = rng.uniform(0.35, 0.65) * dur
    R = np.sqrt(h ** 2 + (v * (t - t0)) ** 2); cth = h / np.maximum(R, 1.0); c_air = 340.0
    dopp = c_air / (c_air - v * (-(v * (t - t0)) / np.maximum(R, 1.0)))
    g_ang = np.clip(cth, 0.0, 1.0) ** 1.5
    amp_env = C_COUPLE * (0.05 + 0.95 * g_ang) * (h / np.maximum(R, 1.0))
    x = np.zeros(n)
    if a["kind"] in ("heli", "prop"):
        for comb, base in (("main", a["bpf"]), ("tail", a["tail"])):
            if base <= 0:
                continue
            for k in range(1, a["nh"] + 1):
                fk = base * k * dopp
                if np.median(fk) > fs / 2:
                    break
                hk = (1.0 / k) * (0.6 if comb == "tail" else 1.0)
                absb = np.exp(-(base * k) / 250.0 * R / np.maximum(R.min(), 1.0))
                x += hk * absb * np.sin(2 * np.pi * np.cumsum(fk) / fs)
    else:
        bb = rng.standard_normal(n); fb = np.fft.rfftfreq(n, 1 / fs)
        hf = 1.0 / (1 + (fb / (60 if a["kind"] == "jet_hi" else 250)) ** 2)
        x = np.fft.irfft(np.fft.rfft(bb) * hf, n=n)
    x = x * amp_env
    lvl = 10 ** rng.uniform(-7.5, -6.0) if a["kind"] != "jet_hi" else 10 ** rng.uniform(-9.5, -8.5)
    x = x / (np.abs(x).max() + 1e-30) * lvl
    return x.astype(np.float64), dict(kind="overflight", subkind=kind, speed=round(v), alt=round(h),
                                      bpf=a["bpf"], aircraft_class=a["kind"])
