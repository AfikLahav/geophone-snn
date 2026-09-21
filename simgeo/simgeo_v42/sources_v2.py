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


def human(path_xy, t_grid, rng, bank_n, gait="walk", payload_kg=0.0, cadence_mul=1.0,
          v_profile=None):
    """v4.3: speed-derived cadence from v_profile (paths.py). When v_profile is provided,
    cadence tracks local speed, pauses produce zero emission, and start/stop ramps modulate
    force. When v_profile=None, falls back to the v4.2 constant-cadence behavior.
    gait: walk / run / stealth / child. payload_kg adds carried load."""
    if gait == "child":
        mass = rng.uniform(18, 42); gait_w = "walk"; scale = 1.0; cad = rng.uniform(1.8, 2.2)
    elif gait == "stealth":
        mass = rng.uniform(55, 95); gait_w = "walk"; scale = rng.uniform(0.5, 0.7)
        cad = rng.uniform(1.0, 1.5)
    elif gait == "run":
        mass = rng.uniform(55, 95); gait_w = "run"; scale = 1.0; cad = rng.uniform(2.6, 3.3)
    else:
        mass = rng.uniform(50, 100); gait_w = "walk"; scale = 1.0; cad = rng.uniform(1.3, 2.2)
    mass_eff = mass + max(payload_kg, 0.0)
    cad_base = cad * cadence_mul
    ff = wavelet.walk_footfall if gait_w == "walk" else wavelet.run_footfall

    # nominal step length for speed->cadence coupling
    speed_base = None
    if v_profile is not None and len(v_profile) > 0:
        speed_base = float(np.median(v_profile[v_profile > 0.15])) if np.any(v_profile > 0.15) else 1.0
        L0 = max(speed_base / cad_base, 0.3)

    em = []; t = t_grid[0] + rng.uniform(0, 0.5)
    step_in_bout = 0  # for start ramp
    was_paused = True  # start of scene counts as a "resume"

    while t < t_grid[-1]:
        i_t = min(int(np.searchsorted(t_grid, t)), len(t_grid) - 1)

        if v_profile is not None:
            v_local = float(v_profile[min(i_t, len(v_profile) - 1)])
            if v_local < 0.15:
                # standing still — skip emission, advance time
                t += 0.1
                was_paused = True
                step_in_bout = 0
                continue
            if was_paused:
                step_in_bout = 0
                was_paused = False
            cad_local = float(np.clip(v_local / L0, 0.5, 5.0))
        else:
            cad_local = cad_base

        # start ramp: first 3 steps after pause/start at reduced force
        start_scale = 1.0
        if v_profile is not None and step_in_bout < 3:
            start_scale = [0.6, 0.8, 0.95][step_in_bout]

        Fz, Fx = ff(rng, mass_eff)
        em.append((t, *path_xy[min(i_t, len(path_xy) - 1)],
                   _pad(Fz * scale * start_scale, bank_n),
                   _pad(Fx * scale * start_scale, bank_n)))
        step_in_bout += 1
        t += 1.0 / cad_local * rng.normal(1, 0.03)

    return em, dict(kind="human", subkind=gait, mass=round(mass, 1),
                    cadence=round(float(cad_base), 2),
                    payload_kg=round(float(payload_kg), 1),
                    has_speed_profile=v_profile is not None)


def human_group_coherent(path_xy, t_grid, rng, bank_n, n_members, phase_coherence,
                         cadence_hz, payload_kg=0.0):
    """v4: N walkers with a controllable phase/cadence coherence axis.
    phase_coherence in [0,1]: 0 = independent civilians (cadence spread sigma=0.12 Hz, random
    phase -> incoherent ~sqrt(N) sum, broad ~2 Hz peak); 1 = locked march (single cadence,
    +-15 ms step lock -> coherent ~N sum, sharp harmonics). cadence_hz ~ 2.0 (quick march) or
    3.0 (double time). Pedestrian-force-model + military-cadence literature (scene-complexity
    report B.1). Members drift a few metres apart, independent mass/footwear."""
    c = float(np.clip(phase_coherence, 0.0, 1.0))
    em = []
    for _ in range(int(n_members)):
        mass = rng.uniform(50, 100) + max(payload_kg, 0.0)
        cad_i = cadence_hz + (1.0 - c) * rng.normal(0.0, 0.12)     # inter-subject cadence spread
        cad_i = float(max(cad_i, 0.8))
        phase0 = (1.0 - c) * rng.uniform(0.0, 1.0 / cad_i)         # start-phase offset
        off = rng.uniform(-1.0, 1.0, 2) * 2.0                       # +-2 m member position
        t = t_grid[0] + rng.uniform(0, 0.3) + phase0
        while t < t_grid[-1]:
            Fz, Fx = wavelet.walk_footfall(rng, mass)
            p = path_xy[min(np.searchsorted(t_grid, t), len(path_xy) - 1)] + off
            em.append((t, p[0], p[1], _pad(Fz, bank_n), _pad(Fx, bank_n)))
            jitter = rng.normal(0.0, 0.015) * c + rng.normal(0.0, 0.03 / cad_i) * (1.0 - c)
            t += 1.0 / cad_i + jitter
    subk = "march" if c >= 0.6 else "group"
    return em, dict(kind="human", subkind=subk, n=int(n_members), coherence=round(c, 2),
                    cadence=round(float(cadence_hz), 2), payload_kg=round(float(payload_kg), 1))


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

# v4.3: species-specific bout/pause parameters (08_ANIMAL_BEHAVIOR_RESEARCH.md)
_STOP_DWELL = {"dog": (1, 5), "jackal": (1, 5), "horse": (2, 10),
               "sheep": (3, 15), "boar": (2, 8)}
_MOVE_DWELL = {"dog": (5, 25), "jackal": (5, 25), "horse": (8, 35),
               "sheep": (5, 20), "boar": (6, 30)}
_GAIT_MIX = {"dog": {"walk": .60, "trot": .35, "gallop": .05},
             "jackal": {"walk": .60, "trot": .35, "gallop": .05},
             "horse": {"walk": .70, "trot": .25, "gallop": .05},
             "sheep": {"walk": .85, "trot": .13, "gallop": .02},
             "boar": {"walk": .75, "trot": .20, "gallop": .05}}
_GAIT_STRIDE_MUL = {"walk": 1.0, "trot": 1.35, "gallop": 1.70}
_GAIT_OFFSETS = {"walk": [0, 0.25, 0.5, 0.75],
                 "trot": [0, 0.0, 0.5, 0.5],
                 "gallop": [0, 0.1, 0.55, 0.6]}


def _activity_schedule(rng, dur, kind, force_gait=None):
    """Semi-Markov MOVE/STOP schedule with species-specific dwell times and gait mix."""
    pause_lo, pause_hi = _STOP_DWELL.get(kind, (2, 8))
    move_lo, move_hi = _MOVE_DWELL.get(kind, (5, 25))
    gw = _GAIT_MIX.get(kind, {"walk": .7, "trot": .25, "gallop": .05})
    gaits = list(gw.keys()); probs = [gw[g] for g in gaits]
    segs = []; t = 0.0
    state = "MOVE" if rng.random() < 0.85 else "STOP"
    if state == "STOP":
        t1 = min(rng.uniform(pause_lo, pause_hi), dur)
        segs.append((0.0, t1, "STOP", None)); t = t1; state = "MOVE"
    while t < dur:
        if state == "MOVE":
            g = force_gait or gaits[int(rng.choice(len(gaits), p=probs))]
            dwell = {"walk": (move_lo, move_hi),
                     "trot": (move_lo * 0.5, move_hi * 0.5),
                     "gallop": (1, 6)}[g]
            t1 = min(t + rng.uniform(*dwell), dur)
            segs.append((t, t1, "MOVE", g)); t = t1; state = "STOP"
        else:
            t1 = min(t + rng.uniform(pause_lo, pause_hi), dur)
            segs.append((t, t1, "STOP", None)); t = t1; state = "MOVE"
    return segs


def animal(path_xy, t_grid, rng, bank_n, kind="dog", force_gait=None, stride_mul=1.0,
           rider_kg=0.0, schedule=None):
    """v4.3: quadruped with semi-Markov bout schedule — species-specific pauses, gait
    transitions coupled to stride frequency, AR(1) correlated jitter, L/R asymmetry.
    Replaces the v4.2 metronomic continuous gait. Backward compatible: force_gait and
    stride_mul work as before. Optional schedule kwarg for herd group sync."""
    mass = {"dog": rng.uniform(10, 30), "boar": rng.uniform(50, 120),
            "jackal": rng.uniform(6, 15), "horse": rng.uniform(400, 700),
            "sheep": rng.uniform(45, 90)}[kind]
    foot = _FOOT[kind]
    stride0 = 1.8 * (mass / 30) ** (-0.148)
    contact = float(np.clip(0.18 * (mass / 30) ** 0.148, 0.05, 0.5))
    mass_eff = mass + max(rider_kg, 0.0)
    per_foot = mass_eff * 9.81 * rng.uniform(1.0, 1.3)
    fore = [True, False, True, False]
    n = int(contact * FS); tt = np.arange(n) / n
    if foot == "hoof":
        imp_ms, shock_amp, ring_f, ring_zeta = 0.006, 0.9, rng.uniform(180, 320), 0.18
    else:
        imp_ms, shock_amp, ring_f, ring_zeta = 0.018, 0.35, rng.uniform(40, 80), 0.45

    # per-individual traits (drawn once)
    lr_bias = rng.uniform(0, 0.08)
    ar_state = 0.0
    ar_rho = rng.uniform(0.7, 0.85)

    dur = float(t_grid[-1] - t_grid[0])
    sched = schedule if schedule is not None else _activity_schedule(rng, dur, kind, force_gait)
    em = []; active_gait = "walk"
    for seg_t0, seg_t1, state, gait in sched:
        if state == "STOP":
            continue
        active_gait = gait or "walk"
        offs = _GAIT_OFFSETS[active_gait]
        stride = stride0 * _GAIT_STRIDE_MUL[active_gait] * float(stride_mul)
        stride = max(stride, 0.85 * stride0 * 0.85)
        t = t_grid[0] + seg_t0 + rng.uniform(0, 0.3)
        abs_end = t_grid[0] + seg_t1
        while t < abs_end and t < t_grid[-1]:
            i = np.searchsorted(t_grid, t)
            x, y = path_xy[min(i, len(path_xy) - 1)]
            # AR(1) stride jitter
            ar_state = ar_rho * ar_state + np.sqrt(1 - ar_rho**2) * rng.normal(0, 0.10)
            jitter = float(np.clip(np.exp(ar_state), 0.7, 1.4))
            # per-stride force noise
            force_noise = rng.normal(1, 0.05)
            for fi, (off, isf) in enumerate(zip(offs, fore)):
                lr = (1.0 + lr_bias / 2) if isf else (0.7 - lr_bias / 2)
                amp = per_foot * lr * force_noise
                shape = np.exp(-((tt - 0.4) / 0.22) ** 2)
                Fz = amp * shape
                imp = max(int(imp_ms * FS), 4); k = np.arange(imp)
                rise = (k / (imp / 3.0)) * np.exp(1 - k / (imp / 3.0))
                Fz[:imp] += amp * shock_amp * rise
                kr = np.arange(n)
                ring = np.exp(-ring_zeta * 2 * np.pi * ring_f * kr / FS) * \
                    np.sin(2 * np.pi * ring_f * kr / FS)
                Fz += amp * shock_amp * 0.5 * ring
                Fx = 0.2 * Fz * rng.choice([-1, 1])
                em.append((t + off / stride, x, y, _pad(Fz, bank_n), _pad(Fx, bank_n)))
            t += (1.0 / stride) * jitter
    return em, dict(kind="animal", subkind=kind, mass=round(mass, 1),
                    stride=round(float(stride0), 2), gait=active_gait, foot=foot,
                    n_bouts=sum(1 for s in sched if s[2] == "MOVE"),
                    n_pauses=sum(1 for s in sched if s[2] == "STOP"))


# ============ v4 composite-scene primitives (emissions at a FIXED position) ============
def vehicle_idle(xy, dur_s, rng, bank_n, kind="car"):
    """Stationary idling vehicle: engine-firing comb + harmonics + mild idle shake at a FIXED
    position. NO static (moving-load) term -- a stationary dead load does not radiate far-field
    seismic energy (same A1 principle as wavelet.py). Reuses vehicle()'s engine model."""
    mass = {"car": rng.uniform(900, 2500), "truck": rng.uniform(3500, 20000),
            "tractor": rng.uniform(3000, 8000)}.get(kind, rng.uniform(900, 2500))
    n_cyl = {"car": 4, "truck": 6, "tractor": 4}.get(kind, 4)
    rpm = rng.uniform(700, 1100) if kind != "tractor" else rng.uniform(600, 900)   # idle RPM
    eng_amp = {"car": 0.01, "truck": 0.02, "tractor": 0.04}.get(kind, 0.01)
    x, y = float(xy[0]), float(xy[1])
    chunk = 0.5; n = int(chunk * FS); win = np.hanning(n); t_rel = np.arange(n) / FS
    static = mass * 9.81; f_fire = rpm * n_cyl / 120.0
    em = []; t = 0.0
    while t < dur_s - chunk:
        eng = np.zeros(n)
        for order in (1, 2, 3):
            eng += (eng_amp * static / order) * np.sin(2 * np.pi * f_fire * order * (t_rel + t))
        shake = 0.02 * static * np.convolve(rng.standard_normal(n), np.ones(8) / 8, "same")
        Fz = (eng + shake) * win
        em.append((t, x, y, _pad(Fz, bank_n), _pad(0.02 * Fz, bank_n)))
        t += chunk / 2
    return em, dict(kind="vehicle", subkind=f"{kind}_idle", rpm=round(rpm), f_fire=round(f_fire, 1))


def door_slam(xy, t0, rng, bank_n):
    """Single impulsive door slam at a FIXED position. Heel-strike-shaped impulse (wavelet A2'):
    A*(k/tau)*exp(1-k/tau). NOTE: amplitude is an engineering ESTIMATE (no calibrated slam GRF) --
    flagged synthetic-only in scene meta."""
    A = float(rng.uniform(500.0, 1500.0)); tau_ms = float(rng.uniform(5.0, 15.0))
    tau = tau_ms * FS / 1000.0; L = int(max(tau * 6, 20))
    k = np.arange(L)
    Fz = A * (k / tau) * np.exp(1 - k / tau)
    x, y = float(xy[0]), float(xy[1])
    return [(float(t0), x, y, _pad(Fz, bank_n), _pad(0.1 * Fz, bank_n))], \
        dict(kind="vehicle", subkind="door_slam", amp_n=round(A), tau_ms=round(tau_ms, 1))


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
    "drone":      dict(bpf=140.0, tail=0.0, nh=4,  kind="drone"),   # v4: small-UAV blade-pass comb
}
C_COUPLE = 2.4e-6           # m/s per Pa (Novoselov 2020); angle-modulated below


def overflight(n, rng, kind=None, fs=FS):
    """Aircraft flyover -> ground velocity (m/s). Comb (heli/prop) or broadband (jet) x Doppler
    glide x incidence-angle coupling (peaks overhead) x 1/R x air absorption."""
    kind = kind or str(rng.choice(list(AIRCRAFT)))
    a = dict(AIRCRAFT[kind]); t = np.arange(n) / fs; dur = n / fs
    if a["kind"] == "drone":
        a["bpf"] = float(rng.uniform(110.0, 190.0))   # small-UAV blade-pass varies per craft
    v = rng.uniform(40, 120) if a["kind"] != "jet_hi" else rng.uniform(200, 260)
    h = rng.uniform(80, 600) if a["kind"] == "heli" else rng.uniform(150, 4000)
    if a["kind"] == "jet_hi":
        h = rng.uniform(8000, 11000)
    elif a["kind"] == "drone":
        v = rng.uniform(5.0, 20.0); h = rng.uniform(30.0, 150.0)   # low + slow small UAV
    t0 = rng.uniform(0.35, 0.65) * dur
    R = np.sqrt(h ** 2 + (v * (t - t0)) ** 2); cth = h / np.maximum(R, 1.0); c_air = 340.0
    dopp = c_air / (c_air - v * (-(v * (t - t0)) / np.maximum(R, 1.0)))
    g_ang = np.clip(cth, 0.0, 1.0) ** 1.5
    amp_env = C_COUPLE * (0.05 + 0.95 * g_ang) * (h / np.maximum(R, 1.0))
    x = np.zeros(n)
    if a["kind"] in ("heli", "prop", "drone"):
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
    if a["kind"] == "jet_hi":
        lvl = 10 ** rng.uniform(-9.5, -8.5)
    elif a["kind"] == "drone":
        lvl = 10 ** rng.uniform(-8.8, -7.5)                  # small UAV: much weaker than a heli
    else:
        lvl = 10 ** rng.uniform(-7.5, -6.0)
    x = x / (np.abs(x).max() + 1e-30) * lvl
    return x.astype(np.float64), dict(kind="overflight", subkind=kind, speed=round(v), alt=round(h),
                                      bpf=a["bpf"], aircraft_class=a["kind"])
