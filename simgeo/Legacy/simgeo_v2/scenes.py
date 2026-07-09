"""Scene assembly (PLAN Phase 4): path -> r(t) -> per-emission GF convolution
((iw)^2 chain + causal t*) -> superposition -> sensor render -> windows+labels.
"""
import os, json
import numpy as np

FS = 1000.0
HERE = os.path.dirname(os.path.abspath(__file__))
BANKS = os.path.join(HERE, "..", "..", "..", "terrain_models")


def settle_clamp(d, radii_km, dt_bank, vr_ms, margin=1.5):
    """Clamp each radius's late tail to its median: after the slowest coda has
    passed, the step response is BY PHYSICS the static. Kills (a) pyprop8's
    alpha-amplified late-time noise and (b) the undamped ringing of trapped modes
    in stiff-over-soft (inverse) profiles — a purely elastic solver rings forever
    where real material damping (Q) kills the ring in ~Q/(pi*f) seconds.
    margin: coda allowance after the slow-wave arrival (stiff ground rings
    briefly -> callers pass 1.0; soft/inverse keep 1.5)."""
    vr_s = max(vr_ms, 60.0) / 1000.0
    nb = d.shape[1]
    t_ax = np.arange(nb) * dt_bank
    for i, r_km in enumerate(np.asarray(radii_km, float)):
        t_settle = r_km / (0.5 * vr_s) + margin
        j = np.searchsorted(t_ax, t_settle)
        if nb - j > int(0.3 / dt_bank):
            d[i, j:] = np.median(d[i, j:])
    return d


def clamp_margin(meta):
    return 1.0 if (meta["vs_top_ms"] >= 800 and not meta.get("inverse")) else 1.5


class Bank:
    """Cached GF bank for one profile. Precomputes the per-radius impulse-VELOCITY
    spectra: V_imp(w) = (iw)^2 * D_step(w) (fable's chain), with causal t* applied."""

    _cache = {}

    def __init__(self, profile_id):
        z = np.load(os.path.join(BANKS, f"{profile_id}.npz"))
        self.meta = json.loads(str(z["meta"]))
        self.radii_m = z["radii_km"] * 1000.0
        dt_bank = float(z["dt"])
        nb = z["z_fz"].shape[1]
        # banks are computed at 250 Hz Nyquist (dt=2ms); upsample to the 1000 Hz
        # render grid by zero-padding the band-limited spectrum (exact).
        up = int(round(dt_bank / 1e-3))
        self.dt = 1e-3
        self.n = nb * up
        w = 2 * np.pi * np.fft.rfftfreq(self.n, self.dt)
        iw2 = (1j * w) ** 2
        f = np.fft.rfftfreq(self.n, self.dt)
        q, vr = self.meta["q"], max(self.meta["vr_ms"], 60.0)
        nbins_b = nb // 2 + 1
        self.VF = {}
        for key in ("z_fz", "z_fx"):
            d = z[key].astype(np.float64)               # km per 1e15 N (step)
            d = settle_clamp(d, z["radii_km"], dt_bank, self.meta["vr_ms"],
                             margin=clamp_margin(self.meta))
            # step -> impulse via TIME differencing (decays to zero: wrap-safe DFT;
            # rfft of the raw step would alias its static offset into leakage)
            h = np.diff(d, axis=1, append=d[:, -1:]) / dt_bank
            D = np.fft.rfft(h, axis=1) * up             # DFT rescale to denser grid
            # cosine-taper the top 10% of the band before zero-padding — a
            # brick-wall cutoff rings (sinc tails smear faint pre-arrival energy)
            fb = np.fft.rfftfreq(nb, dt_bank)
            fny = fb[-1]
            tap = np.ones_like(fb)
            m = fb > 0.9 * fny
            tap[m] = 0.5 * (1 + np.cos(np.pi * (fb[m] - 0.9 * fny) / (0.1 * fny)))
            D *= tap[None, :]
            Dup = np.zeros((d.shape[0], self.n // 2 + 1), complex)
            Dup[:, :nbins_b] = D
            iw = 1j * w                                  # remaining disp -> vel derivative
            spec = iw[None, :] * Dup * (1e3 / 1e15)      # m/s per N (impulse-velocity)
            # causal t* (Azimi): amplitude exp(-pi f t*), KK dispersion phase
            tstar = (self.radii_m / (q * vr))[:, None]
            fr = np.maximum(f[None, :], 1e-3)
            att = np.exp(-np.pi * fr * tstar) * np.exp(2j * fr * tstar *
                                                       np.log(fr / 25.0))
            # pavement plate model (research-validated 2026-06-12): the stiff plate
            # is NOT in the wave stack (its guided modes are killed by material
            # damping within ~3 m in reality, and defeat wavenumber integration in
            # a purely elastic solver). It acts as an SDOF resonance filter shaping
            # the source force entering the soil: H = f0^2/((f0^2-f^2)+2i*zeta*f0*f).
            if self.meta.get("plate"):
                pf = self.meta["plate"]
                f0p, zp = pf["f0"], pf["zeta"]
                Hp = f0p**2 / ((f0p**2 - f**2) + 2j * zp * f0p * f)
                att = att * Hp[None, :]
            self.VF[key] = spec * att
        z.close()

    @classmethod
    def get(cls, profile_id):
        if profile_id not in cls._cache:
            if len(cls._cache) > 12:
                cls._cache.clear()
            cls._cache[profile_id] = cls(profile_id)
        return cls._cache[profile_id]

    def emit(self, r_m, heading_xy, src_xy, fz, fx):
        """Ground velocity wavelet (m/s, length self.n) for one emission at range r."""
        ri = int(np.clip(np.searchsorted(self.radii_m, r_m), 1, len(self.radii_m) - 1))
        ri = ri if (self.radii_m[ri] - r_m) < (r_m - self.radii_m[ri - 1]) else ri - 1
        scale = np.sqrt(self.radii_m[ri] / max(r_m, 0.5))   # r^-0.5 between grid radii
        Fz = np.fft.rfft(fz, self.n)
        out = self.VF["z_fz"][ri] * Fz
        if fx is not None and np.any(fx):
            # azimuth factor: tangential force along heading, receiver at -src direction
            to_rx = -np.asarray(src_xy)
            nr = np.linalg.norm(to_rx)
            if nr > 1e-6:
                cosaz = float(np.dot(heading_xy, to_rx) / (np.linalg.norm(heading_xy)
                                                           * nr + 1e-12))
                out = out + cosaz * self.VF["z_fx"][ri] * np.fft.rfft(fx, self.n)
        return np.fft.irfft(out, n=self.n) * self.dt * scale


def modal_floor_ir(f0, zeta, nmodes, n, rng):
    """Bounded-floor impulse response (spec §1 modal branch)."""
    t = np.arange(n) / FS
    h = np.zeros(n)
    for k in range(nmodes):
        fk = f0 * (1 + 0.9 * k + rng.uniform(-0.1, 0.1))
        if fk > 480:
            break
        zk = zeta * rng.uniform(0.8, 1.3)
        wk = 2 * np.pi * fk
        h += rng.uniform(0.3, 1.0) / (k + 1) * np.exp(-zk * wk * t) * np.sin(
            wk * np.sqrt(1 - zk**2) * t)
    return h * 1e-9                                      # coupling scale (by-analogy, v0)


# ---------------------------------------------------------------- paths
def sample_path(rng, kind, r_detect, duration_cap=120.0):
    """Returns (path_xy fn of t, t_grid, duration). Sensor at origin."""
    speed = dict(walk=rng.uniform(0.8, 1.8), run=rng.uniform(2.5, 4.0),
                 animal=rng.uniform(0.5, 3.0), vehicle=rng.uniform(4, 25))[kind]
    d = rng.uniform(1.5, 0.6 * r_detect)                 # closest approach
    style = rng.choice(["pass_by", "oblique", "approach_leave", "partial"],
                       p=[0.4, 0.25, 0.2, 0.15])
    dur = float(np.clip(2 * r_detect / speed, 8, duration_cap))
    t = np.arange(0, dur, 0.1)
    if style == "pass_by":
        x = speed * (t - dur / 2); y = np.full_like(t, d)
    elif style == "oblique":
        ang = rng.uniform(0.3, 1.2)
        x = speed * np.cos(ang) * (t - dur / 2)
        y = d + speed * np.sin(ang) * (t - dur / 2) * rng.choice([-1, 1])
    elif style == "approach_leave":
        half = dur / 2
        rr = np.where(t < half, d + speed * (half - t), d + speed * (t - half))
        ang = rng.uniform(0, 2 * np.pi)
        x = rr * np.cos(ang); y = rr * np.sin(ang)
    else:                                                # partial traverse
        x = speed * (t - dur * 0.8); y = np.full_like(t, d)
    # meander wobble (radial-speed decoupling, spec §6)
    wob = rng.uniform(0, 0.3) * d
    x = x + wob * np.sin(2 * np.pi * t / rng.uniform(8, 20))
    return np.stack([x, y], 1), t, dur, dict(style=style, speed=speed, d=float(d))


# ---------------------------------------------------------------- assembly
def assemble(emissions_list, bank_or_floor, dur, rng, is_floor=False):
    """Sum all sources' emissions into one ground-velocity timeline (m/s)."""
    n_scene = int(dur * FS)
    v = np.zeros(n_scene + (6000 if not is_floor else 4000))
    for emissions in emissions_list:
        for (t0, x, y, fz, fx) in emissions:
            i0 = int(t0 * FS)
            if is_floor:
                ir = bank_or_floor
                w = np.convolve(fz, ir)[: len(ir)]
            else:
                r = float(np.hypot(x, y))
                if r > 350:
                    continue
                heading = (1.0, 0.0)
                w = bank_or_floor.emit(r, heading, (x, y), fz, fx)
            seg = min(len(w), len(v) - i0)
            if seg > 0:
                v[i0:i0 + seg] += w[:seg]
    return v[:n_scene]


def window_labels(clean_mv, out_mv, coarse, fs=FS, win=3.0, hop=1.5, snr_db_thresh=3.0):
    """Per-window deployment labels from EXACT SNR (clean channel rendered through the
    same chain): snr = rms(clean_window) / rms(noise_window). PLAN 4.4."""
    nw = int(win * fs); nh = int(hop * fs)
    noise_mv = out_mv.astype(np.float64) - clean_mv
    rows = []
    for i0 in range(0, len(clean_mv) - nw + 1, nh):
        cs = float(np.std(clean_mv[i0:i0 + nw]))
        ns = float(np.std(noise_mv[i0:i0 + nw])) + 1e-9
        snr = 20 * np.log10(max(cs, 1e-12) / ns)
        label = coarse if snr > snr_db_thresh else "nothing"
        rows.append((round(i0 / fs, 2), label, round(snr, 1)))
    return rows
