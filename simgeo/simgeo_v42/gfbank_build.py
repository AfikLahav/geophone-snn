"""GF-bank builder, OPTIMIZED (verified 2026-06-11, ~4-6x vs naive):
 1. Fz+Fx as TWO SOURCES IN ONE pyprop8 CALL — shares the propagator matrices
    (verified bit-identical to separate runs, rel err ~1e-15).
 2. Computed at 250 Hz Nyquist (dt=2 ms) — the wavenumber stencil only converges
    to ~250 Hz anyway; the 250-500 Hz half of a 1 ms bank is unconverged junk.
    Render upsamples to 1000 Hz by spectrum zero-padding (exact for band-limited).
 3. Stencil trimmed per profile stiffness (cost is linear in nk — measured).
 4. Adaptive r_max / duration: lossy soft profiles carry nothing beyond the
    Q-attenuation horizon; don't compute dead range/time.

Stores RAW STEP-FORCE DISPLACEMENT per radius (km per 1e15 N); the (iω)² chain +
t* are applied at render. Bank: z_fz, z_fx (nr, nt), radii_km, dt, meta.
Usage: python gfbank_build.py [nworkers] [profile_id ...]
"""
import os, sys, json, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BANKS = os.path.join(HERE, "..", "..", "terrain_models")
DT = 2e-3                                                # 250 Hz Nyquist (see header)


def stencil_for(profile):
    """Full numerical configuration per profile (literature-grounded, 2026-06-11;
    see Bouchon & Aki 1977 / Bouchon 2003 discrete-wavenumber theory).

    Discretizing k at spacing dk = kmax/nk creates GHOST source copies at spatial
    period L = 2*pi/dk; the nearest ghost arrives at t = L/Vp_max. For stiff media
    that lands INSIDE the window (the convergence failures: stiff 100-1900% off,
    soft fine because its ghost arrives long after the window). Cure, two-sided:
      - contour damping alpha*t_ghost >= 13.5 suppresses the ghost to ~2e-6
        (1e-3 left ~5% residue at far radii where the signal itself decays ~40 dB,
        since the ghost scales with NEAR-field amplitude), capped by re-amplification
        (exp(+alpha*t) noise growth): alpha <= 9/T (float64 headroom).
      - where the cap would bind, nk RISES instead (alpha cannot rescue an
        undersampled grid): nk >= 0.239*kmax*Vp_max*T  (alpha at cap, ghost time pushed out).
    kmax: (a) slowest Rayleigh at 250 Hz, (b) near-field evanescent tail at 1 m
    depth needs ~10000/km regardless of stiffness. Window T: residence of the
    slowest waves + coda; stiff non-inverse profiles ring briefly -> short T
    (which also cheapens their ghost bound); inverse keeps a modest ring margin
    (the settle-clamp owns the late ring)."""
    vr_kms = max(profile["vr_ms"], 80.0) / 1000.0
    q = profile["q"]
    r_max = float(np.clip(6.9 * q * vr_kms / (np.pi * 25.0), 0.06, 0.32))
    inv = bool(profile.get("inverse"))
    stiff = profile["vs_top_ms"] >= 800 and not inv
    coda = 1.5 if stiff else 2.5
    t_extra = 1.5 if inv else 0.0
    T = r_max / vr_kms + coda + t_extra
    nt = int(np.ceil(T / DT / 2) * 2)
    kmax = float(np.clip(max(1.25 * 2 * np.pi * 250.0 / vr_kms, 10000.0),
                         10000, 24000))
    vp_max = max(l[1] for l in profile["layers"])        # km/s (sampler caps 6.0)
    nk = int(np.clip(max(kmax * 0.85, 0.239 * kmax * vp_max * T),
                     8000, 30000))
    dk = kmax / nk
    alpha = float(min(2.15 * dk * vp_max, 9.0 / T))
    if 2.15 * dk * vp_max > 13.8 / T:
        print(f"  WARNING {profile['profile_id']}: ghost bound unmet at nk cap "
              f"(dk={dk:.2f}, vp={vp_max:.1f}, T={T:.1f})", flush=True)
    return dict(kmax=kmax, nk=nk, alpha=alpha, r_max=r_max, nt=nt)


def build_one(profile):
    """Crash-tolerant: an exception in ONE profile must not kill an unattended
    overnight run — log it, keep building, report the casualty list at the end."""
    pid = profile["profile_id"]
    t0 = time.time()
    try:
        return _build_one_inner(profile, pid, t0)
    except Exception as e:
        return pid, time.time() - t0, f"FAIL {type(e).__name__}: {e}"


def _build_one_inner(profile, pid, t0):
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")        # no BLAS oversubscription
    from pyprop8 import (LayeredStructureModel, PointSource, ListOfReceivers,
                         compute_seismograms)
    out = os.path.join(BANKS, f"{pid}.npz")
    if os.path.exists(out):
        return pid, 0.0, "cached"
    model = LayeredStructureModel([tuple(l) for l in profile["layers"]])
    sten = stencil_for(profile)
    r_max = sten["r_max"]
    n_rad = int(np.clip(round(48 * np.log(r_max / 0.0005) /
                              np.log(0.32 / 0.0005)), 36, 48))
    radii = np.geomspace(0.0005, r_max, n_rad)
    st = ListOfReceivers(radii, np.zeros_like(radii), depth=0.0)
    F2 = np.stack([np.array([[0.], [0.], [1.]]),         # Fz
                   np.array([[1.], [0.], [0.]])])        # Fx
    src = PointSource(0., 0., 0.001, np.zeros((2, 3, 3)), F2, 0.)   # 1 m burial
    tt, seis = compute_seismograms(
        model, src, st, nt=sten["nt"], dt=DT, alpha=sten["alpha"],
        stencil_kwargs={"kmin": 0, "kmax": sten["kmax"], "nk": sten["nk"]},
        number_of_processes=1, show_progress=False)
    a = np.asarray(seis)                                  # (2, nr, 3, nt)
    np.savez_compressed(out, radii_km=radii, dt=DT, meta=json.dumps(profile),
                        z_fz=a[0, :, 2, :].astype(np.float32),
                        z_fx=a[1, :, 2, :].astype(np.float32))
    return pid, time.time() - t0, "built"


def main():
    """Usage: python gfbank_build.py [nworkers] [profile_id profile_id ...]
    With profile_ids given, builds only those (test mode).
    RESTART-SAFE: library.json is written ONCE and loaded thereafter — never
    regenerated (a code/seed drift between runs would silently re-roll profiles
    while cached banks keep the old draws). Delete library.json deliberately to
    re-roll. Sampler+builder code snapshotted alongside for provenance."""
    sys.path.insert(0, HERE)
    os.makedirs(BANKS, exist_ok=True)
    libpath = os.path.join(BANKS, "library.json")
    if os.path.exists(libpath):
        lib = json.load(open(libpath))
        print(f"loaded existing library.json ({len(lib)} profiles) — NOT regenerated")
    else:
        from profiles import build_library
        lib = build_library()
        with open(libpath, "w") as f:
            json.dump(lib, f, indent=1)
        import shutil
        for src in ("profiles.py", "gfbank_build.py"):
            shutil.copy(os.path.join(HERE, src),
                        os.path.join(BANKS, f"provenance_{src}"))
        print(f"library.json written ({len(lib)} profiles) + code snapshot")
    lib = [p for p in lib if not p.get("modal")]          # floors need no bank
    if "--skip-inverse" in sys.argv:
        lib = [p for p in lib if not p.get("inverse")]    # held for the waveguide fix
    only = set(a for a in sys.argv[2:] if not a.startswith("--"))
    if only:
        lib = [p for p in lib if p["profile_id"] in only]
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    print(f"{len(lib)} profiles, {nw} workers", flush=True)
    t0 = time.time()
    fails = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for pid, dt_s, status in ex.map(build_one, lib):
            print(f"  {pid}: {status} ({dt_s:.0f}s)", flush=True)
            if status.startswith("FAIL"):
                fails.append(pid)
    print(f"done in {(time.time()-t0)/60:.1f} min; {len(fails)} failures"
          + (f": {fails}" if fails else ""), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
