"""Phase A — SNR(r) inverse maps for the v4 target-SNR sampler.

For every profile (340 v3 banks) x class (human/vehicle/animal) x closest-approach
distance, render single-source pass-by scenes under the CALM ambient and the v4
re-anchored coupling (coupling.anchor_fc — maps MUST match the corpus sensor model),
and record the median active-window in-band SNR. The B1 sampler inverts these curves:
target SNR -> closest-approach distance.

Method mirrors simgeo_v3_calib/_snr_distance_map_calib.py:_render_one, widened to all
banks, with (a) anchor_fc coupling and (b) the corpus r>320 m assembly cutoff
(generate_corpus.py:173), not scenes.assemble's 350.

Output: G:/geophone_synth/config/snr_maps_v42.npz + subkind_ceilings_v42.json. Resume-aware
(per-profile npy chunks under snr_maps_v42_chunks).

v4.2 [R4]: MUST be launched under the SAME sensor env as the corpus render so the maps match —
specifically GEO_COUPLING_MODE=mix (each rep draws lowpass/bump; the per-rep median is the
expected in-band SNR under the mixture) and GEO_LINES_MODE=v41. The gain/quant/rail spans do NOT
affect the maps (SNR is gain-invariant, and quant/rail touch only out_mv which the map ignores).
  $env:GEO_COUPLING_MODE="mix"; $env:GEO_LINES_MODE="v41"; python simgeo_v42/snr_maps.py 8
Usage: python simgeo_v42/snr_maps.py [nworkers=15]
"""
import os, sys, json, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB_ROOT = os.environ.get("GEO_DB_ROOT", r"G:/geophone_synth")   # <- set GEO_DB_ROOT to your data location
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
CFG_DIR = DB_ROOT + r"/config"
CHUNKS = os.path.join(CFG_DIR, "snr_maps_v42_chunks")          # v4.2 [R4]: rebuilt under mix coupling
OUT_NPZ = os.path.join(CFG_DIR, "snr_maps_v42.npz")
CEIL_OUT = os.path.join(CFG_DIR, "subkind_ceilings_v42.json")  # v4.2 Change 2: F-gate reachability

FS = 1000.0
CLASSES = ["human", "vehicle", "animal"]                       # fixed order in the output
REF = {"human": ("walk", (1.0, 1.8)), "vehicle": ("car", (8.0, 18.0)),
       "animal": ("horse", (2.0, 4.0))}
D_GRID = {
    "human":   [1, 2, 3.5, 5, 7.5, 10, 15, 20, 30, 40, 50],
    "animal":  [1, 2, 3.5, 5, 7.5, 10, 15, 20, 30, 40],
    "vehicle": [2, 3.5, 5, 7.5, 10, 15, 20, 30, 45, 65, 90, 130, 180, 250, 320],
}
ND_MAX = max(len(v) for v in D_GRID.values())
# v4.2 [R4]: RUN this with GEO_COUPLING_MODE=mix so each rep draws lowpass/bump 50/50 -> the
# per-rep median IS the expected in-band SNR under the corpus form mixture (the residual per-scene
# form scatter is the ~5 dB the coverage gate already absorbs). N_REP env-overridable (>=3).
N_REP = int(os.environ.get("GEO_SNRMAP_NREP", "3"))
BASE_SEED = 20260706
R_CUTOFF = 320.0                                              # corpus assembly cutoff


def _stationary_path(d, dur=12.0):
    """Source held at fixed range d (small x wobble so azimuth is defined). Removes the
    pass-by 'which window is the peak' lottery -> a stable, monotone SNR(d) that the sampler
    can invert. For a 3 s window a walking source barely moves, so SNR(range=d) ~ the
    moving-scene closest-approach window (within the acknowledged +-few dB scatter)."""
    t = np.arange(0.0, dur, 0.1)
    x = 0.3 * np.sin(2 * np.pi * t / 4.0); y = np.full_like(t, float(d))
    return np.stack([x, y], 1), t, dur


def _assemble320(emissions, bank, dur):
    """gen_scene-convention assembly (r>320 cutoff), single source."""
    n = int(dur * FS); v = np.zeros(n + bank.n)
    for (t0, x, y, fz, fx) in emissions:
        r = float(np.hypot(x, y))
        if r > R_CUTOFF:
            continue
        w = bank.emit(r, (1.0, 0.0), (x, y), fz, fx)
        i0 = int(t0 * FS); seg = min(len(w), len(v) - i0)
        if seg > 0:
            v[i0:i0 + seg] += w[:seg]
    return v[:n]


def _one_render(cls, bank, d, rng, src, sensor, r3_noise, label, anchor_fc):
    band = label.BANDS[cls]
    subkind, _ = REF[cls]
    path, tg, dur = _stationary_path(float(d))
    if cls == "human":
        em, _ = src.human(path, tg, rng, bank.n, gait="walk")
    elif cls == "vehicle":
        em, _ = src.vehicle(path, tg, rng, bank.n, kind="car")
    else:
        em, _ = src.animal(path, tg, rng, bank.n, kind=subkind)
    v_sig = _assemble320(em, bank, dur)
    n = len(v_sig)
    v_noise = r3_noise.ground_noise(n, 0.0, 0.0, rng)          # calm
    fc = anchor_fc(bank.meta["coupling_fc"], rng)              # v4 re-anchored coupling
    _, clean_mv, noise_mv, _ = sensor.render_hp(v_sig, v_noise, rng, fc,
                                                bank.meta["coupling_q"], p_lines=0.0)
    clean_mv = clean_mv.astype(np.float64); noise_mv = noise_mv.astype(np.float64)
    nw = int(label.WIN * FS); lo, hi = band
    # stationary source at range d: all windows are at ~range d -> take the MEDIAN window SNR
    # over windows that actually contain source energy. Stable + monotone in d.
    snrs = []
    for i0 in range(0, max(1, n - nw + 1), int(label.HOP * FS)):
        cs = label._band_rms(clean_mv[i0:i0 + nw], lo, hi)
        if cs > 1e-9:
            ns = label._band_rms(noise_mv[i0:i0 + nw], lo, hi) + 1e-12
            snrs.append(20.0 * np.log10(max(cs, 1e-12) / ns))
    return float(np.median(snrs)) if snrs else np.nan


def run_profile(pid):
    chunk = os.path.join(CHUNKS, f"{pid}.npy")
    if os.path.exists(chunk):
        return pid, "skip"
    import sources_v2 as src, sensor, r3_noise, label
    from scenes import Bank
    from coupling import anchor_fc
    bank = Bank.get(pid)
    med = np.full((len(CLASSES), ND_MAX), np.nan)
    p10 = np.full((len(CLASSES), ND_MAX), np.nan)
    p90 = np.full((len(CLASSES), ND_MAX), np.nan)
    for ci, cls in enumerate(CLASSES):
        for di, d in enumerate(D_GRID[cls]):
            vals = []
            for r in range(N_REP):
                seed = (BASE_SEED * 1000003 + hash((pid, cls, di, r)) % 100000) % (2**32)
                rng = np.random.default_rng(seed)
                vals.append(_one_render(cls, bank, float(d), rng, src, sensor, r3_noise, label, anchor_fc))
            vals = np.array(vals, float)
            if np.isfinite(vals).any():
                med[ci, di] = np.nanmedian(vals)
                p10[ci, di] = np.nanpercentile(vals, 10)
                p90[ci, di] = np.nanpercentile(vals, 90)
    np.save(chunk, np.stack([med, p10, p90]))                 # [3, nclass, ND_MAX]
    return pid, "done"


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    os.makedirs(CHUNKS, exist_ok=True)
    lib = json.load(open(os.path.join(ROOT, "..", "terrain_models", "library_v3.json")))
    pids = [p["profile_id"] for p in lib if not p.get("modal")]
    print(f"snr_maps: {len(pids)} profiles x {len(CLASSES)} classes, {nw} workers", flush=True)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for pid, st in ex.map(run_profile, pids):
            done += 1
            if done % 20 == 0 or st == "skip":
                print(f"  {done}/{len(pids)} ({st}) {(time.time()-t0)/60:.1f} min", flush=True)
    # merge chunks -> npz
    import r3_noise; r3_noise._load()
    COND = r3_noise.COND_BINS
    cond_shift = np.array([10 * np.log10(max(r3_noise._LEVELS[c], 1e-9)) for c in COND], np.float32)
    med = np.full((len(pids), len(CLASSES), ND_MAX), np.nan, np.float32)
    p10 = np.full_like(med, np.nan); p90 = np.full_like(med, np.nan)
    miss = 0
    for i, pid in enumerate(pids):
        f = os.path.join(CHUNKS, f"{pid}.npy")
        if not os.path.exists(f):
            miss += 1; continue
        a = np.load(f); med[i] = a[0]; p10[i] = a[1]; p90[i] = a[2]
    kw = dict(pids=np.array(pids), classes=np.array(CLASSES),
              snr_med=med, snr_p10=p10, snr_p90=p90,
              cond_bins=np.array(COND), cond_shift_db=cond_shift, nd_max=ND_MAX)
    for c in CLASSES:
        kw[f"d_grid_{c}"] = np.array(D_GRID[c], np.float32)
    np.savez(OUT_NPZ, **kw)
    print(f"DONE {len(pids)-miss}/{len(pids)} profiles ({miss} missing) in "
          f"{(time.time()-t0)/60:.1f} min -> {OUT_NPZ}", flush=True)

    # v4.2 Change 2: per-subkind reachability ceiling = max realizable in-band SNR = the class
    # REFERENCE SNR at MIN_STANDOFF under CALM (the loudest case: closest distance, no ambient
    # shift) PLUS the subkind source-level offset. The F gate exempts bins above this ceiling as
    # physically unreachable (a bicycle -22 dB below car simply cannot fill the loud bins).
    import generate_corpus_v4 as gcv                            # SUBKIND_SNR_OFFSET_DB + subkind lists
    subk_class = {}
    for sk, _ in gcv.HUMAN_V4:   subk_class[sk] = "human"
    for sk, _ in gcv.VEHICLE_V4: subk_class[sk] = "vehicle"
    for sk, _ in gcv.ANIMAL_V4:  subk_class[sk] = "animal"
    ceil_ref = {}
    for ci, c in enumerate(CLASSES):
        col0 = med[:, ci, 0]                                    # d[0] = closest, calm
        ceil_ref[c] = (round(float(np.nanmax(col0)), 2) if np.isfinite(col0).any() else None)
    ceilings = {}
    for sk, c in subk_class.items():
        if ceil_ref.get(c) is not None:
            ceilings[sk] = round(ceil_ref[c] + float(gcv.SUBKIND_SNR_OFFSET_DB.get(sk, 0.0)), 2)
    json.dump({"ceiling_ref_calm_closest_db": ceil_ref, "subkind_ceiling_db": ceilings,
               "note": "max realizable in-band SNR = class ref SNR at MIN_STANDOFF (calm) + subkind "
                       "offset (SUBKIND_SNR_OFFSET_DB); F-gate bins above this are unreachable"},
              open(CEIL_OUT, "w"), indent=1)
    print(f"wrote {CEIL_OUT}: ref={ceil_ref}", flush=True)


if __name__ == "__main__":
    main()
