"""v4 PRODUCTION render — target-SNR sampling + elaborate/hard scenes + decoupled labeling.

Self-contained (imports unchanged constants/helpers from generate_corpus as gc so the v3/v2
generators stay byte-identical). Key differences vs v3 (GENERATION_PLAN_V4 rev 4):
  - Closest approach is DERIVED from a target in-band SNR (sample_target_snr -> snr_to_distance
    via the Phase-A snr_maps_v4.npz), not sampled directly -> fills all loudness levels.
  - Per-SUBKIND range caps (R_MAX_V4), literature-corrected (car<=120 m, not v3's 400).
  - Elaborate subkinds: march/loaded human, slow_quad/horse_rider animal, idle_exit vehicle.
  - Pause/gap scenes (D4): per-source emission gaps; scene noise bed runs continuously.
  - Activity-aware labels (D3): per-source emission intervals -> activity_json (Phase L reads it).
  - Per-class clean blobs for mixed scenes (D2): labels-only answer key; model sees the sum.
  - Coupling re-anchored to the measured-real value (D5, coupling.anchor_fc).
  - NO windows table at render: labeling is a separate parameterized pass (label_windows.py).

Usage: python simgeo_v42/generate_corpus_v4.py <n_scenes> <out_dir(basename corpus_v42 or
       corpus_v42_pilot*)> <nw> [max_specs]

v4.2 (V4_2_PLAN.md): weather is a class-blind global TIER axis (Change 1), SNR inversion is
subkind-offset-aware (Change 2 [R1]), the sensor chain adds gain/quant/rail/coupling-form domain
randomization recorded per scene (Change 3 [R5]). Everything is env-flagged; flags at their v4
defaults reproduce v4. Output roots: corpus_v42 / labels_v42 / features_v42_3s / snr_maps_v42.npz.
"""
import os, sys, json, sqlite3, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DB_ROOT = os.environ.get("GEO_DB_ROOT", r"G:/geophone_synth")   # <- set GEO_DB_ROOT to your data location
sys.path.insert(0, HERE)
import generate_corpus as gc                         # unchanged constants/helpers (MIN_STANDOFF, LINE_TIER, ...)
import sources_v2 as src
import sensor, label, paths, r3_noise
from scenes import Bank
from coupling import anchor_fc

FS = 1000.0
BASE_SEED_V4 = 20260706
SNRMAP_PATH = DB_ROOT + r"/config/snr_maps_v42.npz"           # v4.2: rebuilt under mix coupling
SUBKIND_CEILINGS_PATH = DB_ROOT + r"/config/subkind_ceilings_v42.json"

# --- target-SNR sampling (density anchored to the 3 s gates; scoring gates are separate) ---
GATES3 = json.load(open(os.path.join(HERE, "..", "..", "dataset_validation", "gates.json")))["classes"]
SNR_TRI = {"human": (-18.89, 1.5, 40.0), "vehicle": (-13.44, 7.5, 45.0), "animal": (-18.89, 4.5, 45.0)}
P_SUBFLOOR = 0.15
R_MAX_V4 = {"human": 50.0, "animal": 40.0, "vehicle": 200.0,      # class-level fallbacks
            "bicycle": 120.0, "motorbike": 120.0, "car": 120.0, "idle_exit": 120.0,
            "convoy": 200.0, "two_vehicle": 200.0, "truck": 200.0, "tractor": 200.0, "tracked": 320.0}
DUR_MIN_V4, DUR_CAP_V4 = 20.0, 60.0
R_PATH_MUL = 1.6            # engagement half-span = R_PATH_MUL x closest approach (tight -> windows
                           # cluster near the target loudness instead of the far faint tail)
P_GAP, GAP_N, GAP_S = 0.25, (1, 2), (1.0, 3.0)
HET_R_MIN, HET_VS_MAX, HET_SIG_DB = 200.0, 400.0, 3.0
R_CUTOFF = 320.0

# weighted subkind cells
HUMAN_V4 = [("walk", 2), ("run", 1), ("stealth", 1), ("child", 1), ("group", 1), ("march", 1), ("loaded", 1)]
VEHICLE_V4 = [("car", 2), ("truck", 1), ("motorbike", 1), ("bicycle", 1), ("tractor", 1),
              ("tracked", 1), ("convoy", 1), ("two_vehicle", 1), ("idle_exit", 1)]
ANIMAL_V4 = [("dog", 1), ("jackal", 1), ("boar", 1), ("horse", 1), ("sheep", 1), ("herd", 1),
             ("slow_quad", 2), ("horse_rider", 1)]
NOTHING_V4 = [("calm", 1), ("wind", 2), ("rain", 1), ("machinery", 1), ("overflight", 1), ("traffic", 1)]
# v4.2 Change 1: weather is no longer double-encoded as nothing subkinds. "ambient" = pure
# background under whatever weather the class-blind axis dealt (weight 3 keeps pure background
# ~half of nothing); machinery/overflight/traffic remain CONTENT confusers (not ambient).
NOTHING_V42 = [("ambient", 3), ("machinery", 1), ("overflight", 1), ("traffic", 1)]
MIXED_V4 = ["human+vehicle", "human+animal", "vehicle+animal"]
MIXED_TARGET = 18000
CONFUSER = ("machinery", "overflight", "traffic")

# --- v4.2 Change 1: weather = class-blind global axis --------------------------------------
# A declared marginal over 6 weather TIERS, dealt deterministically per (pid x coarse x subkind)
# cell by largest-remainder (NOT per-scene i.i.d.) so the tier marginal is IDENTICAL across every
# class -> exact class-blindness (kills the P/N-condition leak). Tiers map 1:1 to the
# r3_noise.condition_of bins.
WEATHER_TIERS = ["calm", "wind_low", "wind_mid", "wind_high", "rain_light", "rain_heavy"]
WEATHER_MARGINAL = {"calm": 0.35, "wind_low": 0.20, "wind_mid": 0.15,
                    "wind_high": 0.10, "rain_light": 0.12, "rain_heavy": 0.08}
# Continuous draw INSIDE a tier, class-independent. Ranges sit STRICTLY inside condition_of's bins
# (rain>0.5 -> light, rain>5 -> heavy; wind>=3/6/10 -> low/mid/high) so condition_of(wind,rain)==tier
# for EVERY draw incl. endpoints -> the assert in gen_scene is provably safe.
WEATHER_TIER_RANGES = {                       # (wind_lo, wind_hi, rain_lo, rain_hi); rain_lo None -> 0
    "calm":       (0.0, 3.0,  None, None),
    "wind_low":   (3.0, 6.0,  None, None),
    "wind_mid":   (6.0, 10.0, None, None),
    "wind_high":  (10.0, 16.0, None, None),
    "rain_light": (0.0, 6.0,  0.6, 5.0),
    "rain_heavy": (0.0, 8.0,  5.5, 30.0),
}


def weather_from_tier(tier, rng):
    """Draw (wind_ms, rain_mmh) uniformly inside the tier's ranges. Applied to ALL coarse types
    identically (no per-coarse branch) -> the ambient draw law is shared across classes."""
    wlo, whi, rlo, rhi = WEATHER_TIER_RANGES[tier]
    wind = float(rng.uniform(wlo, whi))
    rain = float(rng.uniform(rlo, rhi)) if rlo is not None else 0.0
    return wind, rain


def deal_tiers(count):
    """Largest-remainder deal of WEATHER_MARGINAL over `count` scenes -> a length-`count` list of
    tiers (fixed WEATHER_TIERS order; tie-break by tier order). Deterministic (no rng). Every cell
    of a given size gets the SAME tier multiset -> class-blind marginal by construction."""
    if count <= 0:
        return []
    raw = {t: WEATHER_MARGINAL[t] * count for t in WEATHER_TIERS}
    base = {t: int(np.floor(raw[t])) for t in WEATHER_TIERS}
    rem = int(count - sum(base.values()))
    order = sorted(WEATHER_TIERS, key=lambda t: (-(raw[t] - base[t]), WEATHER_TIERS.index(t)))
    for t in order[:rem]:
        base[t] += 1
    out = []
    for t in WEATHER_TIERS:
        out += [t] * base[t]
    return out


# --- v4.2 Change 2 [R1]: per-subkind source-level offset (dB rel. each class map's REFERENCE
# source: vehicle=car, human=walk, animal=horse). Applied at the snr_to_distance CALL SITE keyed on
# the REAL subkind -> the car/walk/horse map is inverted at (snr_t - offset) so the subkind REALIZES
# snr_t. Quiet subkinds (bicycle -22) get placed closer (fills the starved loud bins); loud subkinds
# (tracked +10) get placed farther. Starting values; the pilot F-gate refines. convoy/two_vehicle
# member kinds and group/march/herd counts are drawn AFTER the closest-approach solve -> a
# representative group-level offset is used for those (documented).
SUBKIND_SNR_OFFSET_DB = {
    # vehicle (ref = car)
    "car": 0.0, "truck": 4.0, "tractor": 2.0, "tracked": 10.0,
    "motorbike": -10.0, "bicycle": -22.0, "idle_exit": -4.0, "convoy": 0.0,
    # human (ref = walk)
    "walk": 0.0, "run": 3.0, "child": -4.0, "stealth": -6.0, "loaded": 1.0,
    "group": 4.0, "march": 6.0,
    # animal (ref = horse)
    "horse": 0.0, "boar": -2.0, "sheep": -4.0, "dog": -8.0, "jackal": -12.0,
    "herd": -4.0, "slow_quad": -2.0, "horse_rider": 0.0,
}

_SNR = None
def _snrmap():
    global _SNR
    if _SNR is None:
        z = np.load(SNRMAP_PATH, allow_pickle=True)
        _SNR = {"pids": list(z["pids"]), "classes": list(z["classes"]),
                "med": z["snr_med"], "cond": list(z["cond_bins"]), "shift": z["cond_shift_db"],
                "d": {c: z[f"d_grid_{c}"] for c in z["classes"]}}
    return _SNR


def _monotone_env(a):
    return np.fmin.accumulate(np.where(np.isfinite(a), a, -np.inf))


def sample_target_snr(rng, c):
    lo, mode, hi = SNR_TRI[c]
    if rng.random() < P_SUBFLOOR:
        return float(rng.uniform(lo - 15.0, lo))
    return float(rng.triangular(lo, mode, hi))


def snr_to_distance(pid, c, subkind, snr_t, cond, rng):
    """Invert the calm-anchored SNR(d) curve at this profile/class/condition -> closest approach.
    log-d interp on the monotone envelope; clip to [MIN_STANDOFF, R_MAX_V4[subkind]]."""
    m = _snrmap()
    pi = m["pids"].index(pid); ci = m["classes"].index(c)
    d = m["d"][c]; nd = len(d)
    shift = float(m["shift"][m["cond"].index(cond)]) if cond in m["cond"] else 0.0
    curve = _monotone_env(m["med"][pi, ci, :nd]) - shift        # SNR under this ambient
    lo, hi = gc.MIN_STANDOFF[c], R_MAX_V4.get(subkind, R_MAX_V4[c])
    if not np.isfinite(curve).any():
        return float(rng.uniform(lo, hi)), True
    if snr_t >= curve[0]:
        r = float(d[0])
    elif snr_t <= curve[-1]:
        r = float(d[-1])
    else:
        r = float(10 ** np.interp(-snr_t, -curve, np.log10(d)))
    rc = float(np.clip(r, lo, hi))
    return rc, (rc != r)


def _merge(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs); out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 2), round(b, 2)] for a, b in out]


def _apply_gaps(ems, dur, rng, has_gaps):
    """Drop emissions inside per-source gaps; return (kept_ems, activity_intervals)."""
    gaps = []
    if has_gaps and rng.random() < 0.7:                         # this source pauses
        for _ in range(int(rng.integers(GAP_N[0], GAP_N[1] + 1))):
            g0 = float(rng.uniform(0.15 * dur, 0.85 * dur)); gl = float(rng.uniform(*GAP_S))
            gaps.append((g0, g0 + gl))
    kept = [e for e in ems if not any(g0 <= e[0] < g1 for g0, g1 in gaps)]
    ivs = _merge([(e[0] - 0.05, e[0] + 0.45) for e in kept])
    return kept, ivs


def _speed(c, csub, rng):
    if csub == "stealth": return rng.uniform(0.2, 0.6)
    if csub == "run": return rng.uniform(2.5, 4)
    if csub == "march": return rng.uniform(1.0, 1.6)
    if csub == "loaded": return rng.uniform(0.8, 1.4)
    if csub == "slow_quad": return rng.uniform(0.4, 1.0)
    if csub == "tractor": return rng.uniform(1, 8)
    if c == "vehicle": return rng.uniform(4, 22)
    return rng.uniform(0.6, 3.0)


def _emit_moving(c, csub, cond, has_gaps, rng, pid, bank, force_snr=None):
    """Emit one moving single-class subject at a target-SNR-derived closest approach, with gaps.
    Returns (ems, dur, ns, infos, speed, pinfo, activity_intervals)."""
    n_bank = bank.n
    sp = _speed(c, csub, rng)
    cap_key = csub if c == "vehicle" else c
    snr_t = float(force_snr) if force_snr is not None else sample_target_snr(rng, c)
    # v4.2 [R1]: offset keyed on the REAL subkind (csub), NOT cap_key (cap_key=="human"/"animal"
    # for non-vehicle -> would leave every gait/species at offset 0). cap_key stays for R_MAX.
    snr_map_target = snr_t - SUBKIND_SNR_OFFSET_DB.get(csub, 0.0)
    r_t, clipped = snr_to_distance(pid, c, cap_key, snr_map_target, cond, rng)
    r_path = float(np.clip(R_PATH_MUL * r_t, 8.0, R_MAX_V4.get(cap_key, R_MAX_V4[c])))
    path, tg, dur, pi = paths.sample(rng, r_path, sp, dur_cap=DUR_CAP_V4)
    dur = max(dur, DUR_MIN_V4)
    path, r_real = gc.retarget_closest_approach(path, r_t)
    pi = {**pi, "d": round(float(r_real), 1), "snr_t": round(snr_t, 1), "clip": int(clipped)}
    ems = []; infos = []; ns = 1
    if c == "human":
        if csub in ("group", "march"):
            ns = int(rng.choice([4, 6, 8])) if csub == "march" else int(rng.choice([2, 3, 4]))
            coh = rng.uniform(0.7, 1.0) if csub == "march" else rng.uniform(0.0, 0.4)
            cad = float(rng.choice([2.0, 3.0])) if csub == "march" else float(rng.uniform(1.6, 2.1))
            pay = float(rng.uniform(10, 30)) if csub == "march" else 0.0
            ems, info = src.human_group_coherent(path, tg, rng, n_bank, ns, coh, cad, payload_kg=pay)
            infos.append(info); pi["coherence"] = round(coh, 2); pi["payload_kg"] = round(pay, 1)
        elif csub == "loaded":
            pay = float(rng.uniform(15, 35))
            ems, info = src.human(path, tg, rng, n_bank, "walk", payload_kg=pay, cadence_mul=rng.uniform(1.0, 1.1))
            infos.append(info); pi["payload_kg"] = round(pay, 1)
        else:
            gait = csub
            if csub == "walk" and rng.random() < gc.P_STEALTH_WINDOW: gait = "stealth"
            off = rng.uniform(0, 2, 2)
            ems, info = src.human(path + off, tg, rng, n_bank, gait); infos.append(info)
    elif c == "vehicle":
        if csub == "convoy":
            ns = int(rng.choice([2, 2, 3])); kind = str(rng.choice(["car", "car", "truck", "motorbike"]))
            offs = np.cumsum([0.0] + [float(rng.uniform(2, 8)) for _ in range(ns - 1)])
            for j in range(ns):
                em, info = src.vehicle(path, tg, rng, n_bank, kind)
                ems += [(t + offs[j], x, y, fz, fx) for (t, x, y, fz, fx) in em]
                infos.append({**info, "headway_s": round(float(offs[j]), 2)})
            dur = float(dur + offs[-1])
        elif csub == "two_vehicle":
            ns = 2; kinds = [str(k) for k in rng.choice(["car", "car", "truck", "motorbike", "bicycle"], 2)]
            for j, k in enumerate(kinds):
                sp2 = float(rng.uniform(4, 22)); snr2 = sample_target_snr(rng, "vehicle")
                snr2_map = snr2 - SUBKIND_SNR_OFFSET_DB.get(k, 0.0)         # v4.2 [R1] per-member
                r2, _ = snr_to_distance(pid, "vehicle", k, snr2_map, cond, rng)
                rp2 = float(np.clip(R_PATH_MUL * r2, 8, R_MAX_V4.get(k, 200.0)))
                pth, t2, d2, _ = paths.sample(rng, rp2, sp2, dur_cap=DUR_CAP_V4)
                pth, _ = gc.retarget_closest_approach(pth, r2)
                em, info = src.vehicle(pth, t2, rng, n_bank, k)
                off = 0.0 if j == 0 else float(rng.uniform(0, 0.5 * max(d2, 1.0)))
                ems += [(t + off, x, y, fz, fx) for (t, x, y, fz, fx) in em]
                infos.append({**info, "offset_s": round(off, 2)})
            dur = max(dur, DUR_MIN_V4)
        else:
            ems, info = src.vehicle(path, tg, rng, n_bank, csub); infos.append(info)
    else:                                                       # animal
        if csub == "herd":
            ns = int(rng.choice([3, 4, 6]))
            for _ in range(ns):
                species = str(rng.choice(gc.HERD_SPECIES))
                wav = species if species in ("dog", "jackal", "boar", "horse") else "boar"
                off = rng.uniform(0, 2, 2)
                em, info = src.animal(path + off, tg, rng, n_bank, wav)
                ems += em; infos.append({**info, "subkind": species})
        elif csub == "slow_quad":
            kind = str(rng.choice(["horse", "boar"]))
            ems, info = src.animal(path, tg, rng, n_bank, kind, force_gait="walk",
                                   stride_mul=float(rng.uniform(0.70, 0.90)))
            infos.append({**info, "subkind": "slow_quad"})
        elif csub == "horse_rider":
            ems, info = src.animal(path, tg, rng, n_bank, "horse", rider_kg=float(rng.uniform(60, 140)))
            infos.append({**info, "subkind": "horse_rider"})
        else:
            kind = csub if csub in ("dog", "jackal", "boar", "horse", "sheep") else "boar"
            wav = kind if kind in ("dog", "jackal", "boar", "horse") else "boar"
            off = rng.uniform(0, 2, 2)
            ems, info = src.animal(path + off, tg, rng, n_bank, wav); infos.append({**info, "subkind": kind})
    ems, ivs = _apply_gaps(ems, dur, rng, has_gaps)
    return ems, dur, ns, infos, sp, pi, ivs


def gen_scene_v42(spec):
    sid, pid, split, coarse, sub, tier = spec
    rng = np.random.default_rng((BASE_SEED_V4 * 100003 + sid) % 2**32)
    bank = Bank.get(pid); vs = bank.meta["vs_top_ms"]; fam = bank.meta["family"]; n_bank = bank.n
    fc_native, qc = bank.meta["coupling_fc"], bank.meta["coupling_q"]
    p_lines = gc.LINE_TIER.get(fam, 0.30)
    counts = {"human": 0, "vehicle": 0, "animal": 0}
    ground = {"human": None, "vehicle": None, "animal": None}
    activity = {}; subj_info = []; speed = None; pinfo = None
    mach_kind = over_kind = None
    masking = 0; het_db = 0.0

    # --- weather FIRST (v4.2 Change 1: class-blind tier -> continuous draw, applied identically to
    # ALL coarse types; the SNR inversion needs the ambient condition). The tier is dealt in
    # build_plan_v42; the continuous draw here stays strictly inside condition_of's bins so
    # cond == tier by construction (provably safe assert). ---
    wind, rain = weather_from_tier(tier, rng)
    cond = r3_noise.condition_of(wind, rain)
    assert cond == tier, f"weather draw wind={wind:.3f} rain={rain:.3f} -> {cond} != tier {tier}"
    has_gaps = rng.random() < P_GAP

    def _assemble(ems, dur):
        n = int(dur * FS); v = np.zeros(n + n_bank)
        for (t0, x, y, fz, fx) in ems:
            r = float(np.hypot(x, y))
            if r > R_CUTOFF:
                continue
            w = bank.emit(r, (1.0, 0.0), (x, y), fz, fx)
            i0 = int(t0 * FS); seg = min(len(w), len(v) - i0)
            if seg > 0:
                v[i0:i0 + seg] += w[:seg]
        return v[:n]

    def _het(g, pi):
        nonlocal het_db
        r_t = pi.get("d") if pi else None
        if r_t is not None and r_t > HET_R_MIN and vs < HET_VS_MAX:
            het_db = float(rng.normal(0, HET_SIG_DB))
            return g * 10 ** (het_db / 20.0)
        return g

    extra_ground = None
    if coarse == "nothing":
        dur = float(rng.uniform(DUR_MIN_V4, 120.0)); n = int(dur * FS); v_sig = np.zeros(n)
        if sub == "machinery":
            mk = str(rng.choice(["pump", "pump", "generator"]))
            extra_ground, mi = (src.pump if mk == "pump" else src.generator_set)(n, rng)
            mach_kind = mi.get("kind"); subj_info.append(mi)
        elif sub == "overflight":
            extra_ground, oi = src.overflight(n, rng); over_kind = oi.get("subkind"); subj_info.append(oi)
        elif sub == "traffic":
            extra_ground = r3_noise.traffic_hum(n, rng); subj_info.append({"kind": "traffic"})
    elif coarse == "vehicle" and sub == "idle_exit":
        # composite: parked idling vehicle + door slams + exiting walkers (native masking scene)
        snr_t = sample_target_snr(rng, "vehicle")
        snr_map = snr_t - SUBKIND_SNR_OFFSET_DB.get("idle_exit", 0.0)       # v4.2 [R1]
        r_t, clip = snr_to_distance(pid, "vehicle", "idle_exit", snr_map, cond, rng)
        ang = rng.uniform(0, 2 * np.pi); vx, vy = r_t * np.cos(ang), r_t * np.sin(ang)
        dur = float(rng.uniform(DUR_MIN_V4, 45.0))
        idle, ii = src.vehicle_idle((vx, vy), dur, rng, n_bank, kind=str(rng.choice(["car", "car", "truck"])))
        n_p = int(rng.choice([1, 2, 3])); vem = list(idle)
        for _ in range(n_p):
            ts = float(rng.uniform(0.2, 0.5) * dur)
            sl, _ = src.door_slam((vx, vy), ts, rng, n_bank); vem += sl
        vground = _assemble(vem, dur); pv = {"d": round(r_t, 1), "snr_t": round(snr_t, 1), "clip": int(clip)}
        ground["vehicle"] = _het(vground, pv); counts["vehicle"] = 1
        activity["vehicle"] = _merge([(0.0, dur)])              # idle spans the scene
        subj_info.append(ii)
        # exiting walkers on radial paths away from the vehicle
        hem_all = []; hivs = []
        for _ in range(n_p):
            t_exit = float(rng.uniform(0.25, 0.6) * dur)
            sp = float(rng.uniform(0.8, 1.5)); wdur = dur - t_exit
            tw = np.arange(0, max(wdur, 3.0), 0.1)
            dirn = rng.uniform(0, 2 * np.pi)
            wx = vx + sp * tw * np.cos(dirn); wy = vy + sp * tw * np.sin(dirn)
            wpath = np.stack([wx, wy], 1)
            em, hi = src.human(wpath, tw + t_exit, rng, n_bank, "walk"); hem_all += em; subj_info.append(hi)
            hivs += [(e[0] - 0.05, e[0] + 0.45) for e in em]
        ground["human"] = _het(_assemble(hem_all, dur), pv); counts["human"] = n_p
        activity["human"] = _merge(hivs)
        v_sig = ground["vehicle"] + ground["human"]; n = len(v_sig)
        speed = 0.0; pinfo = pv; masking = 1
    elif coarse == "mixed":
        c1, c2 = sub.split("+")
        do_mask = rng.random() < (0.5 if sub == "human+vehicle" else 0.35 if sub == "vehicle+animal" else 0.0)
        def _force(c):
            if not do_mask:
                return None
            if c == "vehicle":
                masking_hi = True; return float(rng.uniform(10, 40))          # masker
            g = GATES3[c]; return float(rng.uniform(g["tau_lo_db"] - 5, g["tau_hi_db"] + 5))
        d1 = "walk" if c1 == "human" else "car" if c1 == "vehicle" else "dog"
        d2 = "walk" if c2 == "human" else "car" if c2 == "vehicle" else "dog"
        e1, du1, n1, i1, s1, pi1, iv1 = _emit_moving(c1, d1, cond, has_gaps, rng, pid, bank, force_snr=_force(c1))
        e2, du2, n2, i2, s2, pi2, iv2 = _emit_moving(c2, d2, cond, has_gaps, rng, pid, bank, force_snr=_force(c2))
        dur = float(max(du1, du2)); n = int(dur * FS)
        ground[c1] = _het(_assemble(e1, dur), pi1); ground[c2] = _het(_assemble(e2, dur), pi2)
        counts[c1] += n1; counts[c2] += n2
        activity[c1] = iv1; activity[c2] = iv2
        subj_info += i1 + i2; speed = s1; pinfo = pi1
        v_sig = ground[c1] + ground[c2]; masking = int(do_mask)
    else:                                                       # single subject class
        ems, dur, ns, infos, speed, pinfo, ivs = _emit_moving(coarse, sub, cond, has_gaps, rng, pid, bank)
        ground[coarse] = _het(_assemble(ems, dur), pinfo); counts[coarse] = ns
        activity[coarse] = ivs; subj_info = infos
        v_sig = ground[coarse]; n = len(v_sig)

    # --- noise + render (coupling re-anchored) ---
    v_noise = r3_noise.ground_noise(n, wind, rain, rng) + sensor.rain_noise(n, rng, rain)
    if coarse == "nothing" and extra_ground is not None:
        v_noise = v_noise + extra_ground
    fc = anchor_fc(fc_native, rng)                              # D5: re-anchored coupling center
    out_mv, clean_mv_full, noise_mv, p = sensor.render_hp(v_sig, v_noise, rng, fc, qc, p_lines)
    # per-class clean via the SAME H (D2 answer key; sensor chain is linear -> sum == full clean).
    # v4.2: pass the SAME per-scene coupling_form render_hp drew, else H mismatches and D2 breaks.
    H = (sensor.coupling_response(n, p["fc"], p["qc"], form=p["coupling_form"]) *
         sensor.geophone_response(n, p["f0"], p["h"], p["G"]) *
         sensor.spurious_bump(n, p["f_spur"], p["q_spur"], p["spur_db"]))
    cbc = {}
    for c in ("human", "vehicle", "animal"):
        if ground[c] is not None:
            cbc[c] = (np.fft.irfft(np.fft.rfft(ground[c]) * H, n=n) * p["gain"] * 1e3).astype(np.float32)

    present = [c for c in ("human", "vehicle", "animal") if ground[c] is not None]
    if not present:                                            # nothing scene
        cm1 = cls1 = cm2 = cls2 = None
    elif len(present) == 1:
        cls1 = present[0]; cm1 = cbc[cls1].tobytes(); cls2 = None; cm2 = None
    else:
        cls1, cls2 = present[0], present[1]
        cm1 = cbc[cls1].tobytes(); cm2 = cbc[cls2].tobytes()

    n_subj = int(sum(counts.values()))
    coh = pinfo.get("coherence") if pinfo else None
    pay = pinfo.get("payload_kg") if pinfo else None
    snr_t_rec = pinfo.get("snr_t") if pinfo else None
    clip_rec = int(pinfo.get("clip", 0)) if pinfo else 0
    ca = round(float(pinfo["d"]), 1) if (pinfo and "d" in pinfo) else None
    scene_row = (sid, pid, fam, float(vs), split, coarse, str(sub), round(dur, 1), FS, n_subj,
                 ca, (round(float(speed), 2) if speed is not None else None), snr_t_rec, clip_rec,
                 cond,                                          # == tier (asserted); weather condition
                 (round(float(coh), 2) if coh is not None else None),
                 (round(float(pay), 1) if pay is not None else None),
                 round(het_db, 2), masking, round(float(p["fc"]), 1),
                 # v4.2 [R5]: weather tier + the class-independent Q_j draws, for the datasheet + N gate
                 str(tier), round(float(np.log10(p["gain"])), 3),
                 round(float(p["quant_lsb"]), 4), float(p["rail_mv"]), str(p["coupling_form"]),
                 json.dumps({"counts": counts, "activity": activity}, default=float))
    wave_row = (sid, n, noise_mv.astype(np.float32).tobytes(), cm1, cls1, cm2, cls2)
    return scene_row, wave_row


def build_plan_v42(n_target, seed=BASE_SEED_V4):
    """v4.2: adds a class-blind weather TIER to each plan tuple. Every (pid x coarse x subkind)
    class is dealt largest-remainder tier counts on the CLASS TOTAL, then shuffled and assigned
    (Change 1 [R3]) -> every class's tier marginal == WEATHER_MARGINAL to within large-N rounding,
    hence IDENTICAL across classes (exact class-blindness). Per-CELL dealing was insufficient: cells
    differ in size across classes (subj pc=8 vs nothing's larger cells), so small-cell rounding left
    a residual class-condition association (smoke measured TV 0.10). Spec is the 6-tuple
    (sid, pid, split, coarse, subkind, tier)."""
    lib = json.load(open(os.path.join(HERE, "..", "..", "terrain_models", "library_v3.json")))
    splits = json.load(open(os.path.join(HERE, "..", "..", "terrain_models", "splits_v3.json")))
    banks = [p for p in lib if not p.get("modal")]
    rng = np.random.default_rng(seed)
    subj, noth, mixed = [], [], []
    for p in banks:
        pid = p["profile_id"]; sp = splits.get(pid, "train"); veh_ok = p["vs_top_ms"] >= gc.VEH_MIN_VS_V3
        for sk, w in HUMAN_V4:
            subj += [(pid, sp, "human", sk)] * w
        if veh_ok:
            for sk, w in VEHICLE_V4:
                subj += [(pid, sp, "vehicle", sk)] * w
        for sk, w in ANIMAL_V4:
            subj += [(pid, sp, "animal", sk)] * w
        for sk, w in NOTHING_V42:                               # v4.2: ambient replaces calm/wind/rain
            noth.append((pid, sp, "nothing", sk, gc.CONFUSER_W if sk in CONFUSER else 1.0))
        for combo in MIXED_V4:
            if combo in ("human+vehicle", "vehicle+animal") and not veh_ok:
                continue
            mixed.append((pid, sp, "mixed", combo))
    rest = max(n_target - MIXED_TARGET, len(subj) + len(noth))
    noth_t = int(rest * 0.52); subj_t = rest - noth_t
    # 1) build the scene stubs (tier assigned afterwards, per-class)
    stubs = []; pc = max(1, round(subj_t / len(subj))); sid = 0
    for (pid, sp, co, sk) in subj:
        for _ in range(pc):
            stubs.append([sid, pid, sp, co, sk]); sid += 1
    wsum = sum(w for *_, w in noth); sid = 1_000_000
    for (pid, sp, co, sk, w) in noth:
        for _ in range(max(1, round(noth_t * w / wsum))):
            stubs.append([sid, pid, sp, co, sk]); sid += 1
    pcm = max(1, round(MIXED_TARGET / len(mixed))); sid = 2_000_000
    for (pid, sp, co, sk) in mixed:
        for _ in range(pcm):
            stubs.append([sid, pid, sp, co, sk]); sid += 1
    # 2) deal tiers PER COARSE CLASS on the class total, then deterministic-shuffle before assigning
    # (shuffle decorrelates tier from pid/subkind so the X gate's subkind x condition pair is covered).
    trng = np.random.default_rng(seed + 777)
    by_co = {}
    for i, s in enumerate(stubs):
        by_co.setdefault(s[3], []).append(i)
    for co, idxs in by_co.items():
        tiers = deal_tiers(len(idxs))                          # marginal == WEATHER_MARGINAL (large N)
        trng.shuffle(tiers)
        for k, i in enumerate(idxs):
            stubs[i].append(tiers[k])
    plan = [tuple(s) for s in stubs]
    rng.shuffle(plan); plan.sort(key=lambda r: r[1])
    comp = (f"subj{sum(1 for r in plan if r[0]<1_000_000)}/"
            f"noth{sum(1 for r in plan if 1_000_000<=r[0]<2_000_000)}/"
            f"mixed{sum(1 for r in plan if r[0]>=2_000_000)}")
    return plan, len(subj) + len(noth) + len(mixed), comp


SCENE_COLS_V42 = ("scene_id,profile_id,family,terrain_vs,split,coarse,subkind,dur_s,fs,n_subjects,"
                  "closest_approach_m,speed_ms,target_snr_db,snr_clipped,noise_condition,"
                  "phase_coherence,payload_kg,het_jitter_db,masking,coupling_fc,"
                  "tier,gain_log10,quant_lsb,rail_mv,coupling_form,activity_json")   # v4.2 [R5] +tier +Q_j
WAVE_COLS_V42 = "scene_id,n_samples,noise_mv,clean_mv,clean_cls,clean_mv2,clean_cls2"


def init_db_v42(path):
    # v4.2 [R2]: EXACT-basename whitelist, NOT the substring `"v4" in path` (which also accepts
    # corpus_v4/v4b/v41 -> and this fn DELETES path/-wal/-shm below). Refuse every non-v42 corpus.
    base = os.path.basename(os.path.dirname(path))
    assert base == "corpus_v42" or base.startswith("corpus_v42_pilot"), \
        f"refusing non-v42 path (data-loss guard): {path}"
    for sfx in ("", "-wal", "-shm"):
        if os.path.exists(path + sfx):
            os.remove(path + sfx)
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(f"""
    CREATE TABLE scenes({SCENE_COLS_V42.replace('activity_json', 'activity_json TEXT')});
    CREATE TABLE waveforms(scene_id INTEGER PRIMARY KEY, n_samples INTEGER,
        noise_mv BLOB, clean_mv BLOB, clean_cls TEXT, clean_mv2 BLOB, clean_cls2 TEXT);
    """)
    return db


def open_or_init_v42(out):
    if os.path.exists(out):
        try:
            db = sqlite3.connect(out); committed = set(r[0] for r in db.execute("SELECT scene_id FROM scenes"))
            if committed:
                db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                return db, committed
            db.close()
        except Exception:
            pass
    return init_db_v42(out), set()


def run_shard_v42(args):
    plan, out, wi = args
    db, committed = open_or_init_v42(out)
    todo = [s for s in plan if s[0] not in committed]
    if committed:
        print(f"  v42-shard{wi}: RESUME {len(committed)} done, {len(todo)} to go", flush=True)
    t0 = time.time(); done = fail = 0; sbuf, vbuf = [], []
    ns = SCENE_COLS_V42.count(",") + 1; nv = WAVE_COLS_V42.count(",") + 1
    for spec in todo:
        try:
            sr, vr = gen_scene_v42(spec); sbuf.append(sr); vbuf.append(vr); done += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  v42-shard{wi} scene {spec[0]} FAIL {type(e).__name__}: {e}", flush=True)
        if len(sbuf) >= 200:
            db.executemany(f"INSERT INTO scenes({SCENE_COLS_V42}) VALUES ({','.join('?'*ns)})", sbuf)
            db.executemany(f"INSERT INTO waveforms({WAVE_COLS_V42}) VALUES ({','.join('?'*nv)})", vbuf)
            db.commit(); sbuf, vbuf = [], []
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"  v42-shard{wi}: {done} scenes, {(time.time()-t0)/60:.1f} min", flush=True)
    if sbuf:
        db.executemany(f"INSERT INTO scenes({SCENE_COLS_V42}) VALUES ({','.join('?'*ns)})", sbuf)
        db.executemany(f"INSERT INTO waveforms({WAVE_COLS_V42}) VALUES ({','.join('?'*nv)})", vbuf)
        db.commit()
    db.close()
    return wi, done, fail


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 171000
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DB_ROOT + r"\corpus_v42"
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    max_specs = int(sys.argv[4]) if len(sys.argv) > 4 else None
    # v4.2 [R2]: same exact-basename whitelist as init_db_v42 (belt-and-suspenders on the dir arg).
    _base = os.path.basename(os.path.normpath(out_dir))
    assert _base == "corpus_v42" or _base.startswith("corpus_v42_pilot"), \
        f"refusing non-v42 out_dir (data-loss guard): {out_dir}"
    os.makedirs(out_dir, exist_ok=True)
    plan, ncells, comp = build_plan_v42(n_target)
    if max_specs:
        step = max(1, len(plan) // max_specs); plan = plan[::step][:max_specs]
    # shard count fixed by nw (so resume matches on-disk shard files); GEO_V4_POOL caps the
    # number of CONCURRENT workers (leave CPU free) without repartitioning the shards.
    pool = int(os.environ.get("GEO_V4_POOL", nw))
    print(f"v42 plan: {len(plan)} scenes ({ncells} cells; {comp}), {nw} shards / {pool} workers -> {out_dir}", flush=True)
    shards = [plan[i::nw] for i in range(nw)]
    args = [(shards[i], os.path.join(out_dir, f"shard_{i}.sqlite"), i) for i in range(nw)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=pool) as ex:
        for wi, done, fail in ex.map(run_shard_v42, args):
            print(f"v42-shard {wi}: {done} scenes, {fail} failed", flush=True)
    dt = (time.time() - t0) / 60
    print(f"DONE {len(plan)} scenes in {dt:.1f} min -> {out_dir}", flush=True)
    if max_specs and dt > 0:
        rate = len(plan) / (dt * 60)
        print(f"  rate {rate:.1f} scenes/s; PROJECT {n_target} -> {n_target/rate/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
