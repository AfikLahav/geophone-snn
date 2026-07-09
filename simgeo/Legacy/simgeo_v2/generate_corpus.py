"""Production corpus generator — GENERATION_PLAN_V2 (~171k scenes). Adapts the v1 generator:
adds the tractor vehicle subtype, folds in multi-vehicle (convoy/two_vehicle), real
machinery/overflight content for the nothing subtypes, terrain-gated 50 Hz lines, cross-class
MIXED scenes, and 4-blob storage (pre-sensor ground velocity + post-sensor mV). 90/10 splits.
v2 noise (physics floor + spurious resonance + lines) lives in sensor.py; new sources in
sources_v2.py. NO ADC quantizer. v1 corpus (corpus_150k shards) is never touched.
Usage: python generate_corpus.py <n_scenes> <out_dir> <nworkers>
"""
import os, sys, json, sqlite3, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FS = 1000.0
BASE_SEED = 20260613                                       # v2 seed (distinct from v1's 20260612)

HUMAN = ["walk", "walk", "run", "stealth", "child", "group"]
VEHICLE = ["car", "car", "truck", "motorbike", "bicycle", "tractor", "convoy", "two_vehicle"]
ANIMAL = ["dog", "jackal", "boar", "horse", "sheep", "herd"]
NOTHING = ["calm", "wind", "wind", "rain", "machinery", "overflight"]
MIXED = ["human+vehicle", "human+animal", "vehicle+animal"]
R_DET = {"human": 60, "vehicle": 220, "animal": 50, "nothing": 0}
VEH_MIN_VS = 250
CONFUSER_W = 1.8                                           # nothing machinery/overflight boost
# terrain family -> P(50 Hz mains present) (data-anchored: zg-desert 33%, dev 88-98%)
LINE_TIER = {"asphalt": 0.90, "concrete": 0.90, "paving": 0.90,
             "dirt_road": 0.55, "kurkar": 0.55, "clay": 0.55, "loess": 0.55, "wet_soil": 0.55,
             "sand": 0.18, "gravel": 0.18, "soft_soil": 0.18, "rock": 0.18, "sabkha": 0.18,
             "frozen": 0.18, "snow": 0.18}


def build_plan(n_target, seed=BASE_SEED):
    """Compose subject (~48%) / nothing (~52%, confuser-weighted) / mixed (~10k) scene specs.
    Splits from the 90/10 mapping. Distinct sid ranges per category."""
    lib = json.load(open(os.path.join(HERE, "..", "..", "..", "terrain_models", "library.json")))
    splits = json.load(open(os.path.join(HERE, "..", "..", "..", "terrain_models", "splits_90_10.json")))
    banks = [p for p in lib if not p.get("modal")]
    rng = np.random.default_rng(seed)
    subj, noth, mixed = [], [], []
    for p in banks:
        pid = p["profile_id"]; split = splits.get(pid, "train"); veh_ok = p["vs_top_ms"] >= VEH_MIN_VS
        for sub in HUMAN:
            subj.append((pid, split, "human", sub))
        if veh_ok:
            for sub in VEHICLE:
                subj.append((pid, split, "vehicle", sub))
        for sub in ANIMAL:
            subj.append((pid, split, "animal", sub))
        for sub in NOTHING:
            noth.append((pid, split, "nothing", sub, CONFUSER_W if sub in ("machinery", "overflight") else 1.0))
        for combo in MIXED:
            if combo in ("human+vehicle", "vehicle+animal") and not veh_ok:
                continue
            mixed.append((pid, split, "mixed", combo))

    mixed_target = 10000
    rest = max(n_target - mixed_target, len(subj) + len(noth))
    noth_target = int(rest * 0.52); subj_target = rest - noth_target
    plan = []
    pc_subj = max(1, round(subj_target / len(subj)))
    sid = 0
    for (pid, split, coarse, sub) in subj:
        for _ in range(pc_subj):
            plan.append((sid, pid, split, coarse, sub)); sid += 1
    wsum = sum(w for *_, w in noth)
    sid = 1_000_000
    for (pid, split, coarse, sub, w) in noth:
        for _ in range(max(1, round(noth_target * w / wsum))):
            plan.append((sid, pid, split, coarse, sub)); sid += 1
    pc_mixed = max(1, round(mixed_target / len(mixed)))
    sid = 2_000_000
    for (pid, split, coarse, sub) in mixed:
        for _ in range(pc_mixed):
            plan.append((sid, pid, split, coarse, sub)); sid += 1
    rng.shuffle(plan)
    plan.sort(key=lambda r: r[1])                          # profile-sorted: each worker loads each bank once
    comp = f"subj{sum(1 for r in plan if r[0]<1_000_000)}/noth{sum(1 for r in plan if 1_000_000<=r[0]<2_000_000)}/mixed{sum(1 for r in plan if r[0]>=2_000_000)}"
    return plan, len(subj) + len(noth) + len(mixed), comp


def gen_scene(spec):
    sid, pid, split, coarse, sub = spec
    import sources_v2 as src, sensor, label, paths, r3_noise
    from scenes import Bank
    rng = np.random.default_rng((BASE_SEED * 100003 + sid) % 2**32)
    bank = Bank.get(pid); vs = bank.meta["vs_top_ms"]; fam = bank.meta["family"]; n_bank = bank.n
    fc, qc = bank.meta["coupling_fc"], bank.meta["coupling_q"]
    nw = int(label.WIN * FS); p_lines = LINE_TIER.get(fam, 0.30)
    counts = {"human": 0, "vehicle": 0, "animal": 0}
    ground = {"human": None, "vehicle": None, "animal": None}
    subj_info = []; speed = None; pinfo = None; mach_kind = None; over_kind = None

    def assemble(em, dur):
        n = int(dur * FS); v = np.zeros(n + n_bank)
        for (t0, x, y, fz, fx) in em:
            r = float(np.hypot(x, y))
            if r > 320: continue
            w = bank.emit(r, (1.0, 0.0), (x, y), fz, fx)
            i0 = int(t0 * FS); seg = min(len(w), len(v) - i0)
            if seg > 0: v[i0:i0 + seg] += w[:seg]
        return v[:n]

    def emit_class(c, csub):
        """Generate one class's emissions (no assemble). Returns (ems, dur, ns, infos, speed, pinfo)."""
        sp = (rng.uniform(0.2, 0.6) if csub == "stealth" else rng.uniform(2.5, 4) if csub == "run"
              else rng.uniform(1, 8) if csub == "tractor"
              else rng.uniform(4, 22) if c == "vehicle" else rng.uniform(0.6, 3.0))
        path, tg, d, pi = paths.sample(rng, R_DET[c], sp)
        ems = []; infos = []; ns = 1; dur = d
        if c == "human":
            ns = int(rng.choice([2, 3, 4])) if csub == "group" else 1
            g = "walk" if csub == "group" else csub
            for _ in range(ns):
                off = rng.uniform(0, 2, 2); em, info = src.human(path + off, tg, rng, n_bank, g)
                ems += em; infos.append({**info, "offset": [round(float(off[0]), 2), round(float(off[1]), 2)]})
        elif c == "vehicle":
            if csub == "convoy":
                ns = int(rng.choice([2, 2, 3])); kind = str(rng.choice(["car", "car", "truck", "motorbike"]))
                offs = np.cumsum([0.0] + [float(rng.uniform(2, 8)) for _ in range(ns - 1)])
                for j in range(ns):
                    em, info = src.vehicle(path, tg, rng, n_bank, kind)
                    ems += [(t + offs[j], x, y, fz, fx) for (t, x, y, fz, fx) in em]
                    infos.append({**info, "headway_s": round(float(offs[j]), 2)})
                dur = float(d + offs[-1])
            elif csub == "two_vehicle":
                ns = 2; kinds = [str(k) for k in rng.choice(["car", "car", "truck", "motorbike", "bicycle"], 2)]
                if rng.random() < 0.4: kinds[1] = kinds[0]
                dur = 0.0
                for j, k in enumerate(kinds):
                    sp2 = float(rng.uniform(4, 22)); pth, t2, d2, _ = paths.sample(rng, R_DET["vehicle"], sp2)
                    em, info = src.vehicle(pth, t2, rng, n_bank, k)
                    off = 0.0 if j == 0 else float(rng.uniform(0, 0.5 * d2))
                    ems += [(t + off, x, y, fz, fx) for (t, x, y, fz, fx) in em]
                    infos.append({**info, "offset_s": round(off, 2), "speed_ms": round(sp2, 2)}); dur = max(dur, d2 + off)
            else:
                em, info = src.vehicle(path, tg, rng, n_bank, csub); ems = em; infos.append(info)
        else:                                              # animal
            ns = int(rng.choice([3, 4, 6])) if csub == "herd" else int(rng.choice([1, 1, 2]))
            k = "boar" if csub in ("sheep", "herd") else csub
            k = k if k in ("dog", "jackal", "boar", "horse") else "boar"
            for _ in range(ns):
                off = rng.uniform(0, 2, 2); em, info = src.animal(path + off, tg, rng, n_bank, k)
                ems += em; infos.append({**info, "offset": [round(float(off[0]), 2), round(float(off[1]), 2)]})
        return ems, dur, ns, infos, sp, pi

    # ---------------- scene content ----------------
    if coarse == "nothing":
        dur = float(rng.uniform(20, 120)); n = int(dur * FS); v_sig = np.zeros(n)
        wind = {"calm": rng.uniform(0, 4), "wind": rng.uniform(5, 16), "rain": rng.uniform(0, 6),
                "machinery": rng.uniform(0, 6), "overflight": rng.uniform(0, 6)}[sub]
        rain = rng.uniform(2, 30) if sub == "rain" else (rng.uniform(0, 4) if rng.random() < 0.2 else 0)
        extra_ground = np.zeros(n)
        if sub == "machinery":
            mk = str(rng.choice(["pump", "pump", "generator"]))
            extra_ground, mi = (src.pump if mk == "pump" else src.generator_set)(n, rng)
            mach_kind = mi.get("kind"); subj_info.append(mi)
        elif sub == "overflight":
            extra_ground, oi = src.overflight(n, rng); over_kind = oi.get("subkind"); subj_info.append(oi)
    elif coarse == "mixed":
        c1, c2 = sub.split("+")
        e1, d1, n1, i1, s1, pi1 = emit_class(c1, "walk" if c1 == "human" else "car" if c1 == "vehicle" else "dog")
        e2, d2, n2, i2, s2, pi2 = emit_class(c2, "car" if c2 == "vehicle" else "dog")
        dur = float(max(d1, d2)); n = int(dur * FS)
        ground[c1] = assemble(e1, dur); ground[c2] = assemble(e2, dur)
        counts[c1] += n1; counts[c2] += n2
        subj_info += i1 + i2; speed = s1; pinfo = pi1
        v_sig = sum(ground[c] for c in (c1, c2)); wind = rng.uniform(0, 12)
        rain = rng.uniform(0, 4) if rng.random() < 0.15 else 0
    else:                                                  # single subject class
        ems, dur, ns, infos, speed, pinfo = emit_class(coarse, sub)
        ground[coarse] = assemble(ems, dur); counts[coarse] = ns; subj_info = infos
        v_sig = ground[coarse]; n = len(v_sig)
        wind = rng.uniform(0, 12); rain = rng.uniform(0, 4) if rng.random() < 0.15 else 0

    # ---------------- noise + render ----------------
    v_noise = r3_noise.ground_noise(n, wind, rain, rng) + sensor.rain_noise(n, rng, rain)
    if coarse == "nothing":
        v_noise = v_noise + extra_ground                   # machinery/overflight are ground noise
    out_mv, clean_mv, noise_mv, p = sensor.render_hp(v_sig, v_noise, rng, fc, qc, p_lines)
    # per-class clean_mv (reapply the SAME H from p so SNR is per-class; handles mixed)
    H = (sensor.coupling_response(n, p["fc"], p["qc"]) *
         sensor.geophone_response(n, p["f0"], p["h"], p["G"]) *
         sensor.spurious_bump(n, p["f_spur"], p["q_spur"], p["spur_db"]))
    cbc = {}
    for c in ("human", "vehicle", "animal"):
        if ground[c] is not None:
            cbc[c] = (np.fft.irfft(np.fft.rfft(ground[c]) * H, n=n) * p["gain"] * 1e3).astype(np.float32)
        else:
            cbc[c] = np.zeros(n, np.float32)

    win_rows = []
    for i0 in range(0, max(1, n - nw + 1), int(label.HOP * FS)):
        pc = label.per_class_window(cbc, noise_mv, counts, i0, nw)
        csnr = max((pc[c]["snr_db"] for c in pc), default=-99.0)
        act = int(any(pc[c]["level"] != "none" for c in pc))
        win_rows.append((sid, round(i0 / FS, 2),
                         pc["human"]["level"], pc["human"]["snr_db"], pc["human"]["soft"],
                         pc["vehicle"]["level"], pc["vehicle"]["snr_db"], pc["vehicle"]["soft"],
                         pc["animal"]["level"], pc["animal"]["snr_db"], pc["animal"]["soft"],
                         round(float(csnr), 2), label.zone(csnr), act))

    lines_present = int(p.get("lines_present", 0))
    cfg = dict(closest_approach_m=(round(float(pinfo["d"]), 1) if pinfo else None),
               speed_ms=(round(float(speed), 2) if speed is not None else None),
               path_style=(pinfo["style"] if pinfo else None),
               onset_free=(int(pinfo["onset_free"]) if pinfo else None),
               f0_hz=round(float(p["f0"]), 3), damping_h=round(float(p["h"]), 3),
               sens_g=round(float(p["G"]), 2), coupling_fc=round(float(p["fc"]), 2),
               coupling_q=round(float(p["qc"]), 3), gain=round(float(p["gain"]), 1),
               f_spur=round(float(p["f_spur"]), 1), p_lines=round(p_lines, 2), lines_present=lines_present,
               noise_condition=r3_noise.condition_of(wind, rain), machinery_kind=mach_kind,
               overflight_kind=over_kind, n_subjects=int(sum(counts.values())),
               primary_mass_kg=(round(float(np.mean([si.get("mass", 0) for si in subj_info if "mass" in si])), 1)
                                if any("mass" in si for si in subj_info) else None),
               wind=round(float(wind), 2), rain=round(float(rain), 2), counts=counts, subjects=subj_info)

    scene_row = (sid, pid, fam, float(vs), split, coarse, str(sub), round(dur, 1), FS, sid,
                 cfg["closest_approach_m"], cfg["speed_ms"], cfg["path_style"], cfg["onset_free"],
                 cfg["f0_hz"], cfg["damping_h"], cfg["sens_g"], cfg["coupling_fc"], cfg["coupling_q"],
                 cfg["gain"], cfg["f_spur"], cfg["noise_condition"], cfg["p_lines"], lines_present, mach_kind,
                 over_kind, cfg["n_subjects"], cfg["primary_mass_kg"], cfg["wind"], cfg["rain"],
                 json.dumps(cfg, default=float))
    vsig_blob = None if coarse == "nothing" else v_sig.astype(np.float32).tobytes()
    wave_row = (sid, n, vsig_blob, v_noise.astype(np.float32).tobytes(),
                clean_mv.astype(np.float32).tobytes(), noise_mv.astype(np.float32).tobytes())
    return scene_row, wave_row, win_rows


SCENE_COLS = ("scene_id,profile_id,family,terrain_vs,split,coarse,subkind,dur_s,fs,seed,"
              "closest_approach_m,speed_ms,path_style,onset_free,f0_hz,damping_h,sens_g,coupling_fc,"
              "coupling_q,gain,f_spur,noise_condition,p_lines,lines_present,machinery_kind,overflight_kind,"
              "n_subjects,primary_mass_kg,wind,rain,config_json")
WAVE_COLS = "scene_id,n_samples,v_signal,v_noise_ground,clean_mv,noise_mv"
WIN_COLS = ("scene_id,t0,human_level,human_snr,human_soft,vehicle_level,vehicle_snr,vehicle_soft,"
            "animal_level,animal_snr,animal_soft,common_snr,zone,activity")


def init_db(path):
    assert "v2" in path, f"refusing non-v2 output path (v1 guard): {path}"
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(f"""
    CREATE TABLE scenes({SCENE_COLS.replace('config_json','config_json TEXT')});
    CREATE TABLE waveforms(scene_id INTEGER PRIMARY KEY, n_samples INTEGER,
        v_signal BLOB, v_noise_ground BLOB, clean_mv BLOB, noise_mv BLOB);
    CREATE TABLE windows(window_id INTEGER PRIMARY KEY AUTOINCREMENT, {WIN_COLS});
    """)
    return db


def open_or_init(out):
    """Resume-aware: if the shard already has committed scenes, open it (WAL-recover) and return
    (db, committed_sids); else init fresh. Lets a crashed run continue without re-rendering."""
    if os.path.exists(out):
        try:
            db = sqlite3.connect(out)
            committed = set(r[0] for r in db.execute("SELECT scene_id FROM scenes"))
            if committed:
                db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                return db, committed
            db.close()
        except Exception:
            pass
    return init_db(out), set()


def run_shard(args):
    plan, out, wi = args
    db, committed = open_or_init(out)
    todo = [s for s in plan if s[0] not in committed]
    if committed:
        print(f"  v2-shard{wi}: RESUME — {len(committed)} committed, {len(todo)} to go", flush=True)
    t0 = time.time(); done = 0; fail = 0
    sbuf, vbuf, wbuf = [], [], []
    ns, nv, nwc = SCENE_COLS.count(",") + 1, WAVE_COLS.count(",") + 1, WIN_COLS.count(",") + 1
    for spec in todo:
        try:
            sr, vr, wr = gen_scene(spec)
            sbuf.append(sr); vbuf.append(vr); wbuf.extend(wr); done += 1
        except Exception as e:
            fail += 1
            if fail <= 5: print(f"  v2-shard{wi} scene {spec[0]} FAIL {type(e).__name__}: {e}", flush=True)
        if len(sbuf) >= 300:
            db.executemany(f"INSERT INTO scenes({SCENE_COLS}) VALUES ({','.join('?'*ns)})", sbuf)
            db.executemany(f"INSERT INTO waveforms({WAVE_COLS}) VALUES ({','.join('?'*nv)})", vbuf)
            db.executemany(f"INSERT INTO windows({WIN_COLS}) VALUES ({','.join('?'*nwc)})", wbuf)
            db.commit(); sbuf, vbuf, wbuf = [], [], []
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"  v2-shard{wi}: {done} scenes, {(time.time()-t0)/60:.1f} min", flush=True)
    if sbuf:
        db.executemany(f"INSERT INTO scenes({SCENE_COLS}) VALUES ({','.join('?'*ns)})", sbuf)
        db.executemany(f"INSERT INTO waveforms({WAVE_COLS}) VALUES ({','.join('?'*nv)})", vbuf)
        db.executemany(f"INSERT INTO windows({WIN_COLS}) VALUES ({','.join('?'*nwc)})", wbuf)
        db.commit()
    db.close()
    return wi, done, fail


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 171000
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"G:\geophone_synth\corpus_v2"
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    assert "v2" in out_dir, "refusing non-v2 out_dir (v1 guard)"
    os.makedirs(out_dir, exist_ok=True)
    plan, ncells, comp = build_plan(n_target)
    print(f"v2 plan: {len(plan)} scenes ({ncells} cells; {comp}), {nw} workers -> {out_dir}", flush=True)
    shards = [plan[i::nw] for i in range(nw)]
    args = [(shards[i], os.path.join(out_dir, f"shard_{i}.sqlite"), i) for i in range(nw)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for wi, done, fail in ex.map(run_shard, args):
            print(f"v2-shard {wi} done: {done} scenes, {fail} failed", flush=True)
    print(f"ALL V2 SHARDS DONE in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
