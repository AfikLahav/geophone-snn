"""Append multi-vehicle scenes to corpus_150k (gap found 2026-06-13: vehicle head had
zero 'multiple' examples). Two subtypes on vehicle-capable terrain (vs>=250):
  convoy      — 2-3 vehicles, SAME kind, same path, 2-8 s headway (column of cars/bikes)
  two_vehicle — 2 independent vehicles (40% same kind: two cars, two motorbikes...),
                independent paths/speeds, second offset 0-50% into the scene
Same chain as generate_corpus.gen_scene: bank GF -> R3 noise -> render_hp -> three-zone
per-class ordinal labels. counts['vehicle']=ns so detectable windows label 'multiple'.
Writes NEW shards (shard_8+) next to the existing 0-7 — the 148k is untouched.
Scene IDs start at 200000 (no collision; seeds derive from sid).
Splits from simgeo_banks/splits_90_10.json (the 90/10 re-split).
Usage: python append_multiveh.py [per_cell=8] [nworkers=6] [out_dir=N:\\geophone_synth\\corpus_150k]
"""
import os, sys, json, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate_corpus import SCENE_COLS, WIN_COLS, init_db, BASE_SEED, FS, VEH_MIN_VS

SID0 = 200000
SUBS = ["convoy", "two_vehicle"]
R_DET_VEH = 220


def build_plan(per_cell):
    lib = json.load(open(os.path.join(HERE, "..", "simgeo_banks", "library.json")))
    splits = json.load(open(os.path.join(HERE, "..", "simgeo_banks", "splits_90_10.json")))
    banks = [p for p in lib if not p.get("modal") and p["vs_top_ms"] >= VEH_MIN_VS]
    plan = []
    sid = SID0
    for p in banks:
        for sub in SUBS:
            for _ in range(per_cell):
                plan.append((sid, p["profile_id"], splits[p["profile_id"]], sub)); sid += 1
    rng = np.random.default_rng(BASE_SEED + 1)
    rng.shuffle(plan)
    plan.sort(key=lambda r: r[1])          # profile-sorted: each worker loads each bank once
    return plan, len(banks)


def gen_scene_mv(spec):
    sid, pid, split, sub = spec
    import sources_v2 as src, sensor, label, paths, r3_noise
    from scenes import Bank
    rng = np.random.default_rng((BASE_SEED * 100003 + sid) % 2**32)
    bank = Bank.get(pid); vs = bank.meta["vs_top_ms"]; n_bank = bank.n
    nw = int(label.WIN * FS)
    subj_info = []                                   # config capture (rng-neutral)

    if sub == "convoy":
        ns = int(rng.choice([2, 2, 3]))
        kind = str(rng.choice(["car", "car", "truck", "motorbike"]))
        kinds = [kind] * ns
        speed = float(rng.uniform(4, 22))
        path, tg, dur, pinfo = paths.sample(rng, R_DET_VEH, speed)
        offs = np.cumsum([0.0] + [float(rng.uniform(2, 8)) for _ in range(ns - 1)])
        ems = []
        for j in range(ns):
            em, info = src.vehicle(path, tg, rng, n_bank, kind)
            ems += [(t + offs[j], x, y, fz, fx) for (t, x, y, fz, fx) in em]
            subj_info.append({**info, "headway_s": round(float(offs[j]), 2)})
        dur = float(dur + offs[-1])
    else:                                   # two_vehicle: independent paths/speeds/kinds
        ns = 2
        kinds = [str(k) for k in rng.choice(["car", "car", "truck", "motorbike", "bicycle"], 2)]
        if rng.random() < 0.4:
            kinds[1] = kinds[0]             # two cars / two motorbikes etc.
        ems = []; dur = 0.0
        for j, k in enumerate(kinds):
            speed = float(rng.uniform(4, 22))
            path, tg, d, pinfo = paths.sample(rng, R_DET_VEH, speed)
            em, info = src.vehicle(path, tg, rng, n_bank, k)
            off = 0.0 if j == 0 else float(rng.uniform(0, 0.5 * d))
            ems += [(t + off, x, y, fz, fx) for (t, x, y, fz, fx) in em]
            subj_info.append({**info, "offset_s": round(float(off), 2), "speed_ms": round(speed, 2)})
            dur = max(dur, d + off)

    counts = {"human": 0, "vehicle": ns, "animal": 0}
    n = int(dur * FS)
    v = np.zeros(n + n_bank)                # assemble (same as generate_corpus)
    for (t0, x, y, fz, fx) in ems:
        r = float(np.hypot(x, y))
        if r > 320: continue
        w = bank.emit(r, (1.0, 0.0), (x, y), fz, fx)
        i0 = int(t0 * FS); seg = min(len(w), len(v) - i0)
        if seg > 0: v[i0:i0 + seg] += w[:seg]
    v_sig = v[:n]
    wind = rng.uniform(0, 12); rain = rng.uniform(0, 4) if rng.random() < 0.15 else 0
    v_noise = r3_noise.ground_noise(n, wind, rain, rng)
    v_noise += sensor.rain_noise(n, rng, rain)
    out_mv, clean_mv, noise_mv, p = sensor.render_hp(
        v_sig, v_noise, rng, bank.meta["coupling_fc"], bank.meta["coupling_q"])
    cbc = {"human": np.zeros(n, np.float32), "vehicle": clean_mv,
           "animal": np.zeros(n, np.float32)}

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
    scene_row = (sid, pid, bank.meta["family"], float(vs), split, "vehicle", str(sub),
                 round(dur, 1), round(float(p["gain"]), 1),
                 json.dumps(dict(counts=counts, kinds=kinds, wind=round(wind, 1),
                                 rain=round(rain, 1)), default=float),
                 out_mv.astype(np.float32).tobytes(), FS, sid)
    cfg = dict(
        closest_approach_m=round(float(pinfo["d"]), 1),
        speed_ms=round(float(speed), 2), path_style=pinfo["style"],
        onset_free=int(pinfo["onset_free"]),
        f0_hz=round(float(p["f0"]), 3), damping_h=round(float(p["h"]), 3),
        sens_g=round(float(p["G"]), 2), coupling_fc=round(float(p["fc"]), 2),
        coupling_q=round(float(p["qc"]), 3), gain=round(float(p["gain"]), 1),
        noise_condition=r3_noise.condition_of(wind, rain), n_subjects=int(ns),
        primary_mass_kg=round(float(np.mean([si.get("mass", 0) for si in subj_info])), 1),
        wind=round(float(wind), 2), rain=round(float(rain), 2), subjects=subj_info)
    return scene_row, win_rows, cfg


def run_shard(args):
    plan, out, wi = args
    import sqlite3
    db = init_db(out)
    t0 = time.time(); done = 0; fail = 0
    sbuf, wbuf = [], []
    for spec in plan:
        try:
            sr, wr, _ = gen_scene_mv(spec)
            sbuf.append(sr); wbuf.extend(wr); done += 1
        except Exception as e:
            fail += 1
            if fail <= 5: print(f"  mv-shard{wi} scene {spec[0]} FAIL {type(e).__name__}: {e}", flush=True)
        if len(sbuf) >= 100:
            db.executemany(f"INSERT INTO scenes VALUES ({','.join('?'*13)})", sbuf)
            db.executemany(f"INSERT INTO windows({WIN_COLS}) VALUES ({','.join('?'*14)})", wbuf)
            db.commit(); sbuf, wbuf = [], []
            print(f"  mv-shard{wi}: {done} scenes, {(time.time()-t0)/60:.1f} min", flush=True)
    if sbuf:
        db.executemany(f"INSERT INTO scenes VALUES ({','.join('?'*13)})", sbuf)
        db.executemany(f"INSERT INTO windows({WIN_COLS}) VALUES ({','.join('?'*14)})", wbuf)
        db.commit()
    db.close()
    return wi, done, fail


def main():
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out_dir = sys.argv[3] if len(sys.argv) > 3 else r"N:\geophone_synth\corpus_150k"
    plan, nprof = build_plan(per_cell)
    print(f"multi-vehicle append: {len(plan)} scenes ({nprof} profiles x {len(SUBS)} subs "
          f"x {per_cell}/cell), {nw} workers -> shards 8..{7+nw}", flush=True)
    shards = [plan[i::nw] for i in range(nw)]
    args = [(shards[i], os.path.join(out_dir, f"shard_{8+i}.sqlite"), i) for i in range(nw)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for wi, done, fail in ex.map(run_shard, args):
            print(f"mv-shard {wi} done: {done} scenes, {fail} failed", flush=True)
    print(f"APPEND DONE in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
