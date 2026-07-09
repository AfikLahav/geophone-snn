"""Dataset v0 generator (PLAN Phase 5, pilot form).

Stratified: per profile x class, n_per cell scenes. Profile-grouped splits assigned
HERE (held-out profiles per family -> val/test) so leakage discipline exists from
sample one. Output: N:\\geophone_synth\\v0\\scenes\\*.npz + manifest.csv + windows.csv.
Usage: python generate.py [n_per_cell] [out_root]
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sources, sensor, scenes
from scenes import Bank, FS
from profiles import FLOORS

OUT = r"N:\geophone_synth\v0"
BASE_SEED = 20260611

CLASSES = [
    ("human", "walk"), ("human", "run"), ("human", "group"),
    ("vehicle", "car"), ("vehicle", "truck"), ("vehicle", "motorbike"),
    ("animal", "dog"), ("animal", "horse"), ("animal", "boar"),
    ("nothing", "quiet"), ("nothing", "wind"), ("nothing", "rain"),
]
# vehicles only on plausible terrain (spec §2, coarse v0 rule)
VEHICLE_OK = {"gravel", "dirt_road", "asphalt", "concrete", "kurkar", "rock"}
R_DETECT = {"human": 60.0, "animal": 50.0, "vehicle": 220.0, "nothing": 0.0}


def detect_range(coarse, prof):
    base = R_DETECT[coarse]
    soft = prof["vs_top_ms"] < 350
    return base * (0.8 if soft else 1.2)


def make_scene(prof, coarse, sub, rng, is_floor=False, floor_cfg=None):
    if is_floor:
        n_ir = 4000
        bank_or_ir = scenes.modal_floor_ir(floor_cfg["f0"], floor_cfg["zeta"],
                                           floor_cfg["nmodes"], n_ir, rng)
        fc, qc = floor_cfg["f0"] * 5, 8.0
        vr = 1000.0
    else:
        bank_or_ir = Bank.get(prof["profile_id"])
        fc, qc = prof["coupling_fc"], prof["coupling_q"]

    weather = dict(wind_ms=float(np.clip(rng.gamma(2, 2), 0, 18)),
                   rain_mmh=float(rng.gamma(0.5, 4)) if rng.random() < 0.25 else 0.0)
    params = dict(weather=weather)

    if coarse == "nothing":
        dur = rng.uniform(15, 45)
        if sub == "wind":
            weather["wind_ms"] = rng.uniform(5, 18)
        if sub == "rain":
            weather["rain_mmh"] = rng.uniform(2, 30)
        v_sig = np.zeros(int(dur * FS))
    else:
        kind = {"walk": "walk", "run": "run", "group": "walk"}.get(sub, None)
        pkind = ("vehicle" if coarse == "vehicle" else
                 "animal" if coarse == "animal" else
                 "run" if sub == "run" else "walk")
        rdet = detect_range(coarse, prof if not is_floor else dict(vs_top_ms=2000))
        path, tgrid, dur, pinfo = scenes.sample_path(rng, pkind, rdet)
        params.update(pinfo)
        if coarse == "human":
            nsrc = rng.integers(2, 5) if sub == "group" else 1
            ems = []
            for s in range(nsrc):
                off = rng.uniform(0, 2, 2)
                em, sp = sources.human(path + off, tgrid, rng, kind)
                ems.append(em)
            params.update(sp, n_sources=int(nsrc))
        elif coarse == "vehicle":
            em, sp = sources.vehicle(path, tgrid, rng, sub)
            ems = [em]; params.update(sp)
        else:
            em, sp = sources.animal(path, tgrid, rng, sub)
            ems = [em]; params.update(sp)
        v_sig = scenes.assemble(ems, bank_or_ir, dur, rng, is_floor=is_floor)

    n = len(v_sig)
    v_noise = (sensor.colored_noise(n, rng, rng.uniform(-2, 0)) *
               10 ** rng.uniform(-9.0, -7.5))            # ambient site level (m/s RMS)
    v_noise += sensor.wind_noise(n, rng, weather["wind_ms"])
    v_noise += sensor.rain_noise(n, rng, weather["rain_mmh"])
    out_mv, clean_mv, sp = sensor.render(v_sig, v_noise, rng, fc, qc)
    params["sensor"] = {k: round(float(v), 4) for k, v in sp.items()}
    wins = scenes.window_labels(clean_mv, out_mv, coarse)
    return out_mv, wins, dur, params


def main(n_per_cell=3, out_root=OUT):
    t0 = time.time()
    os.makedirs(os.path.join(out_root, "scenes"), exist_ok=True)
    lib = json.load(open(os.path.join(HERE, "..", "simgeo_banks", "library.json")))
    # profile-grouped split: last profile of each family -> val, 2nd-last -> test
    fam_members = {}
    for p in lib:
        fam_members.setdefault(p["family"], []).append(p["profile_id"])
    split = {}
    for fam, ids in fam_members.items():
        for pid in ids:
            split[pid] = "train"
        if len(ids) >= 3:
            split[ids[-1]] = "val"; split[ids[-2]] = "test"
    for i, fl in enumerate(FLOORS):
        split[fl["profile_id"]] = "train" if i < 2 else ("val" if i == 2 else "test")

    man, wrows = [], []
    sid = 0
    cells = [(p, c, s, False, None) for p in lib for (c, s) in CLASSES
             if not (c == "vehicle" and p["family"] not in VEHICLE_OK)]
    cells += [(None, c, s, True, fl) for fl in FLOORS
              for (c, s) in CLASSES if c in ("human", "nothing")]
    print(f"{len(cells)} cells x {n_per_cell} scenes", flush=True)
    for (prof, coarse, sub, is_floor, fl) in cells:
        for k in range(n_per_cell):
            rng = np.random.default_rng(hash((BASE_SEED, sid)) % 2**32)
            try:
                amp, wins, dur, params = make_scene(prof, coarse, sub, rng,
                                                    is_floor, fl)
            except Exception as e:
                print(f"  scene {sid} FAIL {type(e).__name__}: {e}", flush=True)
                sid += 1
                continue
            pid = fl["profile_id"] if is_floor else prof["profile_id"]
            scene_id = f"s{sid:06d}"
            np.savez_compressed(os.path.join(out_root, "scenes", scene_id + ".npz"),
                                amplitude=amp, fs=FS)
            man.append(dict(scene_id=scene_id, profile_id=pid,
                            family=fl["family"] if is_floor else prof["family"],
                            split=split[pid], coarse=coarse, sub=sub,
                            dur_s=round(dur, 1), seed=sid,
                            params=json.dumps(params, default=float)))
            for (t, lab, snr) in wins:
                wrows.append(dict(scene_id=scene_id, t0=t, label=lab, snr_db=snr,
                                  split=split[pid]))
            sid += 1
        if sid % 120 < n_per_cell:
            print(f"  {sid} scenes, {(time.time()-t0)/60:.1f} min", flush=True)
    pd.DataFrame(man).to_csv(os.path.join(out_root, "manifest.csv"), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(out_root, "windows.csv"), index=False)
    print(f"DONE: {len(man)} scenes, {len(wrows)} windows, "
          f"{(time.time()-t0)/60:.1f} min -> {out_root}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3,
         sys.argv[2] if len(sys.argv) > 2 else OUT)
