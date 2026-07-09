"""Vertical-slice generator — GENERATION_PLAN v1.2.
A few hundred scenes across 2-3 terrains x {human, vehicle, animal, nothing}, high-precision
geophone output (NO quantizer), three-zone analog-floored per-class ordinal labels, written to
SQLite. Proves the corrected pipeline end-to-end before the full corpus.
Usage: python slice_generate.py [n_per_cell] [out_db]
"""
import os, sys, json, sqlite3, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes import Bank, FS
import sources_v2 as src, sensor, label

OUT = r"N:\geophone_synth\slice_v0.sqlite"
TERRAINS = ["soft_soil_0", "soft_soil_3", "gravel_0", "rock_0", "asphalt_0"]
BASE_SEED = 20260612
CLASSES = ["human", "vehicle", "animal", "nothing"]


def sample_path(rng, r_det, fast=False):
    speed = rng.uniform(3, 18) if fast else rng.uniform(0.8, 2.0)
    d = rng.uniform(2, 0.7 * r_det)
    dur = float(np.clip(2 * r_det / speed, 8, 60))
    t = np.arange(0, dur, 0.1)
    x = speed * (t - dur / 2)
    y = np.full_like(t, d) + rng.uniform(0, 0.2) * d * np.sin(2 * np.pi * t / rng.uniform(8, 20))
    return np.stack([x, y], 1), t, dur


def assemble(emissions_list, bank, dur, rng):
    n = int(dur * FS); v = np.zeros(n + bank.n)
    for em in emissions_list:
        for (t0, x, y, fz, fx) in em:
            r = float(np.hypot(x, y))
            if r > 320:
                continue
            w = bank.emit(r, (1.0, 0.0), (x, y), fz, fx)
            i0 = int(t0 * FS); seg = min(len(w), len(v) - i0)
            if seg > 0:
                v[i0:i0 + seg] += w[:seg]
    return v[:n]


def init_db(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript("""
    CREATE TABLE scenes(scene_id INTEGER PRIMARY KEY, profile_id TEXT, family TEXT, terrain_vs REAL,
      coarse TEXT, subkind TEXT, dur_s REAL, gain REAL, params TEXT, amplitude BLOB, fs REAL, seed INTEGER);
    CREATE TABLE windows(window_id INTEGER PRIMARY KEY, scene_id INTEGER, t0 REAL,
      human_level TEXT, human_snr REAL, human_soft REAL,
      vehicle_level TEXT, vehicle_snr REAL, vehicle_soft REAL,
      animal_level TEXT, animal_snr REAL, animal_soft REAL,
      common_snr REAL, zone TEXT, activity INTEGER);
    CREATE INDEX ix_win_scene ON windows(scene_id);
    CREATE INDEX ix_win_zone ON windows(zone);
    """)
    return db


def main(n_per_cell=8, out=OUT):
    t0 = time.time()
    db = init_db(out)
    sid = wid = 0
    nw = int(label.WIN * FS)
    R_DET = {"human": 60, "vehicle": 220, "animal": 50, "nothing": 0}
    rows_s, rows_w = [], []
    for pid in TERRAINS:
        bank = Bank.get(pid); vs = bank.meta["vs_top_ms"]
        for coarse in CLASSES:
            for k in range(n_per_cell):
                rng = np.random.default_rng((BASE_SEED * 100003 + sid) % 2**32)
                # vehicles only on firm/paved terrain (compatibility, coarse slice rule)
                if coarse == "vehicle" and vs < 250:
                    sid += 1; continue
                counts = {"human": 0, "vehicle": 0, "animal": 0}
                emissions = []; subkind = coarse
                if coarse == "nothing":
                    dur = rng.uniform(15, 40); v_sig = np.zeros(int(dur * FS))
                    weather = dict(wind=float(np.clip(rng.gamma(2, 2), 0, 16)),
                                   rain=float(rng.gamma(0.5, 4)) if rng.random() < 0.3 else 0.0)
                else:
                    fast = coarse == "vehicle"
                    path, tg, dur = sample_path(rng, R_DET[coarse], fast)
                    if coarse == "human":
                        nsrc = rng.choice([1, 1, 1, 2, 3])
                        for s in range(nsrc):
                            em, info = src.human(path + rng.uniform(0, 2, 2), tg, rng, bank.n,
                                                 rng.choice(["walk", "walk", "run"]))
                            emissions.append(em)
                        counts["human"] = int(nsrc); subkind = info["subkind"]
                    elif coarse == "vehicle":
                        em, info = src.vehicle(path, tg, rng, bank.n, rng.choice(["car", "truck"]))
                        emissions.append(em); counts["vehicle"] = 1; subkind = info["subkind"]
                    else:
                        nsrc = rng.choice([1, 1, 2, 3])
                        kind = rng.choice(["dog", "jackal", "boar", "horse"])
                        for s in range(nsrc):
                            em, info = src.animal(path + rng.uniform(0, 2, 2), tg, rng, bank.n, kind)
                            emissions.append(em)
                        counts["animal"] = int(nsrc); subkind = kind
                    weather = dict(wind=float(np.clip(rng.gamma(1.5, 2), 0, 14)),
                                   rain=float(rng.gamma(0.5, 3)) if rng.random() < 0.2 else 0.0)
                    v_sig = assemble(emissions, bank, dur, rng)
                n = len(v_sig)
                v_noise = (sensor.colored_noise(n, rng, rng.uniform(-2, 0)) *
                           10 ** rng.uniform(-9, -7.5))
                v_noise += sensor.wind_noise(n, rng, weather["wind"])
                v_noise += sensor.rain_noise(n, rng, weather["rain"])
                # per-class clean signals (for per-class SNR) — render each class separately
                clean_by_class = {}
                # render full signal + noise at high precision
                out_mv, clean_mv, noise_mv, p = sensor.render_hp(
                    v_sig, v_noise, rng, bank.meta["coupling_fc"], bank.meta["coupling_q"])
                # for the slice, single-coarse scenes: that class's clean = clean_mv, others 0
                for c in ("human", "vehicle", "animal"):
                    clean_by_class[c] = clean_mv if counts.get(c, 0) > 0 else np.zeros(n, np.float32)
                amp = out_mv
                rows_s.append((sid, pid, bank.meta["family"], float(vs), coarse, str(subkind),
                               round(dur, 1), round(float(p["gain"]), 1),
                               json.dumps(dict(weather=weather, counts=counts), default=float),
                               amp.tobytes(), FS, sid))
                # windows
                for i0 in range(0, max(1, n - nw + 1), int(label.HOP * FS)):
                    pc = label.per_class_window(clean_by_class, noise_mv, counts, i0, nw)
                    csnr = max((pc[c]["snr_db"] for c in pc), default=-99)
                    act = int(any(pc[c]["level"] != "none" for c in pc))
                    rows_w.append((wid, sid, round(i0 / FS, 2),
                                   pc["human"]["level"], pc["human"]["snr_db"], pc["human"]["soft"],
                                   pc["vehicle"]["level"], pc["vehicle"]["snr_db"], pc["vehicle"]["soft"],
                                   pc["animal"]["level"], pc["animal"]["snr_db"], pc["animal"]["soft"],
                                   round(csnr, 2), label.zone(csnr), act))
                    wid += 1
                sid += 1
        print(f"  {pid}: {sid} scenes, {wid} windows, {(time.time()-t0)/60:.1f} min", flush=True)
    db.executemany("INSERT INTO scenes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows_s)
    db.executemany("INSERT INTO windows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows_w)
    db.commit()
    # quick summary
    print(f"\nDONE: {len(rows_s)} scenes, {len(rows_w)} windows in {(time.time()-t0)/60:.1f} min -> {out}")
    cur = db.execute("SELECT coarse, COUNT(*) FROM scenes GROUP BY coarse")
    print("scenes/class:", dict(cur.fetchall()))
    cur = db.execute("SELECT zone, COUNT(*) FROM windows GROUP BY zone")
    print("windows/zone:", dict(cur.fetchall()))
    for c in ("human", "vehicle", "animal"):
        cur = db.execute(f"SELECT {c}_level, COUNT(*) FROM windows GROUP BY {c}_level")
        print(f"{c} levels:", dict(cur.fetchall()))
    db.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
         sys.argv[2] if len(sys.argv) > 2 else OUT)
