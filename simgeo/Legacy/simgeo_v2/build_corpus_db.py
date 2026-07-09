"""Phase 2 of the DB refactor: merge the 14 shards + per-shard config DBs into ONE
analyzable database, G:\\geophone_synth\\corpus.sqlite.

NON-DESTRUCTIVE: the 14 shards and config DBs are only READ (ATTACH). A fresh
corpus.sqlite is written; nothing is moved or deleted. Schema separates the heavy
waveform blob from light metadata so config/label queries never touch the 38 GB:

  scenes    : metadata + recovered config columns + config_json   (NO blob -> fast queries)
  waveforms : scene_id + amplitude blob                            (the 38 GB, read on demand)
  windows   : per-window labels                                    (as before)

split column is taken from terrain_models/splits_90_10.json (the 90/10 re-split), fixing
the stale 70/15/15 still sitting in the shard split columns. Scenes whose config wasn't
recovered (bit-mismatch) keep NULL config but are NOT dropped (LEFT JOIN) and are reported.
Usage: python build_corpus_db.py [corpus_dir] [config_dir] [out_db]
"""
import os, sys, glob, json, sqlite3, time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = r"G:\geophone_synth\corpus_150k"
CONFIG = r"G:\geophone_synth\config"
OUT = r"G:\geophone_synth\corpus.sqlite"
CFG_COLS = ["closest_approach_m", "speed_ms", "path_style", "onset_free", "f0_hz",
            "damping_h", "sens_g", "coupling_fc", "coupling_q", "noise_condition",
            "n_subjects", "primary_mass_kg", "wind", "rain", "config_json"]
WIN_COLS = ("scene_id,t0,human_level,human_snr,human_soft,vehicle_level,vehicle_snr,"
            "vehicle_soft,animal_level,animal_snr,animal_soft,common_snr,zone,activity")


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else CORPUS
    cfgdir = sys.argv[2] if len(sys.argv) > 2 else CONFIG
    out = sys.argv[3] if len(sys.argv) > 3 else OUT
    for suf in ("", "-wal", "-shm"):                       # fresh output only; shards untouched
        if os.path.exists(out + suf):
            os.remove(out + suf)
    splits = json.load(open(os.path.join(HERE, "..", "..", "..", "terrain_models", "splits_90_10.json")))

    db = sqlite3.connect(out)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript("""
    CREATE TABLE scenes(
      scene_id INTEGER PRIMARY KEY, profile_id TEXT, family TEXT, terrain_vs REAL,
      split TEXT, coarse TEXT, subkind TEXT, dur_s REAL, gain REAL, fs REAL, seed INTEGER,
      closest_approach_m REAL, speed_ms REAL, path_style TEXT, onset_free INTEGER,
      f0_hz REAL, damping_h REAL, sens_g REAL, coupling_fc REAL, coupling_q REAL,
      noise_condition TEXT, n_subjects INTEGER, primary_mass_kg REAL, wind REAL, rain REAL,
      config_json TEXT);
    CREATE TABLE waveforms(scene_id INTEGER PRIMARY KEY, fs REAL, n_samples INTEGER, amplitude BLOB);
    CREATE TABLE windows(window_id INTEGER PRIMARY KEY AUTOINCREMENT, """ + WIN_COLS.replace(
        "scene_id", "scene_id INTEGER") + """);
    CREATE TEMP TABLE profsplit(profile_id TEXT PRIMARY KEY, split TEXT);
    """)
    db.executemany("INSERT INTO profsplit VALUES (?,?)", list(splits.items()))
    db.commit()

    shards = sorted(glob.glob(os.path.join(corpus, "shard_*.sqlite")),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))
    t0 = time.time(); tot_s = tot_w = nocfg = 0
    sel_cfg = ",".join("c." + c for c in CFG_COLS)
    for sh in shards:
        cfgp = os.path.join(cfgdir, os.path.basename(sh).replace("shard_", "config_shard_"))
        db.execute(f"ATTACH '{sh}' AS s"); db.execute(f"ATTACH '{cfgp}' AS c")
        # scenes: shard metadata + 90/10 split + recovered config (LEFT JOIN keeps all scenes)
        db.execute(f"""INSERT INTO scenes
            SELECT s.scene_id, s.profile_id, s.family, s.terrain_vs,
                   COALESCE(ps.split, s.split), s.coarse, s.subkind, s.dur_s, s.gain, s.fs, s.seed,
                   {sel_cfg}
            FROM s.scenes s
            LEFT JOIN c.config c ON s.scene_id=c.scene_id
            LEFT JOIN profsplit ps ON s.profile_id=ps.profile_id""")
        # waveforms: the heavy blob, separated out
        db.execute("INSERT INTO waveforms SELECT scene_id, fs, length(amplitude)/4, amplitude "
                   "FROM s.scenes")
        # windows: labels (drop per-shard window_id; corpus autoincrements its own)
        db.execute(f"INSERT INTO windows({WIN_COLS}) SELECT {WIN_COLS} FROM s.windows")
        db.commit()
        ns = db.execute("SELECT changes()").fetchone()[0]
        nc = db.execute(f"SELECT COUNT(*) FROM s.scenes WHERE scene_id NOT IN "
                        f"(SELECT scene_id FROM c.config)").fetchone()[0]
        tot_s = db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        nocfg += nc
        db.execute("DETACH s"); db.execute("DETACH c")
        print(f"  {os.path.basename(sh)}: merged (scenes so far {tot_s}, {nc} without config), "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

    print("building indexes ...", flush=True)
    db.executescript("""
    CREATE INDEX ix_sc_split ON scenes(split);
    CREATE INDEX ix_sc_coarse ON scenes(coarse);
    CREATE INDEX ix_sc_family ON scenes(family);
    CREATE INDEX ix_sc_style ON scenes(path_style);
    CREATE INDEX ix_sc_dist ON scenes(closest_approach_m);
    CREATE INDEX ix_sc_cond ON scenes(noise_condition);
    CREATE INDEX ix_w_scene ON windows(scene_id);
    CREATE INDEX ix_w_zone ON windows(zone);
    """)
    db.commit()
    tot_w = db.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    nwav = db.execute("SELECT COUNT(*) FROM waveforms").fetchone()[0]
    db.close()
    sz = os.path.getsize(out) / 1e9
    print(f"BUILT {out} ({sz:.1f} GB): {tot_s} scenes, {nwav} waveforms, {tot_w} windows, "
          f"{nocfg} scenes without config, {(time.time()-t0)/60:.1f} min", flush=True)
    print("(the 14 shards + config DBs are unchanged — kept as backup)", flush=True)


if __name__ == "__main__":
    main()
