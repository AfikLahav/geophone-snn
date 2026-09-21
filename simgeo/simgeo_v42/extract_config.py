"""Phase 1 of the DB refactor: recover the per-scene config that generation discarded.

For every scene, re-run the deterministic generator from its stored seed, ASSERT the
re-rendered waveform is bit-exact vs the stored blob (guards against code/bank drift —
a mismatch means the config can't be trusted, so it's flagged not written), and write
the recovered config to a per-shard config DB (scene_id + scalar columns + full
config_json). Tiny output (~MB/shard); the 38 GB waveforms are untouched here.

sid < 200000 -> generate_dataset.gen_scene ; sid >= 200000 -> append_multiveh.gen_scene_mv.
Usage: python extract_config.py [nworkers=7] [dataset_dir] [out_dir]
"""
import os, sys, glob, json, sqlite3, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATASET = os.path.join(_GEO_ROOT, "dataset_v2")
OUT = os.path.join(_GEO_ROOT, "config")
COLS = ["closest_approach_m", "speed_ms", "path_style", "onset_free", "f0_hz",
        "damping_h", "sens_g", "coupling_fc", "coupling_q", "gain", "noise_condition",
        "n_subjects", "primary_mass_kg", "wind", "rain"]
SID_MV = 200000


def init_cfg_db(path):
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(path + suf):
            os.remove(path + suf)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    coldefs = ("closest_approach_m REAL, speed_ms REAL, path_style TEXT, onset_free INTEGER, "
               "f0_hz REAL, damping_h REAL, sens_g REAL, coupling_fc REAL, coupling_q REAL, "
               "gain REAL, noise_condition TEXT, n_subjects INTEGER, primary_mass_kg REAL, "
               "wind REAL, rain REAL, config_json TEXT")
    db.execute(f"CREATE TABLE config(scene_id INTEGER PRIMARY KEY, {coldefs})")
    return db


def run_shard(args):
    shard_path, out_path = args
    import generate_dataset as gc, append_multiveh as mv

    name = os.path.basename(shard_path)
    src = sqlite3.connect(shard_path)
    rows = src.execute("SELECT scene_id,profile_id,split,coarse,subkind,amplitude "
                       "FROM scenes ORDER BY scene_id").fetchall()
    db = init_cfg_db(out_path)
    t0 = time.time(); done = 0; mism = 0; fail = 0
    buf = []
    ins = (f"INSERT INTO config(scene_id,{','.join(COLS)},config_json) "
           f"VALUES ({','.join('?' * (len(COLS) + 2))})")
    for sid, pid, split, coarse, sub, blob in rows:
        try:
            if sid >= SID_MV:
                sr, wr, cfg = mv.gen_scene_mv((sid, pid, split, sub))
            else:
                sr, wr, cfg = gc.gen_scene((sid, pid, split, coarse, sub))
            recon = np.frombuffer(sr[10], np.float32)
            stored = np.frombuffer(blob, np.float32)
            if len(recon) != len(stored) or not np.array_equal(recon, stored):
                mism += 1
                if mism <= 5:
                    print(f"  {name} sid={sid} BIT-MISMATCH (config NOT trusted)", flush=True)
                continue
            buf.append([sid] + [cfg.get(k) for k in COLS] + [json.dumps(cfg, default=float)])
            done += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  {name} sid={sid} FAIL {type(e).__name__}: {e}", flush=True)
        if len(buf) >= 500:
            db.executemany(ins, buf); db.commit(); buf = []
            if done % 4000 == 0:
                print(f"  {name}: {done} verified, {(time.time()-t0)/60:.1f} min", flush=True)
    if buf:
        db.executemany(ins, buf); db.commit()
    db.close(); src.close()
    return name, done, mism, fail, (time.time() - t0) / 60


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    dataset = sys.argv[2] if len(sys.argv) > 2 else DATASET
    out_dir = sys.argv[3] if len(sys.argv) > 3 else OUT
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(dataset, "shard_*.sqlite")),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))
    args = [(s, os.path.join(out_dir, os.path.basename(s).replace("shard_", "config_shard_")))
            for s in shards]
    print(f"config extract: {len(shards)} shards, {nw} workers (re-render + bit-exact verify)",
          flush=True)
    t0 = time.time(); tot = mismtot = failtot = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, done, mism, fail, mins in ex.map(run_shard, args):
            tot += done; mismtot += mism; failtot += fail
            print(f"{name}: {done} verified, {mism} mismatch, {fail} fail, {mins:.1f} min",
                  flush=True)
    print(f"CONFIG EXTRACT DONE: {tot} scenes verified, {mismtot} mismatches, {failtot} fails, "
          f"{(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)
    if mismtot:
        print("WARNING: bit-mismatches found -> those scenes' config was NOT written", flush=True)


if __name__ == "__main__":
    main()
