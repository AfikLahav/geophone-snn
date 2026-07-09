"""Full-corpus feature extraction pass — all windows, all shards (user decision:
analysis on the entire thing; targeted passes later).

Per shard: load the windows table into RAM (BLOB-free, fast), then stream scenes
(one BLOB read each), scene_precompute once, window_features per labeled window,
write one parquet per shard to N:\\geophone_synth\\features\\.

Split comes from terrain_models/splits_90_10.json (the 90/10 re-split), NOT the stale
shard split column. Levels stored as int8 (0 none / 1 single / 2 multiple),
zone as int8 (0 nothing / 1 marginal / 2 detectable).

Usage: python extract_features.py [nworkers=7] [out_dir] [corpus_dir]
"""
import os, sys, json, sqlite3, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CORPUS = r"G:/geophone_synth/corpus_v2"        # v2 (forward slashes — bash-safe)
OUT = r"G:/geophone_synth/features_v2"
LVL = {"none": 0, "single": 1, "multiple": 2}
ZON = {"nothing": 0, "marginal": 1, "detectable": 2}


def run_shard(args):
    shard_path, out_path = args
    import pyarrow as pa
    import pyarrow.parquet as pq
    import features as F
    splits = json.load(open(os.path.join(HERE, "..", "..", "..", "terrain_models", "splits_90_10.json")))

    db = sqlite3.connect(shard_path)
    # windows table -> per-scene label rows (BLOB-free, loads in seconds)
    wrows = {}
    for r in db.execute("SELECT scene_id,t0,human_level,human_snr,human_soft,"
                        "vehicle_level,vehicle_snr,vehicle_soft,animal_level,animal_snr,"
                        "animal_soft,common_snr,zone,activity FROM windows"):
        wrows.setdefault(r[0], []).append(r[1:])
    n_scenes = db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]

    meta_cols = {k: [] for k in ("scene_id", "t0", "profile_id", "family", "coarse",
                                 "subkind", "split", "terrain_vs", "wind", "rain",
                                 "human_level", "human_snr", "human_soft",
                                 "vehicle_level", "vehicle_snr", "vehicle_soft",
                                 "animal_level", "animal_snr", "animal_soft",
                                 "common_snr", "zone", "activity")}
    feats = []
    t0 = time.time(); done = 0; fail = 0
    # v2: signal = clean_mv + noise_mv (waveforms table), meta from scene columns;
    # v1: amplitude inline, meta from params JSON + splits_90_10 override. (schema-detected)
    is_v2 = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='waveforms'").fetchone() is not None
    if is_v2:
        cur = db.execute("SELECT s.scene_id,s.profile_id,s.family,s.terrain_vs,s.coarse,s.subkind,"
                         "s.split,s.wind,s.rain,w.clean_mv,w.noise_mv "
                         "FROM scenes s JOIN waveforms w USING(scene_id)")
    else:
        cur = db.execute("SELECT scene_id,profile_id,family,terrain_vs,coarse,subkind,"
                         "params,amplitude FROM scenes")
    for row in cur:
        try:
            if is_v2:
                sid, pid, fam, vs, coarse, sub, split, wind, rain, cm, nm = row
                noise = np.frombuffer(nm, np.float32)
                x = (np.frombuffer(cm, np.float32) + noise) if cm is not None else noise
                wind = float(wind or 0.0); rain = float(rain or 0.0)
            else:
                sid, pid, fam, vs, coarse, sub, params, blob = row
                x = np.frombuffer(blob, np.float32); pj = json.loads(params)
                split = splits.get(pid, "train"); wind = pj.get("wind", 0.0); rain = pj.get("rain", 0.0)
            wr = sorted(wrows.get(sid, []), key=lambda r: r[0])
            if not wr:
                continue
            pre = F.scene_precompute(x)
            for r in wr:
                i0 = int(round(r[0] * F.FS))
                if i0 + F.NW > len(x):
                    continue
                feats.append(F.window_features(pre, i0).astype(np.float32))
                m = meta_cols
                m["scene_id"].append(sid); m["t0"].append(r[0])
                m["profile_id"].append(pid); m["family"].append(fam)
                m["coarse"].append(coarse); m["subkind"].append(sub)
                m["split"].append(split); m["terrain_vs"].append(vs)
                m["wind"].append(wind); m["rain"].append(rain)
                (m["human_level"].append(LVL[r[1]]), m["human_snr"].append(r[2]),
                 m["human_soft"].append(r[3]))
                (m["vehicle_level"].append(LVL[r[4]]), m["vehicle_snr"].append(r[5]),
                 m["vehicle_soft"].append(r[6]))
                (m["animal_level"].append(LVL[r[7]]), m["animal_snr"].append(r[8]),
                 m["animal_soft"].append(r[9]))
                m["common_snr"].append(r[10]); m["zone"].append(ZON[r[11]])
                m["activity"].append(r[12])
            done += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  {os.path.basename(shard_path)} scene FAIL {type(e).__name__}: {e}", flush=True)
        if done % 2000 == 0:
            print(f"  {os.path.basename(shard_path)}: {done}/{n_scenes} scenes "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    db.close()

    X = np.stack(feats) if feats else np.empty((0, F.NFEAT), np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    cols = {}
    for k, v in meta_cols.items():
        if k in ("profile_id", "family", "coarse", "subkind", "split"):
            cols[k] = pa.array(v, pa.string()).dictionary_encode()
        elif k in ("scene_id",):
            cols[k] = pa.array(v, pa.int64())
        elif k in ("human_level", "vehicle_level", "animal_level", "zone", "activity"):
            cols[k] = pa.array(v, pa.int8())
        else:
            cols[k] = pa.array(np.asarray(v, np.float32))
    for i, name in enumerate(F.FEATURE_NAMES):
        cols[name] = pa.array(X[:, i])
    pq.write_table(pa.table(cols), out_path, compression="zstd")
    return os.path.basename(shard_path), done, fail, len(X), (time.time() - t0) / 60


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT
    corpus = sys.argv[3] if len(sys.argv) > 3 else CORPUS
    os.makedirs(out_dir, exist_ok=True)
    import glob
    shards = sorted(glob.glob(os.path.join(corpus, "shard_*.sqlite")),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))
    args = [(s, os.path.join(out_dir, f"features_shard_{i}.parquet"))
            for i, s in enumerate(shards)]
    print(f"feature pass: {len(shards)} shards, {nw} workers -> {out_dir}", flush=True)
    t0 = time.time()
    tot_w = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, done, fail, nwin, mins in ex.map(run_shard, args):
            tot_w += nwin
            print(f"{name}: {done} scenes, {fail} failed, {nwin} windows, "
                  f"{mins:.1f} min", flush=True)
    print(f"FEATURE PASS DONE: {tot_w} windows in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
