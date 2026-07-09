"""Phase D1 — v4 feature extraction. Joins the Phase-L labels (labels_v4/windows_<W>s) with
the corpus blobs: reconstructs the MODEL INPUT (sum of per-class clean blobs + noise, railed
+-256 mV) and computes the 132-feature bank per LABELED window, carrying all label columns
through. One parquet per shard -> features_v4_<W>s/.

Model-input reconstruction (D2): x = clean_mv (+ clean_mv2) + noise_mv, clipped +-256 mV
(same convention as extract_features.py:72-76; the per-class blobs are the answer key, the
sum is what the sensor delivers).

Usage: python simgeo_v4/extract_features_v4.py <W_s> [corpus] [labels_root] [feat_root] [nw]
"""
import os, sys, glob, time, sqlite3, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FS = 1000.0
CLASSES = ("human", "vehicle", "animal")
LABEL_CARRY = (["scene_id", "t0", "split", "coarse", "subkind", "profile_id", "family",
                "terrain_vs", "noise_condition", "masking", "common_snr", "activity"]
               + [f"{c}_{s}" for c in CLASSES for s in ("level", "snr", "soft", "occ", "zone3")])


def run_shard(args):
    corpus_shard, labels_shard, out_path, W = args
    import pyarrow as pa, pyarrow.parquet as pq
    import features as F
    nw = int(W * FS)
    lab = pq.read_table(labels_shard).to_pydict()
    # group labeled window indices by scene_id
    by_scene = {}
    for i, sid in enumerate(lab["scene_id"]):
        by_scene.setdefault(sid, []).append(i)
    db = sqlite3.connect(corpus_shard)
    feats = []; carry = {k: [] for k in LABEL_CARRY}
    t0 = time.time(); done = fail = 0
    cur = db.execute("SELECT scene_id,noise_mv,clean_mv,clean_mv2 FROM waveforms")
    blobs = {sid: (nm, c1, c2) for sid, nm, c1, c2 in cur}
    db.close()
    for sid, idxs in by_scene.items():
        try:
            if sid not in blobs:
                continue
            nm, c1, c2 = blobs[sid]
            x = np.frombuffer(nm, np.float32).astype(np.float64)
            if c1 is not None:
                x = x + np.frombuffer(c1, np.float32)
            if c2 is not None:
                x = x + np.frombuffer(c2, np.float32)
            x = np.clip(x, -256.0, 256.0)                      # D5 rail (model input)
            pre = F.scene_precompute(x)
            for i in idxs:
                i0 = int(round(lab["t0"][i] * FS))
                if i0 + nw > len(x):
                    continue
                feats.append(F.window_features(pre, i0).astype(np.float32))
                for k in LABEL_CARRY:
                    carry[k].append(lab[k][i])
            done += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  {os.path.basename(corpus_shard)} sid {sid} FAIL {type(e).__name__}: {e}", flush=True)
    X = np.stack(feats) if feats else np.empty((0, F.NFEAT), np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    cols = {}
    for k in LABEL_CARRY:
        v = carry[k]
        if k in ("split", "coarse", "subkind", "profile_id", "family", "noise_condition"):
            cols[k] = pa.array(v, pa.string()).dictionary_encode()
        elif k == "scene_id":
            cols[k] = pa.array(v, pa.int64())
        elif k.endswith("_level") or k.endswith("_zone3") or k in ("masking", "activity"):
            cols[k] = pa.array(np.asarray(v, np.int8()))
        else:
            cols[k] = pa.array(np.asarray(v, np.float32))
    for j, name in enumerate(F.FEATURE_NAMES):
        cols[name] = pa.array(X[:, j])
    pq.write_table(pa.table(cols), out_path, compression="zstd")
    return os.path.basename(corpus_shard), done, fail, len(X), (time.time() - t0) / 60


def main():
    W = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    corpus = sys.argv[2] if len(sys.argv) > 2 else r"G:/geophone_synth/corpus_v4"
    lroot = sys.argv[3] if len(sys.argv) > 3 else r"G:/geophone_synth/labels_v4"
    froot = sys.argv[4] if len(sys.argv) > 4 else r"G:/geophone_synth/features_v4"
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    tag = f"{W:g}s"
    ldir = os.path.join(lroot, f"windows_{tag}"); fdir = os.path.join(froot + f"_{tag}")
    os.makedirs(fdir, exist_ok=True)
    cshards = sorted(glob.glob(os.path.join(corpus, "shard_*.sqlite")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
    args = []
    for cs in cshards:
        i = int(cs.split("_")[-1].split(".")[0])
        ls = os.path.join(ldir, f"labels_shard_{i}.parquet")
        op = os.path.join(fdir, f"features_shard_{i}.parquet")
        if os.path.exists(op) and os.path.getsize(op) > 1_000_000:   # resume: skip completed shards
            continue
        if os.path.exists(ls):
            args.append((cs, ls, op, W))
    print(f"feature pass W={W}s: {len(args)} shards, {nw} workers -> {fdir}", flush=True)
    t0 = time.time(); tot = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, done, fail, n, mins in ex.map(run_shard, args):
            tot += n
            print(f"{name}: {done} scenes, {fail} fail, {n} windows, {mins:.1f} min", flush=True)
    print(f"FEATURE PASS DONE: {tot} windows in {(time.time()-t0)/60:.1f} min -> {fdir}", flush=True)


if __name__ == "__main__":
    main()
