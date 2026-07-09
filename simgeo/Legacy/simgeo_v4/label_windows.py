"""Phase L — parameterized window/label pass over the v4 corpus (render-decoupled labeling).

For a given window length W and hop, slide over every scene's reconstructed per-class clean +
noise blobs and emit per-window labels: per-class in-band SNR (exact, via the D2 answer-key
blobs), OCCUPANCY (fraction of the window covered by that class's emission activity, from
activity_json / D3), LEVEL (ordinal of the class source count when occupied, else 'none' -> D3/D4
so gap/pause windows label correctly), soft target, and the 3-zone gate label (sub-floor/marginal/
detectable at gates_<W>s.json). One parquet per shard -> labels_v4/windows_<W>s/.

Quota: K = ceil(2*T_COVER/W) windows/scene (T_COVER=36 s of hop-coverage), even temporal stride
(preserves the approach/pass/recede SNR spread; deterministic).

Usage: python simgeo_v4/label_windows.py <W_s> <hop_s> [corpus_dir] [labels_root] [nworkers]
"""
import os, sys, json, sqlite3, time, glob, warnings, math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label

FS = 1000.0
T_COVER = 36.0
LVL = {"none": 0, "single": 1, "multiple": 2}
CLASSES = ("human", "vehicle", "animal")


def _overlap(intervals, w0, w1):
    tot = 0.0
    for a, b in intervals:
        lo = max(a, w0); hi = min(b, w1)
        if hi > lo:
            tot += hi - lo
    return tot


def run_shard(args):
    shard_path, out_path, W, hop, gates = args
    import pyarrow as pa, pyarrow.parquet as pq
    nw = int(W * FS); nh = int(hop * FS)
    tau_lo = {c: gates[c]["tau_lo_db"] for c in CLASSES}
    tau_hi = {c: gates[c]["tau_hi_db"] for c in CLASSES}
    K = math.ceil(2 * T_COVER / W)
    db = sqlite3.connect(shard_path)
    meta = {r[0]: r for r in db.execute(
        "SELECT scene_id,split,coarse,subkind,profile_id,family,terrain_vs,noise_condition,"
        "masking,n_subjects,activity_json FROM scenes")}
    cols = {k: [] for k in (
        "scene_id", "t0", "split", "coarse", "subkind", "profile_id", "family", "terrain_vs",
        "noise_condition", "masking", "common_snr", "activity",
        *[f"{c}_{s}" for c in CLASSES for s in ("level", "snr", "soft", "occ", "zone3")])}
    done = fail = 0; t0 = time.time()
    cur = db.execute("SELECT scene_id,n_samples,noise_mv,clean_mv,clean_cls,clean_mv2,clean_cls2 FROM waveforms")
    for sid, n, nmv, cm1, cls1, cm2, cls2 in cur:
        try:
            m = meta.get(sid)
            if m is None:
                continue
            noise = np.frombuffer(nmv, np.float32).astype(np.float64)
            clean_by = {}
            if cm1 is not None:
                clean_by[cls1] = np.frombuffer(cm1, np.float32).astype(np.float64)
            if cm2 is not None:
                clean_by[cls2] = np.frombuffer(cm2, np.float32).astype(np.float64)
            aj = json.loads(m[10]); counts = aj.get("counts", {}); acts = aj.get("activity", {})
            # window positions (even-stride quota)
            pos = list(range(0, max(1, n - nw + 1), nh))
            if len(pos) > K:
                pos = [pos[j] for j in np.round(np.linspace(0, len(pos) - 1, K)).astype(int)]
            for i0 in pos:
                w0, w1 = i0 / FS, (i0 + nw) / FS
                nb = noise[i0:i0 + nw]
                row_snr = {}
                any_act = 0
                for c in CLASSES:
                    lo, hi = label.BANDS[c]
                    if c in clean_by:
                        cs = label._band_rms(clean_by[c][i0:i0 + nw], lo, hi)
                        ns = label._band_rms(nb, lo, hi) + 1e-12
                        snr = 20.0 * np.log10(max(cs, 1e-12) / ns)
                    else:
                        snr = -99.0
                    occ = _overlap(acts.get(c, []), w0, w1) / W if c in acts else 0.0
                    occ = float(min(max(occ, 0.0), 1.0))
                    if occ > 0.0:
                        lvl = LVL[label.ordinal(int(counts.get(c, 0)))]
                        any_act = 1
                    else:
                        lvl = 0
                    z3 = 0 if snr < tau_lo[c] else (1 if snr < tau_hi[c] else 2)
                    cols[f"{c}_level"].append(np.int8(lvl))
                    cols[f"{c}_snr"].append(np.float32(snr))
                    cols[f"{c}_soft"].append(np.float32(label.soft_target(snr) if occ > 0 else 0.0))
                    cols[f"{c}_occ"].append(np.float32(occ))
                    cols[f"{c}_zone3"].append(np.int8(z3))
                    row_snr[c] = snr
                # common band SNR (max over present classes)
                pres = [row_snr[c] for c in CLASSES if c in clean_by]
                cols["common_snr"].append(np.float32(max(pres) if pres else -99.0))
                cols["activity"].append(np.int8(any_act))
                cols["scene_id"].append(sid); cols["t0"].append(np.float32(round(w0, 3)))
                cols["split"].append(m[1]); cols["coarse"].append(m[2]); cols["subkind"].append(m[3])
                cols["profile_id"].append(m[4]); cols["family"].append(m[5])
                cols["terrain_vs"].append(np.float32(m[6])); cols["noise_condition"].append(m[7] or "calm")
                cols["masking"].append(np.int8(m[8] or 0))
            done += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  {os.path.basename(shard_path)} sid {sid} FAIL {type(e).__name__}: {e}", flush=True)
    db.close()
    arr = {}
    for k, v in cols.items():
        if k in ("split", "coarse", "subkind", "profile_id", "family", "noise_condition"):
            arr[k] = pa.array(v, pa.string()).dictionary_encode()
        elif k == "scene_id":
            arr[k] = pa.array(v, pa.int64())
        else:
            arr[k] = pa.array(np.asarray(v))
    pq.write_table(pa.table(arr), out_path, compression="zstd")
    return os.path.basename(shard_path), done, fail, len(cols["scene_id"]), (time.time() - t0) / 60


def main():
    W = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    hop = float(sys.argv[2]) if len(sys.argv) > 2 else W / 2
    corpus = sys.argv[3] if len(sys.argv) > 3 else r"G:/geophone_synth/corpus_v4"
    lroot = sys.argv[4] if len(sys.argv) > 4 else r"G:/geophone_synth/labels_v4"
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    tag = f"{W:g}s"
    gpath = os.path.join(HERE, "..", "..", "..", "dataset_validation", f"gates_{tag}.json")
    gates = json.load(open(gpath))["classes"]
    out_dir = os.path.join(lroot, f"windows_{tag}"); os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(corpus, "shard_*.sqlite")),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))
    args = [(s, os.path.join(out_dir, f"labels_shard_{i}.parquet"), W, hop, gates)
            for i, s in enumerate(shards)]
    print(f"label pass W={W}s hop={hop}s: {len(shards)} shards, K={math.ceil(2*T_COVER/W)} win/scene, "
          f"{nw} workers -> {out_dir}", flush=True)
    t0 = time.time(); tot = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, done, fail, nwin, mins in ex.map(run_shard, args):
            tot += nwin
            print(f"{name}: {done} scenes, {fail} fail, {nwin} windows, {mins:.1f} min", flush=True)
    print(f"LABEL PASS DONE: {tot} windows in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
