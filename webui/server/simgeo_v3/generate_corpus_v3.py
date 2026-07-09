"""v3 LEAN production render. Reuses generate_corpus.gen_scene (validated v3 compute) but persists
ONLY: the amplitude blobs (clean_mv + noise_mv) + the full window LABELS + minimal scene info.
Dropped vs the full generator: v_signal & v_noise_ground blobs, config_json, and ~20 metadata columns
(user-requested corner-cut to get to training fast; labels stay correct — they're computed in-memory).
Reads library_v3.json + splits_v3.json. v1/v2 (corpus_150k, corpus_v2, library.json) are untouched.

Usage: python generate_corpus_v3.py <n_scenes> <out_dir(must contain 'v3')> <nworkers> [max_specs]
  max_specs (even-stride subsample) => TIMING mode: render a representative slice and project to 171k.
"""
import os, sys, json, sqlite3, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_corpus as gc                       # validated gen_scene + scene constants


def build_plan_v3(n_target, seed=gc.BASE_SEED):
    """Same composition logic as gc.build_plan but reads library_v3.json + splits_v3.json."""
    lib = json.load(open(os.path.join(HERE, "..", "simgeo_banks", "library_v3.json")))
    splits = json.load(open(os.path.join(HERE, "..", "simgeo_banks", "splits_v3.json")))
    banks = [p for p in lib if not p.get("modal")]
    rng = np.random.default_rng(seed)
    subj, noth, mixed = [], [], []
    for p in banks:
        pid = p["profile_id"]; split = splits.get(pid, "train"); veh_ok = p["vs_top_ms"] >= gc.VEH_MIN_VS_V3
        for sub in gc.HUMAN:
            subj.append((pid, split, "human", sub))
        if veh_ok:
            for sub in gc.VEHICLE:
                subj.append((pid, split, "vehicle", sub))
        for sub in gc.ANIMAL:
            subj.append((pid, split, "animal", sub))
        for sub in gc.NOTHING:
            noth.append((pid, split, "nothing", sub, gc.CONFUSER_W if sub in ("machinery", "overflight") else 1.0))
        for combo in gc.MIXED:
            if combo in ("human+vehicle", "vehicle+animal") and not veh_ok:
                continue
            mixed.append((pid, split, "mixed", combo))
    mixed_target = 10000
    rest = max(n_target - mixed_target, len(subj) + len(noth))
    noth_target = int(rest * 0.52); subj_target = rest - noth_target
    plan = []; pc_subj = max(1, round(subj_target / len(subj))); sid = 0
    for (pid, split, coarse, sub) in subj:
        for _ in range(pc_subj):
            plan.append((sid, pid, split, coarse, sub)); sid += 1
    wsum = sum(w for *_, w in noth); sid = 1_000_000
    for (pid, split, coarse, sub, w) in noth:
        for _ in range(max(1, round(noth_target * w / wsum))):
            plan.append((sid, pid, split, coarse, sub)); sid += 1
    pc_mixed = max(1, round(mixed_target / len(mixed))); sid = 2_000_000
    for (pid, split, coarse, sub) in mixed:
        for _ in range(pc_mixed):
            plan.append((sid, pid, split, coarse, sub)); sid += 1
    rng.shuffle(plan); plan.sort(key=lambda r: r[1])         # profile-sorted: each worker loads each bank once
    comp = (f"subj{sum(1 for r in plan if r[0]<1_000_000)}/"
            f"noth{sum(1 for r in plan if 1_000_000<=r[0]<2_000_000)}/"
            f"mixed{sum(1 for r in plan if r[0]>=2_000_000)}")
    return plan, len(subj) + len(noth) + len(mixed), comp


# ---- LEAN schema: amplitude blobs + labels + minimal scene info ----
SCENE_COLS_V3 = "scene_id,profile_id,family,terrain_vs,split,coarse,subkind,dur_s,fs,n_subjects"
WAVE_COLS_V3 = "scene_id,n_samples,clean_mv,noise_mv"
WIN_COLS = gc.WIN_COLS


def init_db_v3(path):
    assert "v3" in path, f"refusing non-v3 path (v1/v2 guard): {path}"
    for sfx in ("", "-wal", "-shm"):
        if os.path.exists(path + sfx):
            os.remove(path + sfx)
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(f"""
    CREATE TABLE scenes({SCENE_COLS_V3});
    CREATE TABLE waveforms(scene_id INTEGER PRIMARY KEY, n_samples INTEGER, clean_mv BLOB, noise_mv BLOB);
    CREATE TABLE windows(window_id INTEGER PRIMARY KEY AUTOINCREMENT, {WIN_COLS});
    """)
    return db


def open_or_init_v3(out):
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
    return init_db_v3(out), set()


def run_shard_v3(args):
    plan, out, wi = args
    db, committed = open_or_init_v3(out)
    todo = [s for s in plan if s[0] not in committed]
    if committed:
        print(f"  v3-shard{wi}: RESUME {len(committed)} done, {len(todo)} to go", flush=True)
    t0 = time.time(); done = fail = 0; sbuf, vbuf, wbuf = [], [], []
    ns = SCENE_COLS_V3.count(",") + 1; nv = WAVE_COLS_V3.count(",") + 1; nwc = WIN_COLS.count(",") + 1
    for spec in todo:
        try:
            sr, vr, wr = gc.gen_scene(spec)
            # lean slices: scene_row 0..8 + n_subjects(26); wave_row sid,n,clean_mv(4),noise_mv(5)
            sbuf.append((sr[0], sr[1], sr[2], sr[3], sr[4], sr[5], sr[6], sr[7], sr[8], sr[26]))
            vbuf.append((vr[0], vr[1], vr[4], vr[5]))
            wbuf.extend(wr); done += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  v3-shard{wi} scene {spec[0]} FAIL {type(e).__name__}: {e}", flush=True)
        if len(sbuf) >= 300:
            db.executemany(f"INSERT INTO scenes({SCENE_COLS_V3}) VALUES ({','.join('?'*ns)})", sbuf)
            db.executemany(f"INSERT INTO waveforms({WAVE_COLS_V3}) VALUES ({','.join('?'*nv)})", vbuf)
            db.executemany(f"INSERT INTO windows({WIN_COLS}) VALUES ({','.join('?'*nwc)})", wbuf)
            db.commit(); sbuf, vbuf, wbuf = [], [], []
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"  v3-shard{wi}: {done} scenes, {(time.time()-t0)/60:.1f} min", flush=True)
    if sbuf:
        db.executemany(f"INSERT INTO scenes({SCENE_COLS_V3}) VALUES ({','.join('?'*ns)})", sbuf)
        db.executemany(f"INSERT INTO waveforms({WAVE_COLS_V3}) VALUES ({','.join('?'*nv)})", vbuf)
        db.executemany(f"INSERT INTO windows({WIN_COLS}) VALUES ({','.join('?'*nwc)})", wbuf)
        db.commit()
    db.close()
    return wi, done, fail


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 171000
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"G:\geophone_synth\corpus_v3"
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    max_specs = int(sys.argv[4]) if len(sys.argv) > 4 else None
    assert "v3" in out_dir, "refusing non-v3 out_dir (v1/v2 guard)"
    os.makedirs(out_dir, exist_ok=True)
    plan, ncells, comp = build_plan_v3(n_target)
    if max_specs:                                            # TIMING: even-stride representative slice
        step = max(1, len(plan) // max_specs)
        plan = plan[::step][:max_specs]
    print(f"v3 plan: {len(plan)} scenes ({ncells} cells; {comp}), {nw} workers -> {out_dir}", flush=True)
    shards = [plan[i::nw] for i in range(nw)]
    args = [(shards[i], os.path.join(out_dir, f"shard_{i}.sqlite"), i) for i in range(nw)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for wi, done, fail in ex.map(run_shard_v3, args):
            print(f"v3-shard {wi}: {done} scenes, {fail} failed", flush=True)
    dt = (time.time() - t0) / 60
    print(f"DONE {len(plan)} scenes in {dt:.1f} min -> {out_dir}", flush=True)
    if max_specs and dt > 0:
        rate = len(plan) / (dt * 60)
        print(f"  rate {rate:.1f} scenes/s on {nw} workers; PROJECT 171k -> {171000/rate/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
