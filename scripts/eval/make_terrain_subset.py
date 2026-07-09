"""Build terrain-filtered training subsets from features_v4_3s for the terrain-matching
experiment (ET). Real car session ~ soft soil/loess; real human session ~ firm
kurkar/rock/gravel/dirt_road (TERRAIN_MATCH.json).

Arms:
  matched : vehicle-scenes on SOFT, human-scenes on FIRM (mirrors the real sessions);
            nothing/mixed on SOFT+FIRM; animal unfiltered (no real data, keeps the head alive).
  anti    : vehicle-scenes on FIRM, human-scenes on SOFT (deliberately wrong) — falsification arm.
  random  : all classes filtered to a random profile subset of the same size as SOFT+FIRM
            (controls for 'fewer terrains/less data').

Usage: python make_terrain_subset.py <matched|anti|random>
Writes G:/geophone_synth/features_v4_3s_t<arm>/features_shard_*.parquet
"""
import os, sys, glob
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa

SOFT = {"soft_soil", "loess"}                                   # car-matched families
FIRM = {"kurkar", "rock", "gravel", "dirt_road"}                # human-matched families
SRC = r"G:/geophone_synth/features_v4_3s"
arm = sys.argv[1]
DST = SRC + f"_t{arm}"
os.makedirs(DST, exist_ok=True)

shards = sorted(glob.glob(os.path.join(SRC, "features_shard_*.parquet")))
if arm == "random":
    pids = set()
    for s in shards:
        pids |= set(pq.read_table(s, columns=["profile_id"]).column("profile_id").to_pylist())
    pids = sorted(pids)
    rng = np.random.default_rng(0)
    n_target = 122                                              # |SOFT families| + |FIRM families| banks
    RAND = set(rng.choice(pids, min(n_target, len(pids)), replace=False))
    print(f"random arm: {len(RAND)} of {len(pids)} profiles")

tot_in = tot_out = 0
for s in shards:
    t = pq.read_table(s)
    fam = pc.cast(t.column("family"), pa.string())
    coarse = pc.cast(t.column("coarse"), pa.string())
    import numpy as _np
    famv = _np.array(fam.to_pylist()); cov = _np.array(coarse.to_pylist())
    if arm == "matched":
        keep = ((cov == "vehicle") & _np.isin(famv, list(SOFT))) | \
               ((cov == "human") & _np.isin(famv, list(FIRM))) | \
               (cov == "animal") | \
               (_np.isin(cov, ["nothing", "mixed"]) & _np.isin(famv, list(SOFT | FIRM)))
    elif arm == "anti":
        keep = ((cov == "vehicle") & _np.isin(famv, list(FIRM))) | \
               ((cov == "human") & _np.isin(famv, list(SOFT))) | \
               (cov == "animal") | \
               (_np.isin(cov, ["nothing", "mixed"]) & _np.isin(famv, list(SOFT | FIRM)))
    elif arm == "random":
        pidv = _np.array(pc.cast(t.column("profile_id"), pa.string()).to_pylist())
        keep = _np.isin(pidv, list(RAND))
    else:
        raise SystemExit(f"unknown arm {arm}")
    sub = t.filter(pa.array(keep))
    tot_in += len(t); tot_out += len(sub)
    pq.write_table(sub, os.path.join(DST, os.path.basename(s)), compression="zstd")
print(f"{arm}: kept {tot_out:,}/{tot_in:,} windows -> {DST}")
