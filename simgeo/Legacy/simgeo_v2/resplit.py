"""Re-split profiles 90/10 train/val (user decision 2026-06-13; replaces 70/15/15;
real recordings remain the true held-out test). Profile-grouped as before — no terrain
profile crosses splits. Stratified per family. Writes terrain_models/splits_90_10.json
(authoritative for the training loader; shard 'split' columns are NOT rewritten — a
column UPDATE would rewrite every 270 KB waveform row = ~40 GB of IO) and updates
library.json 'split' in place, preserving the old assignment as 'split_70_15_15'.
"""
import os, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..", "..", "terrain_models", "library.json")
OUT = os.path.join(HERE, "..", "..", "..", "terrain_models", "splits_90_10.json")
SEED = 20260613


def main():
    lib = json.load(open(LIB))
    banks = [p for p in lib if not p.get("modal")]
    rng = np.random.default_rng(SEED)
    fams = collections.defaultdict(list)
    for p in banks:
        fams[p["family"]].append(p["profile_id"])
    split = {}
    for fam, pids in sorted(fams.items()):
        pids = sorted(pids)
        rng.shuffle(pids)
        n_val = max(1, round(0.10 * len(pids)))
        for pid in pids[:n_val]:
            split[pid] = "val"
        for pid in pids[n_val:]:
            split[pid] = "train"
    # sanity: val must include vehicle-capable terrain (vs>=250)
    vs = {p["profile_id"]: p["vs_top_ms"] for p in banks}
    val = [pid for pid, s in split.items() if s == "val"]
    val_veh = sum(1 for pid in val if vs[pid] >= 250)
    assert val_veh >= 10, f"val has only {val_veh} vehicle-capable profiles"
    json.dump(split, open(OUT, "w"), indent=1)
    # update library.json, preserving the old assignment
    for p in lib:
        if p.get("modal"):
            continue
        p["split_70_15_15"] = p.get("split", "train")
        p["split"] = split[p["profile_id"]]
    json.dump(lib, open(LIB, "w"))
    n = collections.Counter(split.values())
    print(f"90/10 re-split: {dict(n)}  ({n['val']/len(split):.1%} val, "
          f"{val_veh}/{len(val)} val profiles vehicle-capable)")
    print(f"wrote {OUT} and updated library.json (old kept as split_70_15_15)")


if __name__ == "__main__":
    main()
