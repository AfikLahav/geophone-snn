"""X gate — 2-way combinatorial coverage over (subkind x terrain-stratum x noise-condition x
difficulty-zone) for each class's present windows (NIST covering-array practice: pairwise
interactions catch >90% of failure modes; full-grid floors are intractable).

Coverage per axis-pair = |realized level-pairs| / (|realized levels A| x |realized levels B|).
Level sets are REALIZED ones, so physically-excluded combinations (e.g. vehicle families under
the VEH_MIN_VS gate never appear as levels) don't count as holes. Missing pairs are listed.

Usage: python coverage_crossings.py <labels_dir> <tag>
Writes COVERAGE_CROSSINGS_<tag>.json next to this script.
"""
import os, sys, glob, json
from itertools import combinations
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = sys.argv[1] if len(sys.argv) > 1 else r"G:/geophone_synth/labels_v4/windows_3s"
TAG = sys.argv[2] if len(sys.argv) > 2 else "v4"
CLASSES = ("human", "vehicle", "animal")
STRATA = {"soft_soil": "soft", "loess": "soft", "sabkha": "soft", "snow": "soft",
          "sand": "medium", "wet_soil": "medium", "clay": "medium", "dirt_road": "medium",
          "chalk_marl": "medium", "terra_rossa": "medium",
          "gravel": "firm", "kurkar": "firm", "cover_basalt": "firm",
          "rock": "hard", "frozen": "hard",
          "asphalt": "paved", "concrete": "paved", "paving": "paved",
          "susp_concrete": "paved", "wood_floor": "paved"}

sh = sorted(glob.glob(os.path.join(LABELS, "labels_shard_*.parquet")))
cols = (["coarse", "subkind", "family", "noise_condition"]
        + [f"{c}_{s}" for c in CLASSES for s in ("level", "zone3")])
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sh], ignore_index=True)
df["stratum"] = df["family"].map(lambda f: STRATA.get(str(f), "other"))

rep = {"labels": LABELS, "per_class": {}, "missing_pairs": {}}
pair_covs = []
for c in CLASSES:
    d = df[(df.coarse == c) & (df[f"{c}_level"] > 0)].copy()
    d["zone"] = d[f"{c}_zone3"].map({0: "subfloor", 1: "marginal", 2: "detectable"})
    axes = {"subkind": d["subkind"].astype(str), "stratum": d["stratum"].astype(str),
            "condition": d["noise_condition"].astype(str), "zone": d["zone"].astype(str)}
    covs = {}
    missing_all = []
    for a, b in combinations(axes, 2):
        la = sorted(set(axes[a])); lb = sorted(set(axes[b]))
        realized = set(zip(axes[a], axes[b]))
        possible = len(la) * len(lb)
        covs[f"{a}x{b}"] = round(len(realized) / possible, 4)
        if len(realized) < possible:
            missing = [(x, y) for x in la for y in lb if (x, y) not in realized]
            missing_all += [f"{a}={x} & {b}={y}" for x, y in missing[:12]]
    rep["per_class"][c] = covs
    rep["missing_pairs"][c] = missing_all[:24]
    pair_covs += list(covs.values())
    print(f"{c:8s} " + "  ".join(f"{k}:{v:.3f}" for k, v in covs.items()))
    if missing_all:
        print(f"         missing (first): {missing_all[:6]}")
rep["min_pair_coverage"] = round(float(min(pair_covs)), 4)
print(f"\nmin pair coverage (X subscore): {rep['min_pair_coverage']}")
json.dump(rep, open(os.path.join(HERE, f"COVERAGE_CROSSINGS_{TAG}.json"), "w"), indent=1)
print(f"wrote COVERAGE_CROSSINGS_{TAG}.json")
