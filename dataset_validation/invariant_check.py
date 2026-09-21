"""Empirical verification of the Dataset Design Invariant diagnostics: compute B (balance),
C (difficulty coverage), N (class-nuisance neutrality) for v3 and v4 and check the ordering
matches the OBSERVED transfer outcomes (v4 > v3). If the diagnostics track reality on corpora
whose transfer we already measured, the formula is a valid steering instrument for v4.3.1+.

  B = 1 - mean_class TV(realized subkind mass within class, uniform)     target >= 0.85
  C = min_class fraction of 5 dB SNR bins in [tau_lo, +40] holding >=1% of present windows
  N = max Cramer's V(class, nuisance) over {family-stratum, noise-condition*}  target <= 0.05
      (*condition only stored in v4; documented physical couplings excluded and listed)
"""
import os, sys, glob, json
import numpy as np, pandas as pd

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
HERE = os.path.dirname(os.path.abspath(__file__))
GATES = json.load(open(os.path.join(HERE, "gates_3s.json")))["classes"]
CLASSES = ("human", "vehicle", "animal")

# v4.3.1 Change 2: per-subkind PHYSICAL reachability ceilings (snr_maps.py writes this). When present,
# the F gate exempts bins ABOVE a subkind's true ceiling (principled) instead of the realized-p99.5
# proxy. Absent (v4 runs) -> falls back to p99.5.
_CEIL_PATH = os.path.join(_GEO_ROOT, "config/subkind_ceilings_v431.json")
SUBK_CEIL = (json.load(open(_CEIL_PATH)).get("subkind_ceiling_db", {})
             if os.path.exists(_CEIL_PATH) else {})

# terrain strata by family stiffness (physics clusters)
STRATA = {"soft_soil": "soft", "loess": "soft", "sabkha": "soft", "snow": "soft",
          "sand": "medium", "wet_soil": "medium", "clay": "medium", "dirt_road": "medium",
          "chalk_marl": "medium", "terra_rossa": "medium",
          "gravel": "firm", "kurkar": "firm", "cover_basalt": "firm",
          "rock": "hard", "asphalt": "paved", "concrete": "paved", "paving": "paved",
          "frozen": "hard", "susp_concrete": "paved", "wood_floor": "paved"}


def cramers_v(a, b):
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    n = ct.sum()
    if n == 0: return np.nan
    chi2 = (((ct - np.outer(ct.sum(1), ct.sum(0)) / n) ** 2) /
            np.maximum(np.outer(ct.sum(1), ct.sum(0)) / n, 1e-12)).sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * max(min(r - 1, k - 1), 1))))


def f_gate(df):
    """F: every physically-reachable (subkind x 5dB detectable bin) cell holds >= FLOOR windows.
    Reachability v1: bins at/below the subkind's own p99.5 realized SNR (bins above it are
    treated unreachable -> exempt+reported; proper ceilings come from per-subkind maps in v4.3.1)."""
    FLOOR = 300
    cells = []
    for c in CLASSES:
        tau_hi = GATES[c]["tau_hi_db"]
        edges = np.arange(np.floor(tau_hi / 5) * 5, 35, 5)
        d = df[(df.coarse == c) & (df[f"{c}_level"] > 0)]
        for sk, g in d.groupby("subkind", observed=True):
            snr = g[f"{c}_snr"].to_numpy()
            if len(snr) < 50: continue
            ceil = SUBK_CEIL.get(str(sk))                     # v4.3.1 true physical reachability ceiling
            if ceil is None:
                ceil = np.percentile(snr, 99.5)               # v4 fallback (realized proxy)
            h, _ = np.histogram(np.clip(snr, edges[0], edges[-1]), bins=edges)
            for i in range(len(h)):
                if edges[i] <= ceil:                          # reachable bin
                    cells.append((c, str(sk), float(edges[i]), int(h[i]), h[i] >= FLOOR))
    ok = sum(1 for *_, p in cells if p)
    worst = sorted(cells, key=lambda t: t[3])[:5]
    return (round(ok / max(len(cells), 1), 3),
            [f"{c}/{sk}@{b:.0f}dB:{n}" for c, sk, b, n, _ in worst])


def diagnose(tag, pat, cols_extra=()):
    sh = sorted(glob.glob(pat))
    import pyarrow.parquet as _pq                              # v4.3.1: detect optional Q_j columns

    avail = set(_pq.ParquetFile(sh[0]).schema.names)
    qcols = [c for c in ("tier", "gain_log10", "quant_lsb", "rail_mv", "coupling_form") if c in avail]
    base = ["coarse", "subkind", "family"] + [f"{c}_{s}" for c in CLASSES for s in ("level", "snr")]
    cols = list(dict.fromkeys(base + list(cols_extra) + qcols))
    df = pd.concat([pd.read_parquet(s, columns=cols) for s in sh], ignore_index=True)
    out = {"tag": tag, "n": len(df)}
    # B: balance of subkind mass within class (scene-type likelihood normalization)
    tvs = []
    for cl in list(CLASSES) + ["nothing"]:
        sub = df[df.coarse == cl]["subkind"].value_counts(normalize=True)
        sub = sub[sub > 0]                                   # drop dictionary-encoded phantom categories
        if len(sub) < 2: continue
        tvs.append(0.5 * np.abs(sub.to_numpy() - 1.0 / len(sub)).sum())
    out["B_balance"] = round(1 - float(np.mean(tvs)), 3)
    # C: difficulty coverage (fraction of 5 dB bins >= 1% of present windows), min over classes
    cs = []
    for c in CLASSES:
        lo = GATES[c]["tau_lo_db"]; edges = np.arange(np.floor(lo / 5) * 5, 35, 5)
        pres = df[f"{c}_level"].to_numpy() > 0
        h, _ = np.histogram(np.clip(df[f"{c}_snr"].to_numpy()[pres], edges[0], edges[-1]), bins=edges)
        frac = h / max(h.sum(), 1)
        cs.append(float((frac >= 0.01).mean()))
    out["C_coverage"] = round(min(cs), 3)
    # N: neutrality — class vs terrain stratum (vehicle soft-ground exclusion is a DOCUMENTED coupling)
    strat = df["family"].map(lambda f: STRATA.get(str(f), "other"))
    out["N_class_stratum_V"] = round(cramers_v(df["coarse"], strat), 3)
    if "noise_condition" in df.columns:
        out["N_class_condition_V"] = round(cramers_v(df["coarse"], df["noise_condition"]), 3)
    # v4.3.1 [R5]: neutrality of each class-independent Q_j axis (V ~ 0 confirms class-independence;
    # a nonzero V would mean a sensor draw leaked the label and MUST sink the CDI). gain is
    # continuous -> quintile-binned; the rest are categorical (inf-safe via astype(str)).
    for a in qcols:
        vals = (pd.qcut(df[a].astype(float), 5, duplicates="drop") if a == "gain_log10"
                else df[a].astype(str))
        out[f"N_{a}_V"] = round(cramers_v(df["coarse"], vals), 3)
    # F: floors over reachable (subkind x detectable-bin) cells
    out["F_floor_frac"], out["F_worst_cells"] = f_gate(df)
    return out


def cdi_score(r, tag=None):
    """CDI scalar = MIN over normalized gate subscores (1 = target met). Min, not mean:
    a single failed gate must sink the score (a mean hides critical defects behind passing
    ones). Unmeasured gates are EXCLUDED but flagged — unmeasured != passing.
    X/P/D subscores are ingested from their tools' JSON artifacts when present."""
    subs = {}
    subs["B"] = min(r["B_balance"] / 0.85, 1.0)                       # target >= 0.85
    subs["C"] = float(r["C_coverage"])                                # target 1.0
    subs["N_stratum"] = min(0.05 / max(r["N_class_stratum_V"], 1e-6), 1.0)   # target V <= 0.05
    if "N_class_condition_V" in r:
        subs["N_condition"] = min(0.05 / max(r["N_class_condition_V"], 1e-6), 1.0)
    for k, val in r.items():                                                  # v4.3.1: each Q_j axis
        if k.startswith("N_") and k.endswith("_V") and k not in ("N_class_stratum_V", "N_class_condition_V"):
            subs["N_" + k[2:-2]] = min(0.05 / max(val, 1e-6), 1.0)
    if "F_floor_frac" in r:
        subs["F"] = float(r["F_floor_frac"])                          # fraction of reachable cells >= floor
    if tag:                                                           # ingest X / P / D artifacts
        xp = os.path.join(HERE, f"COVERAGE_CROSSINGS_{tag}.json")
        if os.path.exists(xp):
            subs["X"] = float(json.load(open(xp))["min_pair_coverage"])
        pp = os.path.join(HERE, f"NUISANCE_PROBE_{tag}.json")
        if os.path.exists(pp):
            auc = float(json.load(open(pp))["max_class_auc_zero_signal"])
            subs["P"] = float(np.clip((0.80 - auc) / 0.25, 0.0, 1.0))  # AUC 0.55->1.0, 0.80->0
        dp = os.path.join(HERE, f"DATASHEET_{tag}.json")
        subs["D"] = 1.0 if os.path.exists(dp) else 0.0
    bott = min(subs, key=subs.get)
    return round(float(subs[bott]), 3), bott, {k: round(v, 3) for k, v in subs.items()}


if __name__ == "__main__":
    if len(sys.argv) > 2:                                     # CLI: tag pattern [has_condition]
        res = [diagnose(sys.argv[1], sys.argv[2],
                        ("noise_condition",) if len(sys.argv) > 3 and sys.argv[3] == "1" else ())]
    else:
        res = [diagnose("v3", os.path.join(_GEO_ROOT, "features_v3/features_shard_*.parquet")),
               diagnose("v4", os.path.join(_GEO_ROOT, "labels_v4/windows_3s/labels_shard_*.parquet"), ("noise_condition",))]
    print(f"{'dataset':6s} {'B':>6s} {'C':>6s} {'N_str':>6s} {'N_cond':>7s} {'F':>6s} {'CDI':>6s}  bottleneck / subscores")
    for r in res:
        cdi, bott, subs = cdi_score(r, tag=r["tag"])
        r["CDI"] = cdi; r["bottleneck"] = bott; r["subscores"] = subs
        missing = [g for g in ("X", "P", "D") if g not in subs or (g == "D" and subs.get("D") == 0.0)]
        print(f"{r['tag']:6s} {r['B_balance']:>6} {r['C_coverage']:>6} {r['N_class_stratum_V']:>6} "
              f"{str(r.get('N_class_condition_V','-')):>7} {str(r.get('F_floor_frac','-')):>6} {cdi:>6}  "
              f"{bott}  {subs}" + (f"  [missing: {','.join(missing)}]" if missing else ""))
        if r.get("F_worst_cells"):
            print(f"       F worst cells: {r['F_worst_cells']}")
    suffix = f"_{sys.argv[1]}" if len(sys.argv) > 2 else ""
    json.dump(res, open(os.path.join(HERE, f"INVARIANT_CHECK{suffix}.json"), "w"), indent=1)
    print(f"wrote INVARIANT_CHECK{suffix}.json")
