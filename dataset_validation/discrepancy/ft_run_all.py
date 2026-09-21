"""
ft_run_all.py - Master orchestrator for the v4 feature-space discrepancy report.

Runs all four analyses in sequence (data loaded once), writes:
  discrepancy_features.json  - machine-readable full results
  discrepancy_features.md    - human-readable dense summary for the report compiler

Usage:
  cd os.environ.get("PROJECT_ROOT", ".")
  python dataset_validation/discrepancy/ft_run_all.py
"""
import sys, os, json, time, io
import numpy as np

# Force UTF-8 stdout/stderr for MD output compatibility on Windows cp1252
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4"))
DISC_DIR = os.path.join(ROOT, "dataset_validation", "discrepancy")
sys.path.insert(0, DISC_DIR)

import ft_00_load as L
import ft_01_auroc as A1
import ft_02_ood   as A2
import ft_03_classconditional as A3
import ft_04_clipz as A4

OUT_JSON = os.path.join(DISC_DIR, "discrepancy_features.json")
OUT_MD   = os.path.join(DISC_DIR, "discrepancy_features.md")

t0 = time.time()
print("="*60)
print("Loading scaler and model features...")
scaler = L.load_scaler()
model_feats = L.load_model_features()
print(f"  Scaler features: {len(scaler['features'])}, model_feats: {len(model_feats)}")

print("Loading synthetic features...")
synth_df = L.load_synth()

# Extract 132-col matrix from synth_df (in FEATURE_NAMES order)
feat_names = L.FEATURE_NAMES
# Some parquet files may have extra meta columns; keep only feature columns that exist
synth_cols_avail = [f for f in feat_names if f in synth_df.columns]
missing_synth = [f for f in feat_names if f not in synth_df.columns]
if missing_synth:
    print(f"  [WARN] {len(missing_synth)} features missing from synth parquet: {missing_synth[:5]}...")
synth_X_full = synth_df[synth_cols_avail].values.astype(np.float64)
# Map to full 132 matrix (fill missing with NaN)
synth_X = np.full((len(synth_df), 132), np.nan)
for j, fn in enumerate(synth_cols_avail):
    fi = feat_names.index(fn)
    synth_X[:, fi] = synth_X_full[:, j]
print(f"  synth_X shape: {synth_X.shape}")

print("Computing real features (this takes ~2-5 min)...")
real_X, real_labels = L.load_real()
print(f"  real_X shape: {real_X.shape}")

elapsed = time.time() - t0
print(f"\nData loading done in {elapsed:.1f}s. Starting analyses...\n")

# ============================================================================
print("="*60)
print("ANALYSIS 1: Per-feature domain AUROC (all 132, all windows pooled)")
auroc_dict = A1.compute_domain_auroc(synth_X, real_X, feat_names)
family_auroc = A1.family_auroc(auroc_dict, L.FAMILIES, feat_names)
ranked = A1.ranked_table(auroc_dict)
print(f"  Done. Top-5 most separable features:")
for fn, auc, dev in ranked[:5]:
    print(f"    {fn:35s}  AUROC={auc:.4f}  |dev|={dev:.4f}")
print(f"  Per-family median AUROC:")
for fam, stats in sorted(family_auroc.items(), key=lambda x: -x[1]["median_auroc"]):
    print(f"    {fam:12s}  median_AUROC={stats['median_auroc']:.4f}")

# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 2: OOD z-score analysis (104 model features via scaler.json)")
z_real, feat_order_104, clip = A2.apply_scaler(real_X, feat_names, scaler)
ood_rows = A2.ood_stats(z_real, feat_order_104, clip)
family_ood = A2.family_ood(ood_rows, L.FAMILIES, feat_names)
print(f"  Top-10 OOD features (median |z|):")
for r in ood_rows[:10]:
    print(f"    {r['feature']:35s}  med|z|={r['median_abs_z']:.2f}  frac>2={r['frac_ood_2']:.2%}")
print(f"  Per-family OOD:")
for fam, stats in sorted(family_ood.items(), key=lambda x: -x[1]["median_median_abs_z"]):
    print(f"    {fam:12s}  median_med_z={stats['median_median_abs_z']:.2f}  mean_frac_ood2={stats['mean_frac_ood_2']:.2%}")

# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 3: Class-conditional domain AUROC")
cond_results = A3.conditional_auroc(synth_df, synth_X, real_X, real_labels, feat_names)
cond_summary = A3.class_conditional_summary(cond_results, feat_names)
print(f"  Class-conditional summaries:")
for cls, stats in cond_summary.items():
    print(f"    {cls:10s}  median_AUROC={stats['median_auroc']:.4f}  frac>0.7={stats['frac_above_0p7']:.2%}")
    for top in stats["top5_separable"][:3]:
        print(f"      {top['feature']:35s}  {top['auroc']:.4f}")
# Compare class-conditional vs overall
print(f"\n  Overall median AUROC:      {np.median([v for v in auroc_dict.values() if np.isfinite(v)]):.4f}")
for cls in cond_summary:
    print(f"  {cls} class-cond median AUROC: {cond_summary[cls]['median_auroc']:.4f}")

# ============================================================================
print("\n" + "="*60)
print("ANALYSIS 4: CLIPZ saturation (fraction real windows hitting ±8 rail)")
z_synth, _, _ = A2.apply_scaler(synth_X, feat_names, scaler)
clip_rows = A4.clip_saturation(z_real, z_synth, feat_order_104, clip)
print(f"  Top-10 clip offenders (real frac clipped):")
for r in clip_rows[:10]:
    print(f"    {r['feature']:35s}  real_clipped={r['real_frac_clipped']:.2%}  synth_clipped={r['synth_frac_clipped']:.2%}")

# ============================================================================
print("\n" + "="*60)
print("Writing JSON output...")

# Build full per-feature table
per_feature = []
for i, fn in enumerate(feat_names):
    # find family
    fam_name = "UNKNOWN"
    for fam, idxs in L.FAMILIES.items():
        if fam == "v2NEW30":
            continue
        if i in idxs:
            fam_name = fam
            break

    row = {
        "feature":  fn,
        "index":    i,
        "family":   fam_name,
        "domain_auroc": auroc_dict.get(fn),
        "domain_auroc_deviation": abs(auroc_dict.get(fn, 0.5) - 0.5) if auroc_dict.get(fn) is not None else None,
    }
    # OOD stats (only for 104 scaler features)
    ood_lookup = {r["feature"]: r for r in ood_rows}
    if fn in ood_lookup:
        od = ood_lookup[fn]
        row.update({
            "median_abs_z": od["median_abs_z"],
            "p90_abs_z":    od["p90_abs_z"],
            "frac_ood_2":   od["frac_ood_2"],
            "frac_ood_4":   od["frac_ood_4"],
        })
    else:
        row.update({"median_abs_z": None, "p90_abs_z": None,
                    "frac_ood_2": None, "frac_ood_4": None})
    # Clip saturation (only for 104 scaler features)
    clip_lookup = {r["feature"]: r for r in clip_rows}
    if fn in clip_lookup:
        cr = clip_lookup[fn]
        row.update({
            "real_frac_clipped":  cr["real_frac_clipped"],
            "synth_frac_clipped": cr["synth_frac_clipped"],
        })
    else:
        row.update({"real_frac_clipped": None, "synth_frac_clipped": None})
    # Class-conditional AUROC
    for cls in ("human", "vehicle", "nothing"):
        key = f"auroc_{cls}"
        if cls in cond_results:
            row[key] = cond_results[cls].get(fn)
        else:
            row[key] = None
    per_feature.append(row)

output = {
    "meta": {
        "n_synth": int(len(synth_df)),
        "n_real":  int(len(real_X)),
        "real_label_dist": {k: int(v) for k, v in
                            zip(*np.unique(real_labels, return_counts=True))},
        "n_features": 132,
        "n_model_features": len(model_feats),
        "clip_value": clip,
        "elapsed_sec": round(time.time() - t0, 1),
    },
    "family_auroc":      family_auroc,
    "family_ood":        family_ood,
    "class_conditional": cond_summary,
    "top25_ood_features": [
        {k: v for k, v in r.items()} for r in ood_rows[:25]
    ],
    "top25_clip_features": [
        {k: v for k, v in r.items()} for r in clip_rows[:25]
    ],
    "top25_domain_auroc": [
        {"feature": fn, "auroc": auc, "dev": abs(auc - 0.5)}
        for fn, auc, _ in ranked[:25]
    ],
    "per_feature": per_feature,
}

with open(OUT_JSON, "w") as fh:
    json.dump(output, fh, indent=2, default=lambda x: None if x != x else x)
print(f"  Written: {OUT_JSON}")

# ============================================================================
print("Writing MD summary...")

# Build MD
overall_median_auroc = float(np.median([v for v in auroc_dict.values() if np.isfinite(v)]))

md_lines = [
    "# Feature-Space Discrepancy Report — v4 Real vs Synthetic",
    "",
    f"N_synth={len(synth_df)}, N_real={len(real_X)} windows | "
    f"132 features | 104 model features | clip=±{clip:.0f}",
    "",
    "## 1. Per-Family Domain AUROC (all classes pooled)",
    "",
    "AUROC = P(real scores higher than synth). 0.5=indistinguishable, >0.7=notable gap.",
    "",
    "| Family     | N_feat | Median AUROC | Max AUROC | Max |dev| |",
    "|------------|--------|-------------|-----------|----------|",
]
for fam, st in sorted(family_auroc.items(), key=lambda x: -x[1]["median_auroc"]):
    if fam == "v2NEW30":
        continue
    md_lines.append(
        f"| {fam:<10} | {st['n_features']:6d} | {st['median_auroc']:.4f}      "
        f"| {st['max_auroc']:.4f}    | {st['max_deviation_from_half']:.4f}   |"
    )
md_lines.append(f"\nOverall median AUROC across all 132 features: **{overall_median_auroc:.4f}**")

md_lines += [
    "",
    "## 2. Top 25 Most Domain-Separable Features (|AUROC−0.5| rank)",
    "",
    "| Rank | Feature                          | AUROC  | Family   |",
    "|------|----------------------------------|--------|----------|",
]
feat2fam = {}
for fam, idxs in L.FAMILIES.items():
    if fam == "v2NEW30":
        continue
    for i in idxs:
        feat2fam[feat_names[i]] = fam

for rank, (fn, auc, dev) in enumerate(ranked[:25], 1):
    fam = feat2fam.get(fn, "?")
    md_lines.append(f"| {rank:4d} | {fn:<32} | {auc:.4f} | {fam:<8} |")

md_lines += [
    "",
    "## 3. OOD Analysis — Top 25 Features by Median |z| (scaler.json z-units)",
    "",
    "|z|>2 = out-of-distribution for the model. Scaler covers 104 model features.",
    "",
    "| Rank | Feature                          | Med|z| | P90|z| | Frac>2 | Frac>4 |",
    "|------|----------------------------------|--------|--------|--------|--------|",
]
for rank, r in enumerate(ood_rows[:25], 1):
    md_lines.append(
        f"| {rank:4d} | {r['feature']:<32} | {r['median_abs_z']:6.2f} "
        f"| {r['p90_abs_z']:6.2f} | {r['frac_ood_2']:6.1%} | {r['frac_ood_4']:6.1%} |"
    )

md_lines += [
    "",
    "### Per-Family OOD (104-feature subset)",
    "",
    "| Family     | Median Med|z| | Max Med|z| | MeanFrac>2 |",
    "|------------|-------------|-----------|------------|",
]
for fam, st in sorted(family_ood.items(), key=lambda x: -x[1]["median_median_abs_z"]):
    if fam == "v2NEW30":
        continue
    md_lines.append(
        f"| {fam:<10} | {st['median_median_abs_z']:13.2f} "
        f"| {st['max_median_abs_z']:9.2f} | {st['mean_frac_ood_2']:10.1%} |"
    )

md_lines += [
    "",
    "## 4. CLIPZ Saturation — Top Offenders",
    "",
    "Fraction of real windows where |z| ≥ 8 (model sees railed feature value).",
    "",
    "| Rank | Feature                          | Real Clipped | Synth Clipped |",
    "|------|----------------------------------|-------------|---------------|",
]
for rank, r in enumerate(clip_rows[:20], 1):
    if r["real_frac_clipped"] < 0.001:
        break
    md_lines.append(
        f"| {rank:4d} | {r['feature']:<32} | {r['real_frac_clipped']:11.1%} "
        f"| {r['synth_frac_clipped']:13.1%} |"
    )

md_lines += [
    "",
    "## 5. Class-Conditional Domain AUROC",
    "",
    "Does the domain gap come from the class signal or the background?",
    "AUROC computed separately for matched class pairs (real_human vs synth_human, etc.)",
    "",
    f"Overall (all classes pooled) median AUROC: **{overall_median_auroc:.4f}**",
    "",
    "| Class   | Median AUROC | Frac>0.7 | Verdict                          |",
    "|---------|-------------|----------|----------------------------------|",
]
for cls, st in cond_summary.items():
    ma = st["median_auroc"] or 0
    f7 = st["frac_above_0p7"] or 0
    if overall_median_auroc > 0.65 and ma < 0.57:
        verdict = "Gap mostly background/noise"
    elif ma > 0.65:
        verdict = "Class signal itself differs"
    else:
        verdict = "Moderate class-signal gap"
    md_lines.append(f"| {cls:<7} | {ma:.4f}      | {f7:.1%}    | {verdict} |")

md_lines += [
    "",
    "### Top-3 Most Separable Features per Class",
    "",
]
for cls, st in cond_summary.items():
    md_lines.append(f"**{cls}:**")
    for t in st.get("top5_separable", [])[:3]:
        md_lines.append(f"  - {t['feature']}: AUROC={t['auroc']:.4f}")

md_lines += [
    "",
    "---",
    f"*Generated by ft_run_all.py | elapsed={round(time.time()-t0)}s*",
]

with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(md_lines) + "\n")
print(f"  Written: {OUT_MD}")
print(f"\nDone in {time.time()-t0:.1f}s.")
