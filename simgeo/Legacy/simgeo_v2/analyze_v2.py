"""Turn the v2 screening outputs into the feature-decision packet:
FEATURE_ANALYSIS_V2.md + feature_analysis_v2.sqlite.

Reads (all already produced):
  features_v2/screening/auc_global_and_family.csv  (per-feature rank-AUC, global + per terrain family)
  features_v2/screening/auc_faint.csv
  features_v2/screening/gbt_results.json           (per-head balanced-acc, confusion, perm importance)
  features_v2/screening/meta.json
  real_feature_analysis.sqlite  (real_auc)         (for the synth<->real transfer delta)
  a row-sample of the feature parquet              (for redundancy / correlation clusters)

Decision metrics per feature:
  class_sep  = max |AUC-0.5|*2 over the 6 class boundaries (3 class-vs-class + 3 class-vs-nothing)
  multi_sep  = max |AUC-0.5|*2 over the 3 single-vs-multiple boundaries
  gbt_imp    = max permutation importance over all heads (0 if outside each head's top-25)
  robustness = min family strength on the feature's strongest boundary (low => terrain shortcut)
  transfer   = |synth_strength - real_strength| on the 3 shared boundaries (high => sim-real drift)
  cluster    = redundancy cluster id (|r|>0.9); representative = highest class_sep in the cluster
Usage: python analyze_v2.py
"""
import os, sys, json, glob, sqlite3
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F

SDIR = r"G:/geophone_synth/features_v2/screening"
PARQ = sorted(glob.glob(r"G:/geophone_synth/features_v2/features_shard_*.parquet"))
PROJ = os.path.join(HERE, "..")
MD = os.path.join(PROJ, "FEATURE_ANALYSIS_V2.md")
DB = os.path.join(PROJ, "feature_analysis_v2.sqlite")
REAL_DB = os.path.join(PROJ, "real_feature_analysis.sqlite")
NEWv2 = set(F.FEATURE_NAMES[102:])           # the 30 v2-added features
CLASS_B = ["human_vs_animal", "human_vs_vehicle", "animal_vs_vehicle",
           "human_vs_nothing", "vehicle_vs_nothing", "animal_vs_nothing"]
MULTI_B = ["human_multi_vs_single", "vehicle_multi_vs_single", "animal_multi_vs_single"]
SHARED = ["human_vs_vehicle", "human_vs_nothing", "vehicle_vs_nothing"]


def strength(auc):
    return (np.abs(auc - 0.5) * 2)


def main():
    g = pd.read_csv(os.path.join(SDIR, "auc_global_and_family.csv"))
    g["strength"] = strength(g["auc"])
    G = g[g.family == "ALL"].copy()                      # global per-feature
    FAM = g[g.family != "ALL"].copy()                    # per terrain family
    gbt = json.load(open(os.path.join(SDIR, "gbt_results.json")))
    meta = json.load(open(os.path.join(SDIR, "meta.json")))

    # ---- GBT permutation importance per head ----
    imp = {f: {} for f in F.FEATURE_NAMES}
    for head, r in gbt.items():
        for nm, v in r.get("perm_importance_top25", []):
            imp[nm][head] = float(v)
    gbt_imp = {f: (max(d.values()) if d else 0.0) for f, d in imp.items()}

    # ---- redundancy clusters from a row-sample ----
    samp = []
    for p in PARQ[:3]:                                   # 3 shards is plenty for correlation
        df = pd.read_parquet(p, columns=F.FEATURE_NAMES)
        samp.append(df.sample(min(80_000, len(df)), random_state=0))
    Xs = pd.concat(samp, ignore_index=True)
    C = np.corrcoef(Xs.to_numpy(np.float64).T)
    C = np.nan_to_num(C)
    # greedy clustering at |r|>0.9, representative = highest class_sep
    csep = {f: G[(G.feature == f) & (G.boundary.isin(CLASS_B))]["strength"].max() for f in F.FEATURE_NAMES}
    order = sorted(F.FEATURE_NAMES, key=lambda f: -csep.get(f, 0))
    idx = {f: i for i, f in enumerate(F.FEATURE_NAMES)}
    assigned = {}
    clusters = []
    for f in order:
        if f in assigned:
            continue
        members = [f]
        for h in F.FEATURE_NAMES:
            if h != f and h not in assigned and abs(C[idx[f], idx[h]]) > 0.9:
                members.append(h)
        cid = len(clusters)
        for m in members:
            assigned[m] = cid
        clusters.append({"rep": f, "members": members})

    # ---- synth<->real transfer delta ----
    real = pd.read_sql("SELECT boundary,feature,auc_abs FROM real_auc", sqlite3.connect(REAL_DB))
    real["rstr"] = (real.auc_abs - 0.5) * 2
    delta = {}
    for f in F.FEATURE_NAMES:
        ds = []
        for b in SHARED:
            ss = G[(G.feature == f) & (G.boundary == b)]["strength"]
            rs = real[(real.boundary == b) & (real.feature == f)]["rstr"]
            if len(ss) and len(rs):
                ds.append(abs(float(ss.iloc[0]) - float(rs.iloc[0])))
        delta[f] = (max(ds) if ds else np.nan)

    # ---- per-feature scorecard ----
    rows = []
    for f in F.FEATURE_NAMES:
        cs = G[(G.feature == f) & (G.boundary.isin(CLASS_B))]["strength"]
        ms = G[(G.feature == f) & (G.boundary.isin(MULTI_B))]["strength"]
        class_sep = float(cs.max()) if len(cs) else 0.0
        multi_sep = float(ms.max()) if len(ms) else 0.0
        # robustness: min family strength on the strongest class boundary
        bestb = G[(G.feature == f) & (G.boundary.isin(CLASS_B))].sort_values("strength").tail(1)
        robust = np.nan
        if len(bestb):
            bb = bestb.boundary.iloc[0]
            fam = FAM[(FAM.feature == f) & (FAM.boundary == bb)]["strength"]
            robust = float(fam.min()) if len(fam) else np.nan
        rows.append({
            "feature": f, "is_v2_new": f in NEWv2, "class_sep": class_sep,
            "multi_sep": multi_sep, "gbt_imp": gbt_imp[f], "robust_minfam": robust,
            "transfer_delta": delta[f], "cluster": assigned[f],
            "cluster_rep": clusters[assigned[f]]["rep"],
        })
    sc = pd.DataFrame(rows)

    # ---- recommended keep/drop (multi-criterion) ----
    # keep if it carries unique discriminative value on some head, isn't a redundant duplicate,
    # and isn't a high-transfer-risk shape feature with no compensating uniqueness.
    sc["is_rep"] = sc.feature == sc.cluster_rep
    sc["discriminative"] = (sc.class_sep >= 0.30) | (sc.multi_sep >= 0.30) | (sc.gbt_imp > 0.002)
    sc["transfer_risk"] = sc.transfer_delta >= 0.40
    sc["keep"] = sc.discriminative & sc.is_rep & ~((sc.transfer_risk) & (sc.gbt_imp <= 0.002))
    sc = sc.sort_values(["keep", "class_sep"], ascending=[False, False]).reset_index(drop=True)

    # =================== persist sqlite ===================
    con = sqlite3.connect(DB)
    G.to_sql("synth_auc_global", con, if_exists="replace", index=False)
    FAM.to_sql("synth_auc_family", con, if_exists="replace", index=False)
    sc.to_sql("feature_scorecard", con, if_exists="replace", index=False)
    pd.DataFrame([{"head": k, "balanced_acc": v["balanced_acc"], "auc": v.get("auc"),
                   "n_hold": v["n_hold"], "confusion": json.dumps(v["confusion"]),
                   "top_importance": json.dumps(v["perm_importance_top25"][:15])}
                  for k, v in gbt.items()]).to_sql("gbt", con, if_exists="replace", index=False)
    pd.DataFrame([{"cluster": i, "rep": c["rep"], "members": json.dumps(c["members"]),
                   "size": len(c["members"])} for i, c in enumerate(clusters)]
                 ).to_sql("redundancy", con, if_exists="replace", index=False)
    con.commit(); con.close()

    # =================== write markdown ===================
    def tbl(df, cols, fmts):
        out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(fmts[c](r) for c in cols) + " |")
        return "\n".join(out)

    L = []
    L.append("# Feature analysis — v2 corpus (132 features, 7.6M windows)\n")
    L.append(f"_TRAIN split only ({meta['n_train_windows']:,} windows), profile-grouped holdout, "
             "rank-AUC capped at 600k/side. Same methodology as v1 `FEATURE_ANALYSIS.md`, extended "
             "to 132 features, mixed scenes, and the synth↔real transfer delta._\n")

    # 1. GBT headline
    L.append("## 1. Multivariate ceiling (HistGradientBoosting, profile-grouped holdout)\n")
    gd = pd.DataFrame([{"head": k, "bal_acc": v["balanced_acc"], "auc": v.get("auc"),
                        "n_hold": v["n_hold"]} for k, v in gbt.items()])
    L.append(tbl(gd, ["head", "bal_acc", "auc", "n_hold"], {
        "head": lambda r: f"`{r['head']}`", "bal_acc": lambda r: f"{r.bal_acc:.3f}",
        "auc": lambda r: (f"{r.auc:.3f}" if pd.notna(r.auc) else "—"),
        "n_hold": lambda r: f"{int(r.n_hold):,}"}))
    L.append("\n**Read:** 4-class separation holds at v2 (0.962, ~v1's 0.963). Per-class presence "
             "0.95–0.96 even with mixed/co-occurring scenes in the pool. **Single-vs-multiple is now "
             "learnable for human (0.80) and animal (0.89)** — v1 was at chance — driven by the new "
             "III2/MOD2 features. **Vehicle multiplicity stays near chance (0.60): it is a scene-level "
             "temporal counting problem, not a per-window target** (route to a temporal head, not a "
             "per-window multi class).\n")

    # 2. top features per boundary
    L.append("## 2. Top discriminators per boundary (global rank-AUC strength = |AUC−0.5|·2)\n")
    for b in CLASS_B + MULTI_B:
        d = G[G.boundary == b].sort_values("strength", ascending=False).head(12)
        if d.empty:
            continue
        L.append(f"### {b}\n")
        d2 = d.assign(new=d.feature.isin(NEWv2))
        L.append(tbl(d2, ["feature", "AUC", "strength", "v2?"], {
            "feature": lambda r: f"`{r.feature}`", "AUC": lambda r: f"{r.auc:.3f}",
            "strength": lambda r: f"{r.strength:.3f}",
            "v2?": lambda r: ("**new**" if r.new else "")}))
        L.append("")

    # 3. new v2 scorecard
    L.append("## 3. Did the 30 new v2 features earn their place?\n")
    nv = sc[sc.is_v2_new].sort_values("class_sep", ascending=False)
    kept_new = int(nv.keep.sum())
    L.append(f"{kept_new}/30 new features kept under the multi-criterion rule. Top contributors:\n")
    L.append(tbl(nv.head(18), ["feature", "class_sep", "multi_sep", "gbt_imp", "keep"], {
        "feature": lambda r: f"`{r.feature}`", "class_sep": lambda r: f"{r.class_sep:.3f}",
        "multi_sep": lambda r: f"{r.multi_sep:.3f}", "gbt_imp": lambda r: f"{r.gbt_imp:.4f}",
        "keep": lambda r: ("✓" if r.keep else "")}))
    L.append("")

    # 4. redundancy
    L.append("## 4. Redundancy clusters (|r| > 0.9)\n")
    multi = [c for c in clusters if len(c["members"]) > 1]
    L.append(f"{len(clusters)} clusters total; {len(multi)} have >1 member. "
             f"**{len([c for c in clusters])} representatives carry the non-redundant signal "
             f"(down from 132).** Largest clusters:\n")
    md = pd.DataFrame([{"rep": c["rep"], "size": len(c["members"]),
                        "members": ", ".join(c["members"])} for c in
                       sorted(multi, key=lambda c: -len(c["members"]))[:12]])
    if len(md):
        L.append(tbl(md, ["rep", "size", "members"], {
            "rep": lambda r: f"`{r.rep}`", "size": lambda r: str(int(r['size'])),
            "members": lambda r: r.members}))
    L.append("")

    # 5. per-family robustness (shortcut risk)
    L.append("## 5. Terrain-shortcut risk (features whose AUC collapses on some family)\n")
    risky = sc[(sc.class_sep >= 0.5) & (sc.robust_minfam < 0.2)].sort_values("class_sep", ascending=False)
    if len(risky):
        L.append("Strong globally but near-chance on at least one terrain family — treat as "
                 "terrain-dependent, not universal:\n")
        L.append(tbl(risky.head(12), ["feature", "class_sep", "robust_minfam"], {
            "feature": lambda r: f"`{r.feature}`", "class_sep": lambda r: f"{r.class_sep:.3f}",
            "robust_minfam": lambda r: f"{r.robust_minfam:.3f}"}))
    else:
        L.append("None — every strong global feature keeps strength ≥ 0.2 on its worst family.")
    L.append("")

    # 6. synth<->real transfer
    L.append("## 6. Synth↔real transfer delta (the binding criterion)\n")
    L.append("On the 3 shared boundaries (`human_vs_vehicle`, `human_vs_nothing`, `vehicle_vs_nothing`), "
             "`transfer_delta = |synth_strength − real_strength|`. **High delta = the feature separates "
             "in sim but drifts in real → down-weight or drop** (the 140–200 Hz shape-feature risk). "
             "_Real strengths are confound-aware per `REAL_FEATURE_ANALYSIS.md`; treat as indicative "
             "(one session per class)._\n")
    tr = sc[pd.notna(sc.transfer_delta)].sort_values("transfer_delta", ascending=False)
    L.append("**Largest sim→real drift (transfer-risk):**\n")
    L.append(tbl(tr.head(12), ["feature", "class_sep", "transfer_delta", "gbt_imp"], {
        "feature": lambda r: f"`{r.feature}`", "class_sep": lambda r: f"{r.class_sep:.3f}",
        "transfer_delta": lambda r: f"{r.transfer_delta:.3f}", "gbt_imp": lambda r: f"{r.gbt_imp:.4f}"}))
    L.append("\n**Best transfer (separate AND stable sim→real) — trust these most:**\n")
    tg = tr[tr.class_sep >= 0.4].sort_values("transfer_delta").head(12)
    L.append(tbl(tg, ["feature", "class_sep", "transfer_delta"], {
        "feature": lambda r: f"`{r.feature}`", "class_sep": lambda r: f"{r.class_sep:.3f}",
        "transfer_delta": lambda r: f"{r.transfer_delta:.3f}"}))
    L.append("")

    # 7. recommended set
    keep = sc[sc.keep].sort_values("class_sep", ascending=False)
    drop = sc[~sc.keep]
    L.append("## 7. Recommended feature set\n")
    L.append(f"**Keep {len(keep)} / 132** = (discriminative on some head) AND (cluster representative) "
             f"AND (not a high-transfer-risk feature lacking compensating GBT value). "
             f"Dropped {len(drop)}: {int((~drop.is_rep).sum())} redundant duplicates, "
             f"{int((drop.is_rep & ~drop.discriminative).sum())} weak, "
             f"{int((drop.transfer_risk & drop.is_rep & drop.discriminative).sum())} transfer-risk.\n")
    L.append("**Kept features:**\n")
    L.append("> " + ", ".join(f"`{f}`" for f in keep.feature) + "\n")
    L.append("**Dropped — redundant (duplicate of a kept representative):**\n")
    rd = drop[~drop.is_rep]
    L.append("> " + ", ".join(f"`{r.feature}`→`{r.cluster_rep}`" for _, r in rd.iterrows()) + "\n")
    L.append("**Dropped — weak / non-discriminative:**\n")
    wk = drop[drop.is_rep & ~drop.discriminative]
    L.append("> " + (", ".join(f"`{f}`" for f in wk.feature) or "none") + "\n")

    L.append("## 8. Architecture implication\n")
    L.append(f"Kept input dimension = **{len(keep)}**. For a feature-input SNN (Config A) that sets "
             "layer-1 width: at int8, layer-1 = `kept × hidden` weights. Final width/depth and whether "
             "to keep Config A vs a raw conv-SNN (Config B) depend on the confirmed edge target "
             "(ESP32-WROOM-32E vs ESP8266/ESP-12E — pending) and the timestep count T. Vehicle "
             "multiplicity should be a temporal/counting head, not a per-window class.\n")
    L.append("---\n_Generated by `analyze_v2.py` from the screening outputs; full tables in "
             "`feature_analysis_v2.sqlite` (synth_auc_global, synth_auc_family, feature_scorecard, "
             "gbt, redundancy)._")
    open(MD, "w", encoding="utf-8").write("\n".join(L))
    print(f"kept {len(keep)}/132 | new-v2 kept {kept_new}/30 | clusters {len(clusters)} "
          f"({len(multi)} multi) | wrote {MD} + {DB}")


if __name__ == "__main__":
    main()
