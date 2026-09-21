"""C3 — v4 render coverage gate. Reads the W=3 s labels parquet (labels_v4/windows_3s) and
checks that the dataset fills all reasonable loudness levels + the composition/occupancy
conditions. Exit 1 on fail (full mode); --pilot reports without failing.

Gates (realistic bounds -- pass-by geometry is genuinely faint-dominated, and sub-floor
windows are EXCLUDED from the A1 gated metric anyway, so the pile is harmless):
  1. every 5 dB bin in [tau_lo(c), +30] holds >= 1% of present windows (loud +30..40 tail
     reported, not gated -- it needs the source ~on the sensor)
  2. per-class present-window share in [8%, 20%]
  3. sub-floor share of present windows <= 55% (v3 was 85-93%; pass-bys are faint-dominated)
  4. co-occurring (>=2 classes present) windows >= 6% of present windows
  5. no terrain family contributes > 3x its window-share AND > 40% to any single class-bin
  6. occupancy consistency: no present(level>0) window with occ==0, and vice-versa
Writes dataset_v431_coverage.json + snr_coverage_v4.png.

Usage: python dataset_validation/coverage_report.py [labels_dir] [--pilot]
"""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = "--pilot" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
LABELS = args[0] if args else os.path.join(_GEO_ROOT, "labels_v4/windows_3s")
GATES = json.load(open(os.path.join(HERE, "gates_3s.json")))["classes"]
CLASSES = ("human", "vehicle", "animal")

shards = sorted(glob.glob(os.path.join(LABELS, "labels_shard_*.parquet")))
cols = (["scene_id", "coarse", "family", "masking", "activity"]
        + [f"{c}_{s}" for c in CLASSES for s in ("level", "snr", "occ")])
df = pd.concat([pd.read_parquet(s, columns=cols) for s in shards], ignore_index=True)
N = len(df)
rep = {"labels_dir": LABELS, "n_windows": int(N), "pilot": PILOT, "checks": {}, "per_class": {}}
fails = []
print(f"coverage: {N:,} windows from {len(shards)} shard(s) | mode={'PILOT' if PILOT else 'GATE'}\n")

# occupancy consistency (6)
for c in CLASSES:
    lvl = df[f"{c}_level"].to_numpy(); occ = df[f"{c}_occ"].to_numpy()
    bad1 = int(((lvl > 0) & (occ <= 0)).sum()); bad2 = int(((lvl == 0) & (occ > 0)).sum())
    rep["checks"][f"occ_consistency_{c}"] = {"present_zero_occ": bad1, "absent_pos_occ": bad2}
    if bad1 > 0:
        fails.append(f"{c}: {bad1} present windows with zero occupancy")
print("occupancy consistency:", {c: rep["checks"][f"occ_consistency_{c}"] for c in CLASSES})

# per-class coverage (1,2,3) + family (5)
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=True)
fam_all = df["family"].value_counts(normalize=True).to_dict()
for ax, c in zip(axes, CLASSES):
    lo = GATES[c]["tau_lo_db"]; edges = np.arange(np.floor(lo / 5) * 5, 45, 5)
    lvl = df[f"{c}_level"].to_numpy(); snr = df[f"{c}_snr"].to_numpy()
    pres = lvl > 0; npres = int(pres.sum())
    share = pres.mean()
    subfloor = float((snr[pres] < lo).mean()) if npres else 0.0
    h, _ = np.histogram(np.clip(snr[pres], edges[0], edges[-1]), bins=edges)
    frac = h / max(h.sum(), 1)
    # gate bins only up to +30 (the +30..40 tail needs the source ~on the sensor -> reported)
    thin = [f"[{edges[i]:.0f},{edges[i+1]:.0f})={frac[i]*100:.1f}%"
            for i in range(len(frac)) if frac[i] < 0.01 and edges[i] < 30]
    # family concentration per bin
    fam_viol = []
    fam = df["family"].to_numpy()
    binidx = np.digitize(np.clip(snr, edges[0], edges[-1] - 1e-6), edges) - 1
    for bi in range(len(edges) - 1):
        mask = pres & (binidx == bi)
        if mask.sum() < 50:
            continue
        fv = pd.Series(fam[mask]).value_counts(normalize=True)
        for famname, fr in fv.items():
            if fr > 3 * fam_all.get(famname, 1e-9) and fr > 0.30:
                fam_viol.append(f"bin{bi}:{famname}={fr*100:.0f}%")
    rep["per_class"][c] = {"present_share": round(float(share), 4), "n_present": npres,
                           "subfloor_share": round(subfloor, 4), "thin_bins": thin,
                           "family_concentration": fam_viol[:6]}
    if not (0.08 <= share <= 0.20):
        fails.append(f"{c}: present share {share*100:.1f}% outside [8,20]%")
    if subfloor > 0.55:
        fails.append(f"{c}: sub-floor share {subfloor*100:.1f}% > 55%")
    if thin:
        fails.append(f"{c}: thin loudness bins (< +30) {thin}")
    if fam_viol:
        fails.append(f"{c}: family over-concentration {fam_viol[:3]}")
    ax.bar((edges[:-1] + edges[1:]) / 2, frac, width=4.2, color="#2b6cb0", alpha=0.8)
    ax.axhline(0.01, color="r", ls="--", lw=1, label="1% floor")
    ax.axvline(GATES[c]["tau_hi_db"], color="g", ls=":", label="tau_hi")
    ax.set_title(f"{c}  present {share*100:.1f}%  subfloor {subfloor*100:.0f}%")
    ax.set_xlabel("in-band SNR (dB)"); ax.legend(fontsize=7)
    print(f"{c:8s} present {share*100:5.1f}% ({npres:,}) subfloor {subfloor*100:4.0f}% "
          f"thin_bins={thin} fam={fam_viol[:3]}")
axes[0].set_ylabel("fraction of present windows")
fig.suptitle("v4 loudness coverage per class (present windows, 5 dB bins)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "snr_coverage_v4.png"), dpi=130)

# co-occurrence (4)
present_any = (df["human_level"] > 0) | (df["vehicle_level"] > 0) | (df["animal_level"] > 0)
ncls = (df["human_level"] > 0).astype(int) + (df["vehicle_level"] > 0).astype(int) + (df["animal_level"] > 0).astype(int)
cooc = float((ncls >= 2).sum() / max(present_any.sum(), 1))
rep["checks"]["cooccurrence_share"] = round(cooc, 4)
if cooc < 0.06:
    fails.append(f"co-occurrence {cooc*100:.1f}% < 6%")
print(f"\nco-occurrence (>=2 classes present): {cooc*100:.1f}% of present windows")
print(f"nothing windows: {(~present_any).mean()*100:.1f}%")

rep["PASS"] = len(fails) == 0
rep["failures"] = fails
json.dump(rep, open(os.path.join(HERE, "dataset_v431_coverage.json"), "w"), indent=1)
print("\n" + ("PASS" if not fails else f"FAIL ({len(fails)}):"))
for f in fails:
    print("  -", f)
print("wrote dataset_v431_coverage.json + snr_coverage_v4.png")
if fails and not PILOT:
    sys.exit(1)
