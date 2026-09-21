"""D gate — dataset datasheet generator (Gebru et al. 2021 'Datasheets for Datasets' applied
to a synthetic dataset). Records everything needed for (a) threshold-time prior inversion
(per-cell training mass pi_train), (b) the reviewer audit trail (declared vs realized
composition), (c) reproducibility (Q_j spans, seeds, documented couplings).

Usage: python datasheet_gen.py <labels_dir> <tag>
Writes DATASHEET_<tag>.json + DATASHEET_<tag>.md next to this script.
"""
import os, sys, glob, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_GEO_ROOT, "labels_v4/windows_3s")
TAG = sys.argv[2] if len(sys.argv) > 2 else "v4"
CLASSES = ("human", "vehicle", "animal")

# ---- DECLARED design (per dataset revision; edit when the generator config changes) ----
DECLARED = {
    "v4": {
        "seed": 20260706,
        "labels": "3 s / 1.5 s hop; presence=emission-activity (D3); per-class in-band SNR; zone3 at gates_3s",
        "subkind_weights": {"human": {"walk": 2}, "animal": {"slow_quad": 2},
                            "nothing_confuser_w": 1.8},
        "W_snr_density": "triangular(tau_lo, mode=+1.5/+7.5/+4.5 h/v/a, hi=+40/45) + 15% subfloor tail U(tau_lo-15, tau_lo)",
        "Q_rig": {"gain_log10": [2.3, 2.9], "coupling": "lowpass, anchor_fc median 53 Hz (lognormal 0.35, clip 30-150)",
                  "quantization": "none (continuous float32)", "rail_mv": 256,
                  "mains": "terrain-gated p 0.18-0.90, fixed-mV amplitude", "spur_150hz_p": 1.0,
                  "machinery_lines": "0-3 per scene (~60-70% prevalence)"},
        "documented_couplings": ["coupling_fc ~ profile stiffness (physical)",
                                 "vehicle excluded on vs<95 m/s (physical scope)",
                                 "KNOWN VIOLATION: weather draw differs for nothing subkinds (N_condition=0.218)"],
        "floors_caps": "F floors 300/reachable cell (v4 predates enforcement; violations recorded in INVARIANT_CHECK)",
    },
    "v431": {
        "seed": 20260706,
        "labels": "3 s / 1.5 s hop; presence=emission-activity (D3); per-class in-band SNR; zone3 at gates_3s",
        "subkind_weights": {"human": {"walk": 2}, "animal": {"slow_quad": 2}, "nothing_confuser_w": 1.8,
                            "note": "declared tempering (Change 4); realized B measured in INVARIANT_CHECK"},
        "weather": ("Change 1: CLASS-BLIND global tier axis. Marginal {calm .35, wind_low .20, wind_mid .15, "
                    "wind_high .10, rain_light .12, rain_heavy .08}, dealt per-cell by largest-remainder [R3] "
                    "-> identical across all classes (kills the v4 P/N-condition weather leak)."),
        "subkind_snr_offset_db": ("Change 2 [R1]: per-subkind source-level offset (car/walk/horse ref) applied "
                                  "at the snr_to_distance CALL SITE keyed on the real subkind -> fills starved loud bins."),
        "W_snr_density": "triangular(tau_lo, mode +1.5/+7.5/+4.5, hi +40/45) + 15% subfloor tail (unchanged from v4)",
        "Q_rig": {"gain_log10": "GEO_GAIN_LOG10 span (uniform-in-dB; bounds set by the amplitude pilot)",
                  "coupling": "Change 3: MIX lowpass/bump 50/50, anchor_fc median 53 Hz (lognormal 0.35, clip 30-150)",
                  "quantization": "GEO_QUANT_MIX categorical LSB {0=continuous, 0.125, 0.2, 0.25} mV (model input only)",
                  "rail_mv": "GEO_RAIL_MIX categorical {inf, 512, 256} mV (model input only; clipping = randomized nuisance)",
                  "mains": "v41 dB-above-floor, terrain-gated", "spur_150hz_p": 0.2},
        "documented_couplings": ["coupling_fc ~ profile stiffness (physical)",
                                 "vehicle excluded on vs<95 m/s (physical scope)"],
        "floors_caps": "F floors 300/reachable cell; per-subkind ceilings from subkind_ceilings_v431.json (Change 2)",
    },
}

sh = sorted(glob.glob(os.path.join(LABELS, "labels_shard_*.parquet")))
# v4.3.1 [R5]: read the class-independent Q_j axes when present (absent for v4 -> skipped).
import pyarrow.parquet as _pq

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
_avail = set(_pq.ParquetFile(sh[0]).schema.names)
Q_AXES = [c for c in ("tier", "gain_log10", "quant_lsb", "rail_mv", "coupling_form") if c in _avail]
cols = (["coarse", "subkind", "family", "noise_condition"]
        + [f"{c}_{s}" for c in CLASSES for s in ("level", "snr")] + Q_AXES)
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sh], ignore_index=True)

ds = {"tag": TAG, "n_windows": int(len(df)), "declared": DECLARED.get(TAG, {}), "realized": {}}
# pi_train: window mass per class and per subkind-within-class (the inversion table)
cm = df["coarse"].value_counts(normalize=True)
ds["realized"]["pi_class"] = {k: round(float(v), 4) for k, v in cm.items() if v > 0}
subm = {}
for cl, g in df.groupby("coarse", observed=True):
    s = g["subkind"].value_counts(normalize=True)
    subm[str(cl)] = {str(k): round(float(v), 4) for k, v in s.items() if v > 0}
ds["realized"]["pi_subkind_given_class"] = subm
# realized W per class (SNR histogram of present windows, 5 dB bins)
Wr = {}
for c in CLASSES:
    pres = df[f"{c}_level"] > 0
    snr = df.loc[pres, f"{c}_snr"].to_numpy()
    edges = np.arange(-40, 50, 5)
    h, _ = np.histogram(np.clip(snr, edges[0], edges[-1]), bins=edges)
    Wr[c] = {f"[{edges[i]:.0f},{edges[i+1]:.0f})": round(float(h[i] / max(h.sum(), 1)), 4)
             for i in range(len(h)) if h[i] > 0}
ds["realized"]["W_snr_present"] = Wr
# marginals for the neutral axes
ds["realized"]["condition_marginal"] = {str(k): round(float(v), 4)
    for k, v in df["noise_condition"].value_counts(normalize=True).items() if v > 0}
ds["realized"]["condition_given_class"] = {
    str(cl): {str(k): round(float(v), 3) for k, v in g["noise_condition"].value_counts(normalize=True).items() if v > 0}
    for cl, g in df.groupby("coarse", observed=True)}

# v4.3.1 [R5]: realized Q_j spans (gain continuous -> min/max/median; the rest categorical, inf-safe
# via astype(str)). These document the randomized sensor-chain axes for the reviewer + reproducibility.
if Q_AXES:
    def _q_summary(a):
        s = df[a]
        if a == "gain_log10":
            arr = s.to_numpy(dtype=float); f = arr[np.isfinite(arr)]
            if not len(f):
                return {}
            return {"min": round(float(f.min()), 3), "max": round(float(f.max()), 3),
                    "median": round(float(np.median(f)), 3), "span_db": round(float(20 * (f.max() - f.min())), 1)}
        vc = s.astype(str).value_counts(normalize=True)         # inf -> "inf"; categorical view
        return {str(k): round(float(v), 4) for k, v in vc.items() if v > 0}
    ds["realized"]["Q_j_spans"] = {a: _q_summary(a) for a in Q_AXES}

json.dump(ds, open(os.path.join(HERE, f"DATASHEET_{TAG}.json"), "w"), indent=1)
# human-readable md
md = [f"# Dataset datasheet — {TAG}", "", f"windows: {ds['n_windows']:,}", "", "## Declared design"]
for k, v in ds["declared"].items():
    md.append(f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
md += ["", "## Realized class mass (pi_train)", json.dumps(ds["realized"]["pi_class"]),
       "", "## Realized subkind mass within class", json.dumps(subm, indent=1),
       "", "## Realized condition | class (neutrality view)",
       json.dumps(ds["realized"]["condition_given_class"], indent=1)]
if ds["realized"].get("Q_j_spans"):
    md += ["", "## Realized Q_j sensor-DR spans (v4.3.1 Change 3)",
           json.dumps(ds["realized"]["Q_j_spans"], indent=1)]
open(os.path.join(HERE, f"DATASHEET_{TAG}.md"), "w").write("\n".join(md))
print(f"wrote DATASHEET_{TAG}.json + .md  (n={ds['n_windows']:,})")
