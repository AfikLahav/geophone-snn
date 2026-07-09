"""Empirical detectability Pd(SNR) per class: at what per-window SNR can a realizable feature
detector actually decide 'present' vs 'nothing'? Compares to the matched-filter theoretical floor.

Positives = single-class scene windows (binned by that class's per-window SNR, incl. sub-0 dB).
Negatives = nothing-scene windows. Detector = HistGBT on the 104 model features, profile-grouped
holdout. Threshold set so false-alarm rate on negatives = Pfa (1% and 10%); Pd measured per SNR bin.
Reports SNR50/SNR90 (realizable) vs the theoretical d'-based threshold. Writes DETECTABILITY.json + plot.
"""
import os, sys, json, glob, sqlite3
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
OUT = os.path.join(HERE, "snn_v2_out"); PNG = os.path.join(OUT, "plots"); FEAT_DIR = r"G:/geophone_synth/features_v2"
con = sqlite3.connect(os.path.join(HERE, "feature_analysis_v2.sqlite"))
FEATURES = [r[0] for r in con.execute("SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]; con.close()
T = 3.0
BANDW = {"human": 70.0, "animal": 70.0, "vehicle": 20.0}   # class diagnostic-band width (Hz)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve

cols = FEATURES + ["coarse", "profile_id", "split", "human_snr", "vehicle_snr", "animal_snr"]
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))[:3]],
               ignore_index=True)
rng = np.random.default_rng(0)
noth = df[df.coarse == "nothing"]

def theo_thr(B, dprime):  # SNR_dB where matched-filter d' = dprime
    snr_pow = dprime ** 2 / (2 * T * B); return 10 * np.log10(snr_pow)

report = {"note": "realizable = HistGBT on 104 features; theoretical = matched-filter d'", "classes": {}}
plt.figure(figsize=(8, 5))
for cls, snrcol, col in [("human", "human_snr", "C0"), ("vehicle", "vehicle_snr", "C1"), ("animal", "animal_snr", "C2")]:
    pos = df[df.coarse == cls]
    n = min(120000, len(pos), len(noth))
    P = pos.sample(n, random_state=0); N = noth.sample(n, random_state=1)
    X = np.nan_to_num(np.vstack([P[FEATURES].to_numpy(np.float32), N[FEATURES].to_numpy(np.float32)]))
    yv = np.r_[np.ones(len(P)), np.zeros(len(N))]
    snr = np.r_[P[snrcol].to_numpy(float), np.full(len(N), -99.0)]
    prof = np.r_[P.profile_id.astype(str).to_numpy(), N.profile_id.astype(str).to_numpy()]
    up = np.unique(prof); hold = set(rng.choice(up, max(2, len(up) // 5), replace=False))
    te = np.isin(prof, list(hold)); tr = ~te
    clf = HistGradientBoostingClassifier(max_iter=250, random_state=0).fit(X[tr], yv[tr])
    sc = clf.predict_proba(X[te])[:, 1]
    neg = sc[(yv[te] == 0)]
    res_cls = {}
    for pfa in (0.01, 0.10):
        thr = np.quantile(neg, 1 - pfa)                       # threshold giving this false-alarm rate
        ps = (yv[te] == 1); s_pos = sc[ps]; snr_pos = snr[te][ps]
        bins = np.arange(-30, 36, 3); cen = (bins[:-1] + bins[1:]) / 2
        pd_bin = []
        for b in range(len(bins) - 1):
            m = (snr_pos >= bins[b]) & (snr_pos < bins[b + 1])
            pd_bin.append((s_pos[m] >= thr).mean() if m.sum() >= 30 else np.nan)
        pd_bin = np.array(pd_bin)
        def crossing(target):
            ok = np.where(pd_bin >= target)[0]
            return float(cen[ok[0]]) if len(ok) else None
        res_cls[f"pfa_{int(pfa*100)}pct"] = {"SNR50_dB": crossing(0.5), "SNR90_dB": crossing(0.9)}
        if pfa == 0.01:
            plt.plot(cen, pd_bin, col + "-o", ms=3, label=f"{cls} (Pfa 1%)")
        else:
            plt.plot(cen, pd_bin, col + "--", lw=1, label=f"{cls} (Pfa 10%)")
    res_cls["theoretical_matchedfilter"] = {"SNR90_dB": round(theo_thr(BANDW[cls], 3.6), 1),
                                            "SNR50_dB": round(theo_thr(BANDW[cls], 2.33), 1)}
    report["classes"][cls] = res_cls
    print(f"{cls:8s} realizable SNR50/90 @1%FAR: {res_cls['pfa_1pct']['SNR50_dB']}/{res_cls['pfa_1pct']['SNR90_dB']} dB"
          f" | @10%FAR: {res_cls['pfa_10pct']['SNR50_dB']}/{res_cls['pfa_10pct']['SNR90_dB']} dB"
          f" | theoretical 50/90: {res_cls['theoretical_matchedfilter']['SNR50_dB']}/{res_cls['theoretical_matchedfilter']['SNR90_dB']} dB", flush=True)
plt.axhline(0.9, color="k", ls=":", lw=.6); plt.axhline(0.5, color="k", ls=":", lw=.6)
plt.axvline(0, color="r", ls="--", lw=.8, label="current 0 dB cutoff")
plt.xlabel("per-window SNR (dB)"); plt.ylabel("detection prob Pd"); plt.legend(fontsize=7)
plt.title("Realizable detectability Pd(SNR) per class"); plt.tight_layout()
plt.savefig(os.path.join(PNG, "80_detectability_curve.png"), dpi=120); plt.close()
json.dump(report, open(os.path.join(OUT, "DETECTABILITY.json"), "w"), indent=1)
print("\nCurrent label cutoff = 0 dB. Realizable floors above show how much lower 'detectable' actually goes.")
print("wrote DETECTABILITY.json + 80_detectability_curve.png")
