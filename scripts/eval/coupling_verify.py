"""C0 — coupling-resonance verification on the REAL recordings (read-only).

The sim centers the ground-coupling resonance per profile (~132 Hz typical draw); the
real rig was previously estimated near ~66 Hz (human-session ring-down). Measure it:
Welch PSD per file (quiet segments for nothing/field files; ALL segments for active
sessions — impulses EXCITE the resonance), suppress narrowband lines with a running
median filter, remove a log-log power-law baseline, and locate the dominant BROAD
residual bump (width >= 6 Hz) in the 25-180 Hz coupling band. Writes
coupling_verify.json + coupling_verify.png next to this script.

Sanctioned analysis use of the real CSVs per the approved v4 plan (C0). No training use.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.signal import welch, medfilt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FS = 1000.0
BAND = (25.0, 180.0)
NPER = 4096
MIN_WIDTH_HZ = 6.0                       # coupling Q<=20 at >=40 Hz -> width >= ~3-6 Hz; lines are ~1 Hz
MED_BINS = 21                            # ~5 Hz running median kills lines, keeps broad bumps

GD = os.path.join(ROOT, "Goephone-Project", "geophone_data")
FILES = ([(p, "quiet") for p in sorted(glob.glob(os.path.join(GD, "*_nothing.csv")))]
         + [(os.path.join(GD, n), "active") for n in ("human.csv", "car.csv")]
         + [(p, "quiet") for p in sorted(glob.glob(os.path.join(ROOT, "geophone_2026*.csv")))])


def file_psd(path, mode):
    a = pd.read_csv(path)["amplitude"].to_numpy(float) * 1000.0   # V -> mV
    a = a[np.isfinite(a)]
    chunk = 2000                                                  # 2 s segments
    segs = [a[i:i + chunk] for i in range(0, len(a) - chunk + 1, chunk)]
    if len(segs) < 3:
        return None, None
    if mode == "quiet":
        rms = np.array([np.std(s) for s in segs])
        segs = [s for s, r in zip(segs, rms) if r <= np.quantile(rms, 0.6)]
        if len(segs) < 3:
            return None, None
    Ps = []
    for s in segs:
        f, P = welch(s - s.mean(), fs=FS, nperseg=min(NPER, len(s)))
        Ps.append(P)
    return f, np.median(np.stack(Ps), 0)


def broad_bump(f, P):
    """Line-suppressed baseline-removed residual peak with width gate."""
    m = (f >= BAND[0]) & (f <= BAND[1])
    fm = f[m]
    lp = 10 * np.log10(medfilt(P, MED_BINS)[m] + 1e-30)           # running-median line kill
    coef = np.polyfit(np.log10(fm), lp, 1)
    rs = lp - np.polyval(coef, np.log10(fm))
    order = np.argsort(rs)[::-1]
    for i in order[:20]:                                          # best peak passing the width gate
        half = rs[i] / 2
        if half <= 0.5:                                           # <1 dB bump: nothing credible
            break
        lo = i
        while lo > 0 and rs[lo] > half: lo -= 1
        hi = i
        while hi < len(rs) - 1 and rs[hi] > half: hi += 1
        if fm[hi] - fm[lo] >= MIN_WIDTH_HZ:
            return float(fm[i]), float(rs[i]), (float(fm[lo]), float(fm[hi])), (fm, rs)
    return None, None, None, (fm, rs)


results = {"band_hz": BAND, "min_width_hz": MIN_WIDTH_HZ, "files": {}}
plt.figure(figsize=(11, 6))
peaks = []
for path, mode in FILES:
    name = os.path.basename(path)
    if not os.path.exists(path):
        continue
    f, P = file_psd(path, mode)
    if f is None:
        results["files"][name] = None
        print(f"{name:36s} SKIP (too short)")
        continue
    fc, amp_db, width, (fm, rs) = broad_bump(f, P)
    if fc is None:
        results["files"][name] = {"peak_hz": None, "mode": mode}
        print(f"{name:36s} [{mode:6s}] no broad bump >= {MIN_WIDTH_HZ:.0f} Hz wide")
        plt.plot(fm, rs, lw=0.8, alpha=0.4, ls=":")
        continue
    results["files"][name] = {"peak_hz": round(fc, 1), "residual_db": round(amp_db, 1),
                              "halfwidth_hz": [round(width[0], 1), round(width[1], 1)], "mode": mode}
    peaks.append((fc, amp_db, mode))
    plt.plot(fm, rs, lw=1.2, alpha=0.85, label=f"{name} [{mode}] ({fc:.0f} Hz, +{amp_db:.1f} dB)")
    print(f"{name:36s} [{mode:6s}] bump {fc:6.1f} Hz  +{amp_db:4.1f} dB  width {width[0]:.0f}-{width[1]:.0f} Hz")

if peaks:
    fcs = np.array([p[0] for p in peaks]); amps = np.array([p[1] for p in peaks])
    w = np.clip(amps, 0.5, None)
    anchor = float(np.exp(np.average(np.log(fcs), weights=w)))
    results["FC_ANCHOR_HZ"] = round(anchor, 1)
    results["spread_hz"] = [round(float(fcs.min()), 1), round(float(fcs.max()), 1)]
    results["n_files"] = len(peaks)
    plt.axvline(anchor, color="k", ls="--", lw=2, label=f"anchor (weighted geo-mean) {anchor:.0f} Hz")
print(f"\nFC_ANCHOR_HZ = {results.get('FC_ANCHOR_HZ')}  spread {results.get('spread_hz')} "
      f"from {results.get('n_files', 0)} file(s)")
plt.xlabel("Hz"); plt.ylabel("PSD residual above power-law baseline (dB, line-suppressed)")
plt.title("Coupling-bump search on real recordings (median-filtered, baseline-removed)")
plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "coupling_verify.png"), dpi=130)
json.dump(results, open(os.path.join(HERE, "coupling_verify.json"), "w"), indent=1)
print("wrote coupling_verify.json + coupling_verify.png")
