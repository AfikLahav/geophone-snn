"""Condition-matched noise validation: synthetic vs REAL, like-for-like by weather.

  calm        -> synthetic-calm output  vs  rig_data.npz (real measured floor, our hardware)
  wind_*/rain -> synthetic output (binned by its own wind/rain) vs RAW R3 windows binned by
                 the SAME condition function from cached ERA5 weather (non-circular: raw
                 windows, not the fitted medians the synth was generated from).

Domain match: R3 is ground velocity at native fs (yw250/zg500/lasso500); apply our 4.5 Hz
geophone response (flat >5 Hz) so shapes compare to the synthetic mV output. Each PSD
normalized to unit energy over 5-100 Hz (response-flat, all sensors cover it). Real curves
plotted only to their Nyquist. ZG (desert, Israel-like) broken out for wind.

Usage: python validate_noise_conditions.py
"""
import os, sys, glob, json, sqlite3, warnings
import numpy as np
from scipy import signal as sig
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "datasets", "real_noise_fetch"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "datasets", "noise_atlas"))
import r3_noise
import fetch_core as fc
import pandas as pd

ROOT = r"N:\geophone_real_noise"
CENTROIDS = {"yw": (36.62, -97.74), "lasso": (36.62, -97.74), "zg": (33.55, -116.57)}
IS_COORDS = {"HRFI": (30.04, 35.03), "GEM": (31.85, 34.92), "MDBI": (30.60, 34.80)}
RANGES = {"yw": ("2016-06-22", "2016-07-25"), "lasso": ("2016-04-14", "2016-05-10"),
          "zg": ("2014-05-08", "2014-06-12"),
          "is_il": [("2021-01-01", "2021-02-28"), ("2021-04-01", "2021-05-31")]}
CONDS = ["calm", "wind_low", "wind_mid", "wind_high", "rain_light", "rain_heavy"]
NSEG = 4096
WINLEN = 8192
CAP = 140
rng = np.random.default_rng(7)


def cond_of(wind, rain):
    return r3_noise.condition_of(float(wind), float(rain))


def geophone_render(v, fs):
    """Apply our 4.5 Hz geophone response to real ground velocity -> output shape."""
    n = len(v)
    w = 2 * np.pi * np.fft.rfftfreq(n, 1 / fs)
    s = 1j * w; w0 = 2 * np.pi * 4.5
    H = 28.8 * s**2 / (s**2 + 2 * 0.6 * w0 * s + w0**2)
    return np.fft.irfft(np.fft.rfft(v) * H, n=n)


def norm_psd(x, fs):
    x = np.asarray(x, float); x = x - x.mean()
    f, P = sig.welch(x, fs=fs, nperseg=min(NSEG, len(x)))
    m = (f >= 5) & (f <= 100)
    return f, P / (P[m].sum() + 1e-30)


def hourly_weather(target, sta=None):
    if target == "is_il":
        lat, lon = IS_COORDS.get(sta, (31.0, 34.9))
        return pd.concat([fc.get_weather_daily(lat, lon, a, b)[1] for a, b in RANGES["is_il"]])
    lat, lon = CENTROIDS[target]; a, b = RANGES[target]
    return fc.get_weather_daily(lat, lon, a, b)[1]


def collect_real(targets):
    """{cond: [psd,...]} and {('zg',cond):[...]} from raw R3 windows, weather-binned."""
    out = {c: [] for c in CONDS}
    zg = {c: [] for c in CONDS}
    wxc = {}
    for tgt in targets:
        files = sorted(glob.glob(os.path.join(ROOT, tgt, "*.npz")))
        rng.shuffle(files)
        for path in files[:10]:
            name = os.path.basename(path).split(".")
            sta, date = name[1], name[3]
            key = (tgt, sta if tgt == "is_il" else None)
            if key not in wxc:
                try: wxc[key] = hourly_weather(tgt, sta)
                except Exception as e: print(f"  wx fail {key}: {e}", flush=True); wxc[key] = None
            wx = wxc[key]
            if wx is None: continue
            try: z = np.load(path, mmap_mode="r"); v = z["v"]; fs = float(z["fs"])
            except Exception: continue
            nh = int(len(v) / fs // 3600)
            day = pd.Timestamp(date, tz="UTC")
            for h in range(min(nh, 24)):
                t = day + pd.Timedelta(hours=h)
                if t not in wx.index: continue
                row = wx.loc[t]
                c = cond_of(row.wind, row.precip)
                if len(out[c]) >= CAP and not (tgt == "zg" and len(zg[c]) < CAP // 2): continue
                base = int(h * 3600 * fs)
                for k in range(3):                                   # 3 windows/hour
                    i0 = base + int((k + 0.5) * 1200 * fs)
                    seg = np.asarray(v[i0:i0 + WINLEN], float)
                    if len(seg) < WINLEN or seg.std() == 0 or not np.isfinite(seg).all(): continue
                    f, P = norm_psd(geophone_render(seg, fs), fs)
                    if len(out[c]) < CAP: out[c].append((f, P))
                    if tgt == "zg" and len(zg[c]) < CAP // 2: zg[c].append((f, P))
        print(f"  real {tgt} done: " + ", ".join(f"{c}:{len(out[c])}" for c in CONDS if out[c]), flush=True)
    return out, zg


def collect_synth():
    """{cond: [psd,...]} from synthetic nothing scenes, binned by their own wind/rain."""
    out = {c: [] for c in CONDS}
    shards = sorted(glob.glob(r"G:\geophone_synth\corpus_150k\shard_*.sqlite"))[:4]
    for sh in shards:
        db = sqlite3.connect(sh)
        for params, blob in db.execute("SELECT params, amplitude FROM scenes WHERE coarse='nothing'"):
            if all(len(out[c]) >= CAP for c in CONDS): break
            pj = json.loads(params); c = cond_of(pj.get("wind", 0), pj.get("rain", 0))
            if len(out[c]) >= CAP: continue
            x = np.frombuffer(blob, np.float32)
            if len(x) < WINLEN: continue
            i0 = rng.integers(0, len(x) - WINLEN)
            f, P = norm_psd(x[i0:i0 + WINLEN], 1000.0)
            out[c].append((f, P))
        db.close()
        print(f"  synth {os.path.basename(sh)}: " + ", ".join(f"{c}:{len(out[c])}" for c in CONDS if out[c]), flush=True)
    return out


def agg(curves, grid):
    """median + p10/p90 of a list of (f,P) onto a common freq grid."""
    if not curves: return None
    Ps = [np.interp(grid, f, P, left=np.nan, right=np.nan) for f, P in curves]
    A = np.array(Ps)
    return (np.nanmedian(A, 0), np.nanpercentile(A, 10, 0), np.nanpercentile(A, 90, 0), len(curves))


def main():
    print("collecting synthetic ...", flush=True)
    syn = collect_synth()
    print("collecting real R3 (raw windows, weather-binned) ...", flush=True)
    real, zg = collect_real(["yw", "zg", "lasso"])
    # rig floor (real calm)
    rd = np.load(os.path.join(HERE, "..", "phase0", "rig_data.npz"))
    fr, Pr = norm_psd(rd["mv"], 1000.0)

    grid = np.linspace(5, 250, 500)
    panels = ["calm", "wind_low", "wind_mid", "wind_high", "rain_light", "rain_heavy"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, c in zip(axes.flat, panels):
        s = agg(syn[c], grid); r = agg(real[c], grid)
        if s:
            ax.semilogy(grid, s[0], "C3-", lw=2, label=f"SYNTH (n={s[3]})")
            ax.fill_between(grid, s[1], s[2], color="C3", alpha=0.15)
        if r:
            ax.semilogy(grid, r[0], "C0-", lw=2, label=f"REAL R3 (n={r[3]})")
            ax.fill_between(grid, r[1], r[2], color="C0", alpha=0.15)
        if c == "calm":
            ax.semilogy(fr, Pr, "C2--", lw=1.6, label="REAL rig floor")
        if c.startswith("wind") and zg[c]:
            z = agg(zg[c], grid)
            if z: ax.semilogy(grid, z[0], "C1:", lw=1.6, label=f"REAL ZG desert (n={z[3]})")
        ax.set_title(c); ax.set_xlabel("Hz"); ax.set_xlim(5, 250)
        ax.axvspan(20, 90, alpha=0.05, color="green")          # footstep band
        ax.legend(fontsize=8)
    axes.flat[0].set_ylabel("normalized PSD"); axes.flat[3].set_ylabel("normalized PSD")
    fig.suptitle("Condition-matched noise: SYNTHETIC vs REAL  (normalized 5-100 Hz; green=footstep band 20-90)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(HERE, "..", "noise_condition_match.png")
    plt.savefig(out, dpi=110); print("saved", out, flush=True)
    # numeric: footstep-band (20-90) energy fraction per condition
    print("\n=== fraction of 5-100 Hz energy in the 20-90 Hz footstep band ===")
    print(f"{'cond':12s} {'SYNTH':>8s} {'REAL':>8s} {'ZG-desert':>10s}")
    for c in panels:
        def fb(curves):
            if not curves: return float("nan")
            vals = []
            for f, P in curves:
                m = (f >= 5) & (f <= 100); mb = (f >= 20) & (f <= 90)
                vals.append(P[mb].sum() / (P[m].sum() + 1e-30))
            return np.median(vals)
        print(f"{c:12s} {fb(syn[c]):8.2f} {fb(real[c]):8.2f} {fb(zg[c]):10.2f}")


if __name__ == "__main__":
    main()
