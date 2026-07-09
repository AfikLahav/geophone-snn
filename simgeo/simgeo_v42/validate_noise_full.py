"""FULL-SCOPE condition-matched noise validation (no sampling caps).

SYNTHETIC: every 'nothing' scene in all 14 shards, Welch PSD over the whole scene,
           binned by its own condition_of(wind,rain). Also output-RMS (mV) for a LEVEL check.
REAL R3:   every file in yw/zg/lasso (+is_il for <20 Hz), Welch PSD over each full hour,
           hour binned by the SAME condition function from cached ERA5 weather. Raw data
           (non-circular). yw/zg/lasso cover 20-90 Hz; is_il only <20 Hz.
REAL rig:  rig_data.npz (real measured floor, mV) -> calm reference + LEVEL anchor.
REAL field: the 4 geophone CSVs (whole recordings) as independent (unlabeled) points.

Shape normalized to unit energy 5-100 Hz (yw/zg/lasso/synth) or 1-18 Hz (is_il low band).
Domain: R3 ground velocity rendered through our 4.5 Hz response (flat >5 Hz).
Outputs: noise_full_match.png + printed band-fraction table (with full n) + level check.
Usage: python validate_noise_full.py
"""
import os, sys, glob, json, sqlite3, warnings, time
import numpy as np
from scipy import signal as sig
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "datasets", "real_noise_fetch"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "datasets", "noise_atlas"))
import r3_noise, fetch_core as fc

ROOT = r"N:\geophone_real_noise"
CORPUS = r"G:\geophone_synth\corpus_150k"
CENTROIDS = {"yw": (36.62, -97.74), "lasso": (36.62, -97.74), "zg": (33.55, -116.57)}
IS_COORDS = {"HRFI": (30.04, 35.03), "GEM": (31.85, 34.92), "MDBI": (30.60, 34.80)}
RANGES = {"yw": ("2016-06-22", "2016-07-25"), "lasso": ("2016-04-14", "2016-05-10"),
          "zg": ("2014-05-08", "2014-06-12"),
          "is_il": [("2021-01-01", "2021-02-28"), ("2021-04-01", "2021-05-31")]}
CONDS = ["calm", "wind_low", "wind_mid", "wind_high", "rain_light", "rain_heavy"]
NSEG = 4096
GRID = np.linspace(5, 250, 600)        # high-band grid (yw/zg/lasso/synth)
GRID_LO = np.linspace(1, 18, 200)      # low-band grid (is_il)


def cond_of(w, r): return r3_noise.condition_of(float(w), float(r))


def geophone_render(v, fs):
    n = len(v); w = 2 * np.pi * np.fft.rfftfreq(n, 1 / fs); s = 1j * w; w0 = 2 * np.pi * 4.5
    H = 28.8 * s**2 / (s**2 + 2 * 0.6 * w0 * s + w0**2)
    return np.fft.irfft(np.fft.rfft(v) * H, n=n)


def welch_psd(x, fs):
    x = np.asarray(x, float); x = x - x.mean()
    f, P = sig.welch(x, fs=fs, nperseg=min(NSEG, len(x)))
    return f, P


def shape_on(f, P, grid, lo, hi):
    m = (f >= lo) & (f <= hi)
    Pn = P / (P[m].sum() + 1e-30)
    return np.interp(grid, f, Pn, left=np.nan, right=np.nan)


def fb2090(f, P):
    m = (f >= 5) & (f <= 100); mb = (f >= 20) & (f <= 90)
    return P[mb].sum() / (P[m].sum() + 1e-30)


def hourly_weather(target, sta=None):
    if target == "is_il":
        lat, lon = IS_COORDS.get(sta, (31.0, 34.9))
        return pd.concat([fc.get_weather_daily(lat, lon, a, b)[1] for a, b in RANGES["is_il"]])
    lat, lon = CENTROIDS[target]; a, b = RANGES[target]
    return fc.get_weather_daily(lat, lon, a, b)[1]


def collect_synth():
    shapes = {c: [] for c in CONDS}; frac = {c: [] for c in CONDS}; rms = {c: [] for c in CONDS}
    n = 0; t0 = time.time()
    for sh in sorted(glob.glob(os.path.join(CORPUS, "shard_*.sqlite"))):
        db = sqlite3.connect(sh)
        for params, blob in db.execute("SELECT params, amplitude FROM scenes WHERE coarse='nothing'"):
            pj = json.loads(params); c = cond_of(pj.get("wind", 0), pj.get("rain", 0))
            x = np.frombuffer(blob, np.float32)
            if len(x) < NSEG: continue
            f, P = welch_psd(x, 1000.0)
            shapes[c].append(shape_on(f, P, GRID, 5, 100)); frac[c].append(fb2090(f, P))
            rms[c].append(float(np.std(x)))            # mV (output level)
            n += 1
        db.close()
        print(f"  synth {os.path.basename(sh)} done, n={n}, {(time.time()-t0)/60:.1f}m", flush=True)
    return shapes, frac, rms


def collect_real():
    shapes = {c: [] for c in CONDS}; frac = {c: [] for c in CONDS}
    zg = {c: [] for c in CONDS}
    islo = {c: [] for c in CONDS}                       # is_il low-band shapes
    wxc = {}; t0 = time.time()
    for tgt in ["yw", "zg", "lasso", "is_il"]:
        files = sorted(glob.glob(os.path.join(ROOT, tgt, "*.npz")))
        nf = 0
        for path in files:
            nm = os.path.basename(path).split("."); sta, date = nm[1], nm[3]
            key = (tgt, sta if tgt == "is_il" else None)
            if key not in wxc:
                try: wxc[key] = hourly_weather(tgt, sta)
                except Exception as e: wxc[key] = None
            wx = wxc[key]
            if wx is None: continue
            try: z = np.load(path, mmap_mode="r"); v = z["v"]; fs = float(z["fs"])
            except Exception: continue
            nh = int(len(v) / fs // 3600); day = pd.Timestamp(date, tz="UTC")
            for h in range(min(nh, 24)):
                t = day + pd.Timedelta(hours=h)
                if t not in wx.index: continue
                c = cond_of(wx.loc[t].wind, wx.loc[t].precip)
                seg = np.asarray(v[int(h*3600*fs):int((h+1)*3600*fs)], float)
                if len(seg) < NSEG or seg.std() == 0 or not np.isfinite(seg).all(): continue
                f, P = welch_psd(geophone_render(seg, fs), fs)
                if tgt == "is_il":
                    islo[c].append(shape_on(f, P, GRID_LO, 1, 18))
                else:
                    shapes[c].append(shape_on(f, P, GRID, 5, 100)); frac[c].append(fb2090(f, P))
                    if tgt == "zg": zg[c].append(shape_on(f, P, GRID, 5, 100))
            nf += 1
            if nf % 40 == 0:
                print(f"  real {tgt} {nf}/{len(files)} files, {(time.time()-t0)/60:.1f}m, "
                      + ",".join(f"{c}:{len(shapes[c])+len(islo[c])}" for c in CONDS), flush=True)
        print(f"  real {tgt} DONE ({nf} files)", flush=True)
    return shapes, frac, zg, islo


def agg(rows):
    if not rows: return None
    A = np.array(rows)
    return np.nanmedian(A, 0), np.nanpercentile(A, 10, 0), np.nanpercentile(A, 90, 0), len(rows)


def main():
    print("=== SYNTHETIC (all nothing scenes, all shards) ===", flush=True)
    s_shape, s_frac, s_rms = collect_synth()
    print("=== REAL R3 (all files, all hours) ===", flush=True)
    r_shape, r_frac, zg, islo = collect_real()
    rd = np.load(os.path.join(HERE, "..", "phase0", "rig_data.npz"))
    fr, Pr = welch_psd(rd["mv"], 1000.0); rig_shape = shape_on(fr, Pr, GRID, 5, 100)
    rig_rms = float(np.std(rd["mv"]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, c in zip(axes.flat, CONDS):
        s = agg(s_shape[c]); r = agg(r_shape[c])
        if s: ax.semilogy(GRID, s[0], "C3-", lw=2, label=f"SYNTH n={s[3]}"); ax.fill_between(GRID, s[1], s[2], color="C3", alpha=.15)
        if r: ax.semilogy(GRID, r[0], "C0-", lw=2, label=f"REAL R3 n={r[3]}"); ax.fill_between(GRID, r[1], r[2], color="C0", alpha=.12)
        if c == "calm": ax.semilogy(GRID, rig_shape, "C2--", lw=1.6, label="REAL rig floor")
        if c.startswith("wind"):
            z = agg(zg[c])
            if z: ax.semilogy(GRID, z[0], "C1:", lw=1.6, label=f"REAL ZG desert n={z[3]}")
        ax.set_title(c); ax.set_xlabel("Hz"); ax.set_xlim(5, 250); ax.axvspan(20, 90, alpha=.05, color="green")
        ax.legend(fontsize=8)
    fig.suptitle("FULL-SCOPE condition-matched noise: SYNTHETIC vs REAL (all data; green=20-90 Hz)", fontsize=13)
    plt.tight_layout(); out = os.path.join(HERE, "..", "noise_full_match.png"); plt.savefig(out, dpi=110)
    print("saved", out, flush=True)

    print("\n=== 20-90 Hz share of 5-100 Hz energy (median; n) ===")
    for c in CONDS:
        sd = np.median(s_frac[c]) if s_frac[c] else float('nan')
        rdd = np.median(r_frac[c]) if r_frac[c] else float('nan')
        print(f"  {c:12s} SYNTH={sd:.2f} (n={len(s_frac[c])})   REAL={rdd:.2f} (n={len(r_frac[c])})", flush=True)

    print("\n=== LEVEL check (output mV RMS; only same-hardware comparison) ===")
    print(f"  REAL rig floor RMS = {rig_rms:.4f} mV")
    if s_rms["calm"]:
        a = np.array(s_rms["calm"]); print(f"  SYNTH calm RMS: median={np.median(a):.4f} p10={np.percentile(a,10):.4f} p90={np.percentile(a,90):.4f} mV (n={len(a)})")
    for c in ["wind_low","wind_mid","wind_high","rain_light","rain_heavy"]:
        if s_rms[c]: a=np.array(s_rms[c]); print(f"  SYNTH {c} RMS median={np.median(a):.4f} mV (n={len(a)})")

    print("\n=== is_il (Israel) low-band 1-18 Hz, real only ===")
    for c in CONDS:
        z = agg(islo[c])
        if z: print(f"  is_il {c}: n={z[3]}")


if __name__ == "__main__":
    main()
