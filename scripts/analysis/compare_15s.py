"""
Compare the new 15 s AD620-chain recording to the inherited CSVs and synthetic 'nothing'.
Robust comparison = SHAPE not amplitude (3 unknown gains). Plots -> compare_15s/.

Step 0  quiet-check the 15 s (don't assume).
Layer A  spectral: normalized PSD (shape), absolute PSD (scale), mains lines, colored-noise slope,
         waveform snippets, amplitude histograms.
Layer B  104-feature PCA placing the 15 s vs real-nothing vs synth-nothing (scale-caveated).
"""
import os, sys, glob, sqlite3, json, warnings
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(ROOT, "simgeo"))
OUT = os.path.join(ROOT, "compare_15s"); os.makedirs(OUT, exist_ok=True)
RD = os.path.join(ROOT, "Goephone-Project", "geophone_data")
FS = 1000
def fig(name): plt.tight_layout(); plt.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight"); plt.close()

# ---------------- load sources (volts / native units; NOT cross-comparable in absolute) ----------------
new_csv = sorted(glob.glob(os.path.join(ROOT, "geophone_2026*.csv")))[-1]
print("new recording:", os.path.basename(new_csv))
d = pd.read_csv(new_csv)
t = d["time_s"].to_numpy(float); a = d["amplitude"].to_numpy(float)
tu = np.arange(0, t[-1], 1.0 / FS)                 # resample 995->1000 Hz uniform
new = np.interp(tu, t, a)
def loadcsv(fn):
    return pd.read_csv(os.path.join(RD, fn))["amplitude"].to_numpy(float)
hn = loadcsv("human_nothing.csv"); cn = loadcsv("car_nothing.csv")
hum = loadcsv("human.csv"); car = loadcsv("car.csv")

# ---------------- synthetic 'nothing' waveforms from dataset_v2 ----------------
synth_waves = []
try:
    sh = sorted(glob.glob(os.path.join(_GEO_ROOT, "dataset_v2/shard_*.sqlite")))[0]
    con = sqlite3.connect(sh)
    rows = con.execute("SELECT s.scene_id, s.fs, w.n_samples, w.clean_mv, w.noise_mv "
                       "FROM scenes s JOIN waveforms w ON s.scene_id=w.scene_id "
                       "WHERE s.coarse='nothing' LIMIT 24").fetchall()
    for sid, fs_s, n, cmv, nmv in rows:
        noise = np.frombuffer(nmv, np.float32).astype(float)
        out = noise.copy()
        if cmv is not None:
            cl = np.frombuffer(cmv, np.float32).astype(float)
            if len(cl) == len(noise): out = out + cl
        if len(out) >= 4096 and 1e-5 < np.std(out) < 100:    # drop non-physical/corrupt scenes
            synth_waves.append((out, float(fs_s or FS)))
    con.close()
    print(f"synthetic nothing scenes kept (plausible): {len(synth_waves)}")
except Exception as e:
    print("synthetic waveform load FAILED:", e)

# ---------------- Step 0: quiet-check the 15 s ----------------
from scipy.signal import butter, sosfiltfilt, hilbert
def bp(x, lo, hi, fs=FS):
    return sosfiltfilt(butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos"), x)
env = np.abs(hilbert(bp(new - new.mean(), 8, 40)))
sta = pd.Series(env).rolling(50, min_periods=1).mean().to_numpy()
lta = pd.Series(env).rolling(1000, min_periods=1).mean().to_numpy()
ratio = sta / (lta + 1e-12)
kt = kurtosis(new - new.mean())
n_events = int(np.sum(ratio > 4))
fig0, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(tu, (new - new.mean()) * 1000, lw=0.4); ax[0].set_ylabel("mV (demeaned)")
ax[0].set_title(f"New 15 s recording — kurtosis={kt:.2f}, STA/LTA>4 samples={n_events} "
                f"({'looks quiet' if kt < 6 and n_events < 50 else 'POSSIBLE transients'})")
ax[1].plot(tu, ratio, lw=0.5, c="firebrick"); ax[1].axhline(4, ls="--", c="k", lw=0.6)
ax[1].set_ylabel("STA/LTA (8-40 Hz)"); ax[1].set_xlabel("time (s)")
fig("00_new15s_quietcheck.png")
print(f"15s quiet-check: kurtosis={kt:.2f}, STA/LTA>4 count={n_events}")

# ---------------- Layer A: spectral ----------------
def psd(x, fs=FS, nper=4096):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x - x.mean()
    f, P = welch(x, fs=fs, nperseg=min(nper, len(x) // 2))
    return f, P
# build PSD set
series = [("NEW 15s (AD620)", new, FS, "#d62728"),
          ("human_nothing", hn, FS, "#1f77b4"),
          ("car_nothing", cn, FS, "#2ca02c")]
# synthetic median PSD on a common grid
if synth_waves:
    fg, _ = psd(synth_waves[0][0], synth_waves[0][1])
    Pg = []
    for w, fs_s in synth_waves:
        f2, P2 = psd(w, fs_s)
        Pg.append(np.interp(fg, f2, P2))
    synth_med = np.median(np.stack(Pg), 0)
else:
    fg = synth_med = None

# RMS table — ALL in mV (real CSVs are volts->x1000; synth already mV). Gains still unknown -> not calibrated.
print("\n--- absolute RMS in mV (real assumed volts per Main.py; DIFFERENT unknown gains, not calibrated) ---")
for nm, x, fs, _ in series:
    print(f"  {nm:18s} RMS={np.std(x)*1000:.4f} mV")
if synth_waves: print(f"  {'synth_nothing':18s} RMS={np.mean([np.std(w) for w,_ in synth_waves]):.4f} mV")

# 01 normalized PSD (shape)
plt.figure(figsize=(9, 5))
for nm, x, fs, c in series:
    f, P = psd(x, fs); Pn = P / np.trapz(P, f)
    plt.loglog(f, Pn, label=nm, c=c, lw=1.3)
if synth_med is not None:
    Pn = synth_med / np.trapz(synth_med, fg); plt.loglog(fg, Pn, label="synth_nothing (median)", c="k", lw=1.3, ls="--")
plt.xlim(1, FS / 2); plt.xlabel("Hz"); plt.ylabel("normalized PSD (area=1)")
plt.title("Noise SHAPE comparison — area-normalized PSD (units-free)"); plt.legend(fontsize=8); plt.grid(alpha=.3, which="both")
fig("01_psd_normalized.png")

# 02 absolute PSD (scale) — ALL in mV^2/Hz (real volts -> x1e6 on PSD; synth already mV^2)
plt.figure(figsize=(9, 5))
for nm, x, fs, c in series:
    f, P = psd(x, fs); plt.loglog(f, P * 1e6, label=nm, c=c, lw=1.2)
if synth_med is not None: plt.loglog(fg, synth_med, label="synth_nothing (median)", c="k", lw=1.2, ls="--")
plt.xlim(1, FS / 2); plt.xlabel("Hz"); plt.ylabel("PSD (mV$^2$/Hz)")
plt.title("Absolute PSD in mV — LEVEL gaps reflect different UNKNOWN gains (not calibrated)")
plt.legend(fontsize=8); plt.grid(alpha=.3, which="both"); fig("02_psd_absolute.png")

# 03 mains zoom
plt.figure(figsize=(9, 5))
for nm, x, fs, c in series:
    f, P = psd(x, fs, nper=8192); Pn = P / np.trapz(P, f)
    m = (f >= 40) & (f <= 160); plt.semilogy(f[m], Pn[m], label=nm, c=c, lw=1.2)
for ln in (50, 100, 150): plt.axvline(ln, ls=":", c="gray", lw=0.8)
plt.xlabel("Hz"); plt.ylabel("normalized PSD"); plt.title("Mains lines (50/100/150 Hz) — normalized")
plt.legend(fontsize=8); plt.grid(alpha=.3); fig("03_psd_mains_zoom.png")

# 04 colored-noise slope (fit 1-20 Hz)
def slope(x, fs=FS):
    f, P = psd(x, fs); m = (f >= 1) & (f <= 20) & (P > 0)
    return np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)[0]
labels = [nm for nm, *_ in series]; slopes = [slope(x, fs) for nm, x, fs, _ in series]
cols = [c for *_, c in series]
if synth_waves: labels.append("synth_nothing"); slopes.append(np.median([slope(w, fs) for w, fs in synth_waves])); cols.append("k")
plt.figure(figsize=(7, 4)); plt.bar(labels, slopes, color=cols)
plt.ylabel("log-log slope b (P ∝ f^b), 1–20 Hz"); plt.title("Colored-noise slope"); plt.xticks(rotation=20, ha="right")
plt.axhline(0, c="gray", lw=0.6); fig("04_noise_slope.png")

# 05 waveform snippets (normalized to unit std)
plt.figure(figsize=(11, 6))
snips = series[:3] + ([("synth_nothing", synth_waves[0][0], synth_waves[0][1], "k")] if synth_waves else [])
for i, (nm, x, fs, c) in enumerate(snips):
    seg = np.asarray(x[:int(2 * fs)], float); seg = (seg - seg.mean()) / (seg.std() + 1e-12)
    plt.subplot(len(snips), 1, i + 1); plt.plot(np.arange(len(seg)) / fs, seg, c=c, lw=0.5)
    plt.ylabel(nm, fontsize=8); plt.ylim(-6, 6)
plt.xlabel("time (s)"); plt.suptitle("Waveform snippets (2 s, each normalized to unit std)")
fig("05_waveform_snippets.png")

# 06 amplitude histograms (standardized)
plt.figure(figsize=(8, 5))
for nm, x, fs, c in series:
    z = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-12)
    plt.hist(z, bins=120, range=(-6, 6), density=True, histtype="step", label=nm, color=c, lw=1.2)
if synth_waves:
    z = np.concatenate([(w - w.mean()) / (w.std() + 1e-12) for w, _ in synth_waves])
    plt.hist(z, bins=120, range=(-6, 6), density=True, histtype="step", label="synth_nothing", color="k", lw=1.2, ls="--")
plt.yscale("log"); plt.xlabel("standardized amplitude (z)"); plt.ylabel("density"); plt.title("Amplitude distribution (standardized)")
plt.legend(fontsize=8); fig("06_amplitude_hist.png")
print("Layer A plots saved.")

# ---------------- Layer B: 104-feature PCA (scale-caveated) ----------------
try:
    import features as F
    scj = json.load(open(os.path.join(ROOT, "snn_v2_out", "scaler.json")))
    FEATS = scj["features"]; mu = np.array(scj["mean"]); sd = np.array(scj["std"])
    fidx = [F.FEATURE_NAMES.index(f) for f in FEATS]
    def featurize(x, fs=FS, maxwin=120):
        x = np.array(x, dtype=np.float64)            # writable copy (pywt rejects read-only buffers)
        if fs != FS:
            tu2 = np.arange(0, len(x) / fs, 1 / FS); x = np.interp(tu2, np.arange(len(x)) / fs, x)
        feats = []
        for c0 in range(0, len(x), 30 * FS):
            seg = x[c0:c0 + 30 * FS]
            if len(seg) < F.NW: continue
            pre = F.scene_precompute(seg)
            for i0 in range(0, len(seg) - F.NW + 1, 1500):
                feats.append(F.window_features(pre, i0).astype(np.float32))
                if len(feats) >= maxwin: break
            if len(feats) >= maxwin: break
        if not feats: return np.empty((0, len(FEATS)))
        return np.nan_to_num(np.stack(feats))[:, fidx]
    Xnew = featurize(new); Xhn = featurize(hn); Xcn = featurize(cn)
    # synthetic nothing features from features_v2 (precomputed)
    pq = sorted(glob.glob(os.path.join(_GEO_ROOT, "features_v2/features_shard_*.parquet")))[0]
    sdf = pd.read_parquet(pq, columns=FEATS + ["coarse"])
    Xsyn = sdf[sdf["coarse"] == "nothing"][FEATS].to_numpy(np.float32)
    rng = np.random.default_rng(0); Xsyn = Xsyn[rng.choice(len(Xsyn), min(800, len(Xsyn)), replace=False)]
    def z(X): return np.clip((X - mu) / sd, -8, 8)
    from sklearn.decomposition import PCA
    pca = PCA(2).fit(z(Xsyn))
    groups = [("synth_nothing", Xsyn, "k", 4, .25), ("human_nothing(old)", Xhn, "#1f77b4", 18, .7),
              ("car_nothing(old)", Xcn, "#2ca02c", 18, .7), ("NEW 15s (AD620)", Xnew, "#d62728", 60, 1.0)]
    plt.figure(figsize=(8, 6))
    for nm, X, c, s, al in groups:
        if len(X) == 0: continue
        p = pca.transform(z(X)); plt.scatter(p[:, 0], p[:, 1], s=s, alpha=al, c=c, label=f"{nm} (n={len(X)})",
                                             edgecolors="none" if nm == "synth_nothing" else "k", linewidths=0.3)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(fontsize=8)
    plt.title("104-feature PCA (z-scored w/ model scaler) — where the NEW 15 s lands vs old-real & synth nothing\n"
              "(scale-sensitive: separation partly reflects the known units/domain gap)")
    fig("07_feature_pca.png")
    print(f"Layer B: NEW {len(Xnew)} win, human_nothing {len(Xhn)}, car_nothing {len(Xcn)}, synth {len(Xsyn)} -> 07_feature_pca.png")
except Exception as e:
    import traceback; print("Layer B FAILED:", e); traceback.print_exc()

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))

print("\nDONE ->", OUT)
