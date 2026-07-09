"""Feature bank v1 — 102 per-window features for the full-corpus screening pass.

Groups (order = FEATURE_NAMES order):
  LEGACY32   the notebook's 32 verbatim (site-tuned bands kept deliberately so their
             cross-terrain transfer is MEASURED, not asserted; band filters applied
             scene-wide then sliced — identical except edge transients)
  PHYS7      physics-band energies/fractions (wind 1-5, vehicle 5-25, footstep 20-90,
             high 90-180 Hz) — label.py band convention; fractions are distance-robust
  CAD12      cadence/gait modulation domain (envelope spectrum 0.3-12 Hz): cadence freq
             + salience, beat structure 2f0/4f0 (biped-vs-quadruped, Park & Dibazar),
             footfall rate, modulation-peak count + envelope entropy (single-vs-multiple
             candidates), inter-impulse intervals, gust AM
  ENG8       tonality/engine: harmonic comb 25-58 Hz orders 1-3, line count/stability,
             12-15 Hz wheel hop, engine-band beat depth (convoy cue), site-line monitor
             (44-47/50/55/100/150 Hz EXCLUDED from comb/lines, measured separately)
  WX2        weather gates: rain-band impulse rate, high-band kurtosis
  CEP16      mfcc_1..8 + lfcc_1..8 (24-filter banks, 1-200 Hz; mel ~ linear down here)
  WPE16      db4 wavelet-packet level-4 leaf energy fractions
  AR8        AR(8) coefficients (Yule-Walker/Levinson on the window autocorrelation)
  NL1        Higuchi fractal dimension (kmax=8)

Window = 3.0 s / 3000 samples @ 1 kHz (matches the corpus label windows).
"""
import numpy as np
from scipy import signal as sig
from scipy import stats as spstats
from scipy.fft import dct
import pywt

FS = 1000.0
NW = 3000
NFFT_ENV = 8192                       # envelope spectrum zero-pad (0.122 Hz resolution)

# ---------------------------------------------------------------- filter banks
LEGACY_BANDS = {
    "LOW_FREQ": (20, 30), "CAR_APPROACH": (30, 34), "CAR_PEAK": (34, 40),
    "CAR_TAIL": (40, 48), "MID_GAP": (48, 60), "HUMAN_PEAK": (60, 70),
    "HUMAN_TAIL": (70, 80), "HIGH_FREQ": (90, 100),
}
PHYS_BANDS = {"wind": (1, 5), "veh": (5, 25), "foot": (20, 90),
              "high": (90, 180), "hop": (12, 15), "eng": (20, 60), "rain": (60, 200)}


def _bp(lo, hi, order=4):
    nyq = FS / 2
    return sig.butter(order, [max(lo / nyq, 1e-3), min(hi / nyq, 0.999)],
                      btype="band", output="sos")


_SOS_LEG = {k: _bp(lo, hi) for k, (lo, hi) in LEGACY_BANDS.items()}
_SOS_PHY = {k: _bp(lo, hi) for k, (lo, hi) in PHYS_BANDS.items()}
_SOS_ENVLP = sig.butter(4, 20 / (FS / 2), btype="low", output="sos")

# rfft grids
_FREQS = np.fft.rfftfreq(NW, 1 / FS)                 # 0.333 Hz bins
_FM = np.fft.rfftfreq(NFFT_ENV, 1 / FS)              # envelope-spectrum grid
_M_MOD = (_FM >= 0.3) & (_FM <= 12.0)                # modulation analysis band
_M_CAD = (_FM >= 0.8) & (_FM <= 4.0)                 # human cadence search
_M_FFR = (_FM >= 0.8) & (_FM <= 10.0)                # footfall-rate search (quadruped 4-8)
_M_PKS = (_FM >= 0.5) & (_FM <= 8.0)                 # multi-subject peak count
_M_GUST = (_FM >= 0.05) & (_FM <= 0.5)               # gust AM
_M_GALL = (_FM >= 0.05) & (_FM <= 12.0)

# site/mains tonal lines (rig-measured 44-47/55 + optional 50 Hz family) — these are
# EXCLUDED from comb/line features and monitored separately as line_mains
SITE_LINES = np.array([44, 45, 46, 47, 50, 55, 100, 150], float)
_SITE_BIN = np.zeros(len(_FREQS), bool)
for _f in SITE_LINES:
    _SITE_BIN |= np.abs(_FREQS - _f) <= 0.75
_M_LINES = (_FREQS >= 10) & (_FREQS <= 180)
# engine comb f0 candidates: 25-58 Hz bins, excluding site-line bins
_COMB_K = np.where((_FREQS >= 25) & (_FREQS <= 58) & ~_SITE_BIN)[0]

# ---- v2 additional grids (new feature groups) ----
_M_MOD1_3 = (_FM >= 1) & (_FM <= 3)        # human cadence band (modulation)
_M_MOD3_8 = (_FM >= 3) & (_FM <= 8)        # quadruped / fast-footfall band
_M_MOD8_12 = (_FM >= 8) & (_FM <= 12)      # herd/group blur band
_ROTOR_K = np.where((_FREQS >= 12) & (_FREQS <= 22))[0]   # helicopter main-rotor BPF comb
_PUMP = (_FREQS >= 75) & (_FREQS <= 180)                  # irrigation-pump tonal band


def _tri_fb(edges, freqs):
    """Triangular filterbank (nfilt x nbins) from band edges."""
    nf = len(edges) - 2
    fb = np.zeros((nf, len(freqs)))
    for i in range(nf):
        l, c, r = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - l) / max(c - l, 1e-9),
                                   (r - freqs) / max(r - c, 1e-9)), 0, None)
    return fb


_mel = lambda f: 2595 * np.log10(1 + f / 700)
_imel = lambda m: 700 * (10 ** (m / 2595) - 1)
_FB_MEL = _tri_fb(_imel(np.linspace(_mel(1), _mel(200), 26)), _FREQS)
_FB_LOG = _tri_fb(np.geomspace(2, 200, 26), _FREQS)

FEATURE_NAMES = (
    # LEGACY32
    ["energy_low_freq", "energy_car_approach", "energy_car_peak", "energy_car_tail",
     "energy_mid_gap", "energy_human_peak", "energy_human_tail", "energy_high_freq",
     "rms_total", "peak_to_peak", "variance", "zcr", "kurtosis", "skewness",
     "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
     "spectral_entropy", "dominant_freq", "sta_lta_ratio", "event_count",
     "mean_burst_len", "activity_concentration", "temporal_entropy", "burst_efficiency",
     "autocorr_250", "autocorr_500", "ratio_car_human", "ratio_human_car",
     "centroid_car_band", "centroid_human_band"]
    # PHYS7
    + ["log_rms", "frac_wind_1_5", "frac_veh_5_25", "frac_foot_20_90",
       "frac_high_90_180", "centroid_foot_band", "ratio_veh_foot"]
    # CAD12
    + ["cad_freq", "cad_salience", "cad_harm2", "cad_harm4", "footfall_rate",
       "mod_peak_count", "env_entropy", "env_cv", "env_ac_strength",
       "iii_mean", "iii_cv", "gust_mod"]
    # ENG8
    + ["comb_salience", "comb_f0", "n_lines", "line_stability", "hop_band_frac",
       "beat_depth", "line_mains", "tonality_max"]
    # WX2
    + ["rain_impulse_rate", "highband_kurtosis"]
    # CEP16
    + [f"mfcc_{i}" for i in range(1, 9)] + [f"lfcc_{i}" for i in range(1, 9)]
    # WPE16
    + [f"wpe_{i}" for i in range(16)]
    # AR8
    + [f"ar_{i}" for i in range(1, 9)]
    # NL1
    + ["higuchi_fd"]
    # ==== v2 NEW (30) ====
    # MOD2 — modulation-spectrum refinement (single/multi + biped/quad/vehicle)
    + ["mod_e_1_3", "mod_e_3_8", "mod_e_8_12", "mod_ratio_low_mid", "mod_peak2_ratio",
       "mod_flatness", "mod_centroid"]
    # III2 — inter-impulse-interval stats (single/multi)
    + ["impulse_density", "iii_entropy", "iii_skew", "iii_range_norm", "impulse_amp_cv"]
    # TONAL — tonal-vs-impulsive / machinery+aircraft confuser separation (nothing/class)
    + ["spectral_crest", "n_sharp_lines", "max_line_prom", "tonal_index",
       "heli_comb_salience", "pump_band_frac", "tonal_in_foot"]
    # STAT2 — stationarity / temporal structure (between-class)
    + ["centroid_var", "rms_cv_sub", "spectral_flux", "temporal_centroid", "attack_sharpness"]
    # HOS — higher-order spectral (engine harmonic coupling)
    + ["spec_skew", "spec_kurt", "comb_stability"]
    # COOC — co-occurrence / multi-source complexity (mixed scenes)
    + ["band_occupancy", "spectral_entropy_full", "veh_foot_simultaneity"]
)
NFEAT = len(FEATURE_NAMES)
assert NFEAT == 132, NFEAT


# ------------------------------------------------------------ scene precompute
def scene_precompute(x):
    """One-time per-scene work: band-filtered signals, envelopes, impulse picks."""
    x = np.asarray(x, np.float64)
    pre = {"x": x}
    for k, sos in _SOS_LEG.items():
        pre["leg_" + k] = sig.sosfilt(sos, x)
    for k, sos in _SOS_PHY.items():
        pre["phy_" + k] = sig.sosfilt(sos, x)
    # broadband envelope (|x| lowpassed 20 Hz) for cadence/modulation work
    pre["env"] = np.clip(sig.sosfilt(_SOS_ENVLP, np.abs(x)), 0, None)
    pre["env_eng"] = np.clip(sig.sosfilt(_SOS_ENVLP, np.abs(pre["phy_eng"])), 0, None)
    env_rain = np.clip(sig.sosfilt(_SOS_ENVLP, np.abs(pre["phy_rain"])), 0, None)
    # impulse picks (scene-wide; windows subselect)
    med = np.median(pre["env"]) + 1e-12
    pk, _ = sig.find_peaks(pre["env"], height=2 * med, distance=int(0.08 * FS))
    pre["imp_t"] = pk
    medr = np.median(env_rain) + 1e-12
    pkr, _ = sig.find_peaks(env_rain, height=3 * medr, distance=int(0.02 * FS))
    pre["rain_t"] = pkr
    return pre


# ------------------------------------------------------------ helpers
def _higuchi(w, kmax=8):
    n = len(w)
    lk = np.empty(kmax)
    for k in range(1, kmax + 1):
        Lm = 0.0
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            Lm += np.abs(np.diff(w[idx])).sum() * (n - 1) / (len(idx) - 1) / k
        lk[k - 1] = Lm / k + 1e-30
    kk = np.log(1.0 / np.arange(1, kmax + 1))
    return float(np.polyfit(kk, np.log(lk), 1)[0])


def _levinson(r, p=8):
    a = np.zeros(p + 1); a[0] = 1.0; e = r[0] + 1e-30
    for i in range(1, p + 1):
        k = -(r[i] + a[1:i] @ r[i - 1:0:-1]) / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= (1 - k * k) + 1e-30
    return a[1:]


# ------------------------------------------------------------ per-window
def window_features(pre, i0):
    f = np.empty(NFEAT, np.float64)
    j = 0
    w = pre["x"][i0:i0 + NW]
    aw = np.abs(w)

    # ---- LEGACY32 -----------------------------------------------------------
    band_e = {}
    for k in LEGACY_BANDS:
        e = float(np.sqrt(np.mean(pre["leg_" + k][i0:i0 + NW] ** 2)))
        band_e[k] = e; f[j] = e; j += 1
    rms_total = float(np.sqrt(np.mean(w ** 2)))
    f[j] = rms_total; j += 1
    f[j] = float(np.ptp(w)); j += 1
    f[j] = float(np.var(w)); j += 1
    f[j] = np.sum(np.diff(np.sign(w)) != 0) / NW; j += 1
    f[j] = float(spstats.kurtosis(w)); j += 1
    f[j] = float(spstats.skew(w)); j += 1
    mag = np.abs(np.fft.rfft(w))
    tot_mag = mag.sum() + 1e-10
    psd = mag ** 2
    tot_psd = psd.sum() + 1e-10
    sc = float((_FREQS * mag).sum() / tot_mag)
    f[j] = sc; j += 1
    f[j] = float(np.sqrt((((_FREQS - sc) ** 2) * mag).sum() / tot_mag)); j += 1
    cs = np.cumsum(psd)
    f[j] = _FREQS[min(np.searchsorted(cs, 0.95 * tot_psd), len(_FREQS) - 1)]; j += 1
    f[j] = float(np.exp(np.mean(np.log(psd + 1e-10))) / (psd.mean() + 1e-10)); j += 1
    pn = psd / tot_psd
    f[j] = float(-(pn * np.log2(pn + 1e-10)).sum()); j += 1
    f[j] = _FREQS[int(np.argmax(mag))]; j += 1
    lta = aw.mean() + 1e-10
    sta = np.convolve(aw, np.ones(100) / 100, mode="valid")
    f[j] = float(sta.max() / lta); j += 1
    pk, _ = sig.find_peaks(aw, height=2 * rms_total)
    f[j] = float(len(pk)); j += 1
    above = aw > rms_total
    d = np.diff(above.astype(np.int8))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    if len(st) and len(en):
        if en[0] < st[0]:
            en = en[1:]
        nb = min(len(st), len(en))
        f[j] = float(np.mean(en[:nb] - st[:nb])) if nb else 0.0
    else:
        f[j] = 0.0
    j += 1
    se = np.sort(w ** 2)[::-1]
    f[j] = float(se[:NW // 4].sum() / (se.sum() + 1e-10)); j += 1
    hist, _ = np.histogram(w, bins=50, density=True)
    hist = hist[hist > 0]; hn = hist / (hist.sum() + 1e-10)
    f[j] = float(-(hn * np.log2(hn + 1e-10)).sum()); j += 1
    f[j] = float((w[above] ** 2).sum() / ((w ** 2).sum() + 1e-10)); j += 1
    for lag in (250, 500):
        ac = np.corrcoef(w[:-lag], w[lag:])[0, 1]
        f[j] = float(ac) if np.isfinite(ac) else 0.0
        j += 1
    car_e = band_e["CAR_PEAK"] + 1e-10
    hum_e = band_e["HUMAN_PEAK"] + 1e-10
    f[j] = car_e / hum_e; j += 1
    f[j] = hum_e / car_e; j += 1
    for lo, hi in ((34, 48), (60, 80)):
        m = (_FREQS >= lo) & (_FREQS <= hi)
        f[j] = float((_FREQS[m] * mag[m]).sum() / (mag[m].sum() + 1e-10)); j += 1

    # ---- PHYS7 --------------------------------------------------------------
    f[j] = float(np.log10(rms_total + 1e-12)); j += 1
    phys_rms = {}
    for k in ("wind", "veh", "foot", "high"):
        phys_rms[k] = float(np.sqrt(np.mean(pre["phy_" + k][i0:i0 + NW] ** 2)))
        f[j] = phys_rms[k] / (rms_total + 1e-12); j += 1
    m = (_FREQS >= 20) & (_FREQS <= 90)
    f[j] = float((_FREQS[m] * mag[m]).sum() / (mag[m].sum() + 1e-10)); j += 1
    f[j] = phys_rms["veh"] / (phys_rms["foot"] + 1e-12); j += 1

    # ---- CAD12 --------------------------------------------------------------
    e = pre["env"][i0:i0 + NW]
    e0 = e - e.mean()
    Espec = np.abs(np.fft.rfft(e0, NFFT_ENV)) ** 2
    modmed = np.median(Espec[_M_MOD]) + 1e-20
    kc = np.where(_M_CAD)[0]
    kpk = kc[int(np.argmax(Espec[kc]))]
    f0 = _FM[kpk]
    f[j] = float(f0); j += 1
    f[j] = float(Espec[kpk] / modmed); j += 1
    pf0 = Espec[max(kpk - 2, 0):kpk + 3].sum() + 1e-20
    for mult in (2, 4):
        kh = int(round(kpk * mult))
        ph = Espec[max(kh - 2, 0):kh + 3].sum() if kh + 3 <= len(Espec) else 0.0
        f[j] = float(ph / pf0); j += 1
    kr = np.where(_M_FFR)[0]
    f[j] = float(_FM[kr[int(np.argmax(Espec[kr]))]]); j += 1
    kp = np.where(_M_PKS)[0]
    pks, _ = sig.find_peaks(Espec[kp], height=3 * modmed)
    f[j] = float(len(pks)); j += 1
    Em = Espec[_M_MOD] / (Espec[_M_MOD].sum() + 1e-20)
    f[j] = float(-(Em * np.log2(Em + 1e-20)).sum()); j += 1
    f[j] = float(e.std() / (e.mean() + 1e-12)); j += 1
    ac = np.fft.irfft(Espec)                      # envelope autocorrelation (padded->linear)
    f[j] = float(np.max(ac[250:1250]) / (ac[0] + 1e-20)); j += 1
    it = pre["imp_t"]
    it = it[(it >= i0) & (it < i0 + NW)]
    if len(it) >= 3:
        iii = np.diff(it) / FS
        f[j] = float(iii.mean()); j += 1
        f[j] = float(iii.std() / (iii.mean() + 1e-9)); j += 1
    else:
        f[j] = 0.0; j += 1
        f[j] = 0.0; j += 1
    f[j] = float(Espec[_M_GUST].sum() / (Espec[_M_GALL].sum() + 1e-20)); j += 1

    # ---- ENG8 ---------------------------------------------------------------
    locmed = sig.medfilt(psd, 51) + 1e-12
    pnorm = psd / locmed
    best_s, best_k = 0.0, _COMB_K[0]
    for k in _COMB_K:
        s = 0.0; nv = 0
        for o in (1, 2, 3):
            ko = k * o
            if ko + 1 < len(pnorm) and not _SITE_BIN[ko - 1:ko + 2].any():
                s += pnorm[ko - 1:ko + 2].max(); nv += 1   # skip orders on site/mains lines
        if nv >= 2:                                        # need >=2 clean orders (real comb)
            s /= nv
            if s > best_s:
                best_s, best_k = s, k
    f[j] = float(best_s); j += 1
    f[j] = float(_FREQS[best_k]); j += 1
    lpk, _ = sig.find_peaks(np.where(_M_LINES & ~_SITE_BIN, pnorm, 0.0), height=6.0)
    f[j] = float(len(lpk)); j += 1
    sub = np.stack([np.log10(np.abs(np.fft.rfft(w[s * 1000:(s + 1) * 1000])) ** 2 + 1e-12)
                    for s in range(3)])
    fsub = np.fft.rfftfreq(1000, 1 / FS)
    msub = (fsub >= 10) & (fsub <= 100)
    cc = np.corrcoef(sub[:, msub])
    f[j] = float((cc[0, 1] + cc[0, 2] + cc[1, 2]) / 3); j += 1
    f[j] = float(np.sqrt(np.mean(pre["phy_hop"][i0:i0 + NW] ** 2)) / (rms_total + 1e-12)); j += 1
    ee = pre["env_eng"][i0:i0 + NW]
    f[j] = float(ee.std() / (ee.mean() + 1e-12)); j += 1
    f[j] = float(pnorm[_SITE_BIN].mean()); j += 1
    f[j] = float(pnorm[_M_LINES].max()); j += 1

    # ---- WX2 ----------------------------------------------------------------
    rt = pre["rain_t"]
    f[j] = float(((rt >= i0) & (rt < i0 + NW)).sum() / (NW / FS)); j += 1
    f[j] = float(spstats.kurtosis(pre["phy_rain"][i0:i0 + NW])); j += 1

    # ---- CEP16 --------------------------------------------------------------
    for fb in (_FB_MEL, _FB_LOG):
        c = dct(np.log(fb @ mag + 1e-10), type=2, norm="ortho")
        f[j:j + 8] = c[1:9]; j += 8

    # ---- WPE16 --------------------------------------------------------------
    wp = pywt.WaveletPacket(w, "db4", maxlevel=4)
    en = np.array([float((n.data ** 2).sum()) for n in wp.get_level(4, "natural")])
    f[j:j + 16] = en / (en.sum() + 1e-20); j += 16

    # ---- AR8 ----------------------------------------------------------------
    r = np.fft.irfft(psd)[:9]
    f[j:j + 8] = _levinson(r, 8); j += 8

    # ---- NL1 ----------------------------------------------------------------
    f[j] = _higuchi(w); j += 1

    # ==== v2 NEW (reuse mag/psd/Espec/modmed/it/pnorm/phys_rms/sub/msub/fsub) ====
    # MOD2 — modulation-spectrum refinement
    Emod = Espec[_M_MOD]; Emt = Emod.sum() + 1e-20
    f[j] = float(Espec[_M_MOD1_3].sum() / Emt); j += 1                 # mod_e_1_3 (human cadence)
    f[j] = float(Espec[_M_MOD3_8].sum() / Emt); j += 1                 # mod_e_3_8 (quadruped/fast)
    f[j] = float(Espec[_M_MOD8_12].sum() / Emt); j += 1                # mod_e_8_12 (group blur)
    f[j] = float(Espec[_M_MOD1_3].sum() / (Espec[_M_MOD3_8].sum() + 1e-20)); j += 1  # mod_ratio_low_mid
    mpk, _ = sig.find_peaks(Espec[_M_PKS], height=3 * modmed)
    if len(mpk) >= 2:
        top = np.sort(Espec[_M_PKS][mpk])[::-1]; f[j] = float(top[1] / (top[0] + 1e-20))
    else:
        f[j] = 0.0
    j += 1                                                             # mod_peak2_ratio (multi)
    f[j] = float(np.exp(np.mean(np.log(Emod + 1e-20))) / (Emod.mean() + 1e-20)); j += 1  # mod_flatness
    f[j] = float((_FM[_M_MOD] * Emod).sum() / Emt); j += 1             # mod_centroid (rhythm speed)

    # III2 — inter-impulse-interval stats (single vs multiple)
    f[j] = float(len(it) / (NW / FS)); j += 1                          # impulse_density
    if len(it) >= 4:
        iii = np.diff(it) / FS
        h, _ = np.histogram(iii, bins=8); h = h[h > 0] / (h.sum() + 1e-12)
        f[j] = float(-(h * np.log2(h + 1e-12)).sum()); j += 1          # iii_entropy
        f[j] = float(spstats.skew(iii)); j += 1                        # iii_skew
        f[j] = float((np.percentile(iii, 90) - np.percentile(iii, 10)) / (np.median(iii) + 1e-9)); j += 1  # iii_range_norm
        amps = pre["env"][it]; f[j] = float(amps.std() / (amps.mean() + 1e-12)); j += 1  # impulse_amp_cv
    else:
        f[j] = 0.0; j += 1; f[j] = 0.0; j += 1; f[j] = 0.0; j += 1; f[j] = 0.0; j += 1

    # TONAL — tonal-vs-impulsive / machinery+aircraft confuser separation
    f[j] = float(psd.max() / (psd.mean() + 1e-20)); j += 1             # spectral_crest
    ratio_line = np.where((_FREQS >= 10) & (_FREQS <= 200) & ~_SITE_BIN, pnorm, 0.0)
    slp, _ = sig.find_peaks(ratio_line, height=8.0)
    f[j] = float(len(slp)); j += 1                                     # n_sharp_lines (non-mains)
    f[j] = float(ratio_line.max()); j += 1                             # max_line_prom
    inb = psd[(_FREQS >= 10) & (_FREQS <= 200)].sum() + 1e-20
    f[j] = float(psd[ratio_line > 8.0].sum() / inb); j += 1            # tonal_index
    best_h = 0.0
    for k in _ROTOR_K:                                                 # helicopter rotor-BPF comb 12-22 Hz
        s = 0.0; nv = 0
        for o in (1, 2, 3):
            ko = k * o
            if ko + 1 < len(pnorm) and not _SITE_BIN[ko - 1:ko + 2].any():
                s += pnorm[ko - 1:ko + 2].max(); nv += 1
        if nv >= 2:
            best_h = max(best_h, s / nv)
    f[j] = float(best_h); j += 1                                       # heli_comb_salience
    f[j] = float(psd[_PUMP].sum() / tot_psd); j += 1                   # pump_band_frac (75-180)
    fbm = (_FREQS >= 20) & (_FREQS <= 90)
    f[j] = float(psd[(ratio_line > 8.0) & fbm].sum() / (psd[fbm].sum() + 1e-20)); j += 1  # tonal_in_foot

    # STAT2 — stationarity / temporal structure
    subc = []; subr = []
    for s_ in range(3):
        ws = w[s_ * 1000:(s_ + 1) * 1000]; m_ = np.abs(np.fft.rfft(ws))
        subc.append((fsub * m_).sum() / (m_.sum() + 1e-10)); subr.append(np.sqrt(np.mean(ws ** 2)))
    subc = np.array(subc); subr = np.array(subr)
    f[j] = float(subc.std()); j += 1                                   # centroid_var (non-stationary)
    f[j] = float(subr.std() / (subr.mean() + 1e-12)); j += 1           # rms_cv_sub (burstiness)
    f[j] = float(np.mean(np.abs(np.diff(sub, axis=0)))); j += 1        # spectral_flux
    w2 = w ** 2; f[j] = float((np.arange(NW) * w2).sum() / ((w2.sum() + 1e-20) * NW)); j += 1  # temporal_centroid
    if len(it) >= 2:
        sl = [(pre["env"][ti] - pre["env"][max(ti - 10, 0)]) / 0.01 for ti in it[:20]]
        f[j] = float(np.mean(sl)) if sl else 0.0
    else:
        f[j] = 0.0
    j += 1                                                             # attack_sharpness

    # HOS — higher-order spectral
    f[j] = float(spstats.skew(psd)); j += 1                            # spec_skew
    f[j] = float(spstats.kurtosis(psd)); j += 1                        # spec_kurt (tonal peaky)
    f[j] = float(np.mean(np.std(sub[:, msub], axis=0))); j += 1        # comb_stability (engine phase-lock)

    # COOC — co-occurrence / multi-source complexity (mixed scenes)
    f[j] = float(sum(1 for k in ("wind", "veh", "foot", "high")
                     if phys_rms.get(k, 0) / (rms_total + 1e-12) > 0.15)); j += 1  # band_occupancy
    pnf = psd / tot_psd; f[j] = float(-(pnf * np.log2(pnf + 1e-12)).sum()); j += 1  # spectral_entropy_full
    f[j] = float(phys_rms.get("veh", 0) / (rms_total + 1e-12) * min(len(it) / (NW / FS), 5.0)); j += 1  # veh_foot_simultaneity

    assert j == NFEAT
    return f
