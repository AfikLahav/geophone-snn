"""sp_lib.py -- shared utilities for the REAL vs SYNTHETIC-v4 spectral discrepancy report.

All PSDs computed at fs=1000 Hz. Analysis band 1-450 Hz.
Two real rigs: human-session (human.csv / human_nothing.csv) and
car-session (car.csv / car_nothing.csv). Root geophone_2026*.csv = field floor.
Synth v4: $GEO_SYNTH_ROOT/dataset_v431/shard_0.sqlite, blobs float32 in mV.
Model input = clip(noise_mv + clean_mv (+ clean_mv2), -256, 256).

This module builds per-group window PSD matrices and caches them to sp_cache.npz
so the downstream sp_*.py scripts do not re-read the 3 GB sqlite.
"""
import os, sqlite3, json
import numpy as np
import pandas as pd
from scipy.signal import welch

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))

FS = 1000.0
BASE = os.environ.get("PROJECT_ROOT", ".")
REALDIR = os.path.join(BASE, "Goephone-Project", "geophone_data")
DB = os.path.join(_GEO_ROOT, "dataset_v431/shard_0.sqlite")
OUTDIR = os.path.join(BASE, "dataset_validation", "discrepancy")
CACHE = os.path.join(OUTDIR, "sp_cache.npz")

# analysis config
WIN_S = 3.0                # window length (matches E0_win3s)
HOP_S = 1.5               # 50% overlap
NPERSEG = 1000           # 1 Hz resolution for band/slope/centroid/coupling
NOVER = 500
NPERSEG_HR = 4096        # ~0.244 Hz resolution for line inventory
NOVER_HR = 2048
CLIP = 256.0
BAND_LO, BAND_HI = 1.0, 450.0
BANDS = [(1, 5), (5, 25), (20, 55), (55, 90), (90, 180), (180, 450)]
SYNTH_SNR_MIN = 12.0     # signal-dominated selection for model-input human/vehicle
N_SYNTH = 120            # scenes sampled per synth group
RNG = np.random.default_rng(20260706)

# ----------------------------------------------------------------------------
def load_real(path):
    df = pd.read_csv(path, usecols=[1])
    return df.iloc[:, 0].to_numpy(dtype=np.float64) * 1000.0  # V -> mV


def _win_bounds(n, win, hop):
    out = []
    i = 0
    while i + win <= n:
        out.append((i, i + win))
        i += hop
    return out


def window_psds(x, active_ref_rms=None, active_k=3.0, clean=None,
                clean_frac=0.25, nperseg=NPERSEG, nover=NOVER):
    """Per-window Welch PSDs. Returns (freqs, psd_matrix[nwin,nf], sel_mask, win_rms).

    Activity selection:
      - if clean is not None (synth): window active where its clean-component RMS
        exceeds clean_frac * (max clean-window RMS of this record).
      - elif active_ref_rms given (real): window active where broadband RMS >
        active_k * active_ref_rms.
      - else: all windows active.
    """
    win = int(round(WIN_S * FS)); hop = int(round(HOP_S * FS))
    bounds = _win_bounds(len(x), win, hop)
    if not bounds:
        return None, None, None, None
    psds = []; rms = []; crms = []
    freqs = None
    for a, b in bounds:
        seg = x[a:b]
        f, p = welch(seg, fs=FS, nperseg=min(nperseg, len(seg)),
                     noverlap=min(nover, len(seg) // 2), detrend='constant',
                     window='hann', scaling='density')
        freqs = f
        psds.append(p)
        rms.append(float(np.sqrt(np.mean(seg**2))))
        if clean is not None:
            cseg = clean[a:b]
            crms.append(float(np.sqrt(np.mean(cseg**2))))
    psds = np.asarray(psds); rms = np.asarray(rms)
    if clean is not None:
        crms = np.asarray(crms)
        thr = clean_frac * (crms.max() if crms.max() > 0 else 1.0)
        sel = (crms > thr) & (crms > 0)
    elif active_ref_rms is not None:
        sel = rms > (active_k * active_ref_rms)
    else:
        sel = np.ones(len(psds), bool)
    return freqs, psds, sel, rms


def hr_psd(x, nperseg=NPERSEG_HR, nover=NOVER_HR):
    f, p = welch(x, fs=FS, nperseg=min(nperseg, len(x)),
                 noverlap=min(nover, len(x) // 2), detrend='constant',
                 window='hann', scaling='density')
    return f, p


# ----------------------------------------------------------------------------
def _fetch_blob(row_field):
    if row_field is None:
        return None
    return np.frombuffer(row_field, dtype=np.float32).astype(np.float64)


def synth_scene_signals(cur, scene_id):
    r = cur.execute('select noise_mv, clean_mv, clean_mv2 from waveforms where scene_id=?',
                    (scene_id,)).fetchone()
    noise = _fetch_blob(r[0]); c1 = _fetch_blob(r[1]); c2 = _fetch_blob(r[2])
    n = len(noise)
    clean = np.zeros(n)
    if c1 is not None and len(c1) == n:
        clean = clean + c1
    if c2 is not None and len(c2) == n:
        clean = clean + c2
    model_in = np.clip(noise + clean, -CLIP, CLIP)
    return noise, clean, model_in


def sample_scene_ids(cur, where, n=N_SYNTH):
    ids = [r[0] for r in cur.execute(f'select scene_id from scenes where {where}').fetchall()]
    ids = np.asarray(ids)
    if len(ids) > n:
        ids = RNG.choice(ids, n, replace=False)
    return ids.tolist()


# ----------------------------------------------------------------------------
def median_from_windowlist(freqs, psd_list):
    """Stack per-record selected window PSDs and take median across windows."""
    M = np.vstack(psd_list)
    return freqs, np.median(M, axis=0), M.shape[0]


def build_groups():
    """Compute + cache all group PSDs. Returns dict of arrays."""
    out = {}
    meta = {}

    # ---- real nothing floors (also give active_ref_rms per rig) ----
    real_files = {
        'real_human_nothing': os.path.join(REALDIR, 'human_nothing.csv'),
        'real_car_nothing': os.path.join(REALDIR, 'car_nothing.csv'),
    }
    ref_rms = {}
    for g, path in real_files.items():
        x = load_real(path)
        f, psds, sel, rms = window_psds(x)  # all windows
        out[f'{g}__f'] = f
        out[f'{g}__psd'] = np.median(psds, axis=0)
        out[f'{g}__n'] = psds.shape[0]
        fh, ph = hr_psd(x)
        out[f'{g}__fhr'] = fh; out[f'{g}__phr'] = ph
        ref_rms[g] = float(np.median(rms))
        meta[g] = {'n_windows': int(psds.shape[0]), 'floor_median_rms_mv': ref_rms[g]}

    # ---- real active (paired rig floor) ----
    active_map = {
        'real_human_active': (os.path.join(REALDIR, 'human.csv'), 'real_human_nothing'),
        'real_car_active': (os.path.join(REALDIR, 'car.csv'), 'real_car_nothing'),
    }
    for g, (path, floorg) in active_map.items():
        x = load_real(path)
        f, psds, sel, rms = window_psds(x, active_ref_rms=ref_rms[floorg], active_k=3.0)
        selpsd = psds[sel]
        if selpsd.shape[0] < 20:  # relax if too few
            k = 2.0
            sel = rms > 2.0 * ref_rms[floorg]; selpsd = psds[sel]
        out[f'{g}__f'] = f
        out[f'{g}__psd'] = np.median(selpsd, axis=0)
        out[f'{g}__n'] = selpsd.shape[0]
        # high-res on the concatenated active windows for line inventory
        win = int(WIN_S*FS); hop = int(HOP_S*FS)
        bounds = _win_bounds(len(x), win, hop)
        act = np.concatenate([x[a:b] for (a, b), s in zip(bounds, sel) if s])
        fh, ph = hr_psd(act)
        out[f'{g}__fhr'] = fh; out[f'{g}__phr'] = ph
        meta[g] = {'n_windows': int(selpsd.shape[0]),
                   'n_total_windows': int(psds.shape[0]),
                   'active_frac': float(sel.mean())}

    # ---- real field floor (2026-06 root, quiet-ish) ----
    import glob

    field = sorted(glob.glob(os.path.join(BASE, 'geophone_20260615_*.csv')))
    fld_ws = []; fld_hr = []
    for p in field:
        x = load_real(p)
        f, psds, sel, rms = window_psds(x)
        if psds is not None:
            fld_ws.append(psds)
        fh, ph = hr_psd(x); fld_hr.append((fh, ph))
    if fld_ws:
        allw = np.vstack(fld_ws)
        out['real_field__f'] = f
        out['real_field__psd'] = np.median(allw, axis=0)
        out['real_field__n'] = allw.shape[0]
        # median hr across files (align by shortest)
        minlen = min(len(ph) for _, ph in fld_hr)
        fh0 = fld_hr[0][0][:minlen]
        phm = np.median(np.vstack([ph[:minlen] for _, ph in fld_hr]), axis=0)
        out['real_field__fhr'] = fh0; out['real_field__phr'] = phm
        meta['real_field'] = {'n_windows': int(allw.shape[0]), 'n_files': len(field)}

    # ---- synth groups ----
    con = sqlite3.connect(DB); cur = con.cursor()

    def synth_group(gname, where, use_clean=False, snr_sel=False, ncap=N_SYNTH):
        w = where
        if snr_sel:
            w = w + f" and target_snr_db >= {SYNTH_SNR_MIN}"
        ids = sample_scene_ids(cur, w, n=ncap)
        wlist = []; hrlist = []; nwin = 0
        for sid in ids:
            noise, clean, model_in = synth_scene_signals(cur, sid)
            if use_clean:
                sig = clean
                f, psds, sel, rms = window_psds(clean, clean=clean, clean_frac=0.25)
            elif clean.any():  # human/vehicle model-input, gate on clean activity
                f, psds, sel, rms = window_psds(model_in, clean=clean, clean_frac=0.25)
            else:  # nothing: all windows active
                f, psds, sel, rms = window_psds(model_in)
            if psds is None:
                continue
            selpsd = psds[sel] if sel is not None else psds
            if selpsd.shape[0] == 0:
                continue
            wlist.append(selpsd); nwin += selpsd.shape[0]
            # hr for lines (nothing groups only need it, but compute for all)
            fh, ph = hr_psd(sig if use_clean else model_in)
            hrlist.append((fh, ph))
        M = np.vstack(wlist)
        out[f'{gname}__f'] = f
        out[f'{gname}__psd'] = np.median(M, axis=0)
        out[f'{gname}__n'] = M.shape[0]
        minlen = min(len(ph) for _, ph in hrlist)
        out[f'{gname}__fhr'] = hrlist[0][0][:minlen]
        out[f'{gname}__phr'] = np.median(np.vstack([ph[:minlen] for _, ph in hrlist]), axis=0)
        meta[gname] = {'n_windows': int(M.shape[0]), 'n_scenes': len(ids)}

    synth_group('synth_human', "coarse='human'", snr_sel=True)
    synth_group('synth_human_clean', "coarse='human'", use_clean=True)
    synth_group('synth_vehicle', "coarse='vehicle'", snr_sel=True)
    synth_group('synth_vehicle_clean', "coarse='vehicle'", use_clean=True)
    # nothing per condition (merge wind_*, rain_*)
    synth_group('synth_nothing_calm', "coarse='nothing' and noise_condition='calm'")
    synth_group('synth_nothing_wind', "coarse='nothing' and noise_condition like 'wind%'")
    synth_group('synth_nothing_rain', "coarse='nothing' and noise_condition like 'rain%'")

    # coupling_fc distribution
    fcs = np.array([r[0] for r in cur.execute('select coupling_fc from scenes where coupling_fc is not null')], float)
    out['synth_coupling_fc'] = fcs
    con.close()

    np.savez_compressed(CACHE, **out)
    with open(os.path.join(OUTDIR, 'sp_meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    return out, meta


def load_cache():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ----------------------------------------------------------------------------
# metrics
def band_fractions(f, p, bands=BANDS, lo=BAND_LO, hi=BAND_HI):
    m = (f >= lo) & (f <= hi)
    ftot = f[m]; ptot = p[m]
    total = np.trapz(ptot, ftot)
    res = {}
    for b0, b1 in bands:
        mm = (f >= b0) & (f <= b1)
        res[f'{b0}-{b1}'] = float(np.trapz(p[mm], f[mm]) / total) if total > 0 else np.nan
    return res, float(total)


def loglog_slope(f, p, lo=2.0, hi=20.0):
    m = (f >= lo) & (f <= hi) & (p > 0)
    if m.sum() < 4:
        return np.nan
    lf = np.log10(f[m]); lp = np.log10(p[m])
    A = np.vstack([lf, np.ones_like(lf)]).T
    slope, _ = np.linalg.lstsq(A, lp, rcond=None)[0]
    return float(slope)


def spectral_centroid(f, p, lo=5.0, hi=180.0):
    m = (f >= lo) & (f <= hi)
    ff = f[m]; pp = p[m]
    return float(np.sum(ff * pp) / np.sum(pp)) if np.sum(pp) > 0 else np.nan


def hf_rolloff(f, p, floor_band=(400, 450), margin_db=3.0):
    """Effective bandwidth: highest freq (<=450) where smoothed PSD exceeds
    (noise-floor + margin_db). floor = median PSD in floor_band."""
    fm = (f >= floor_band[0]) & (f <= floor_band[1])
    floor = np.median(p[fm])
    thr = floor * 10 ** (margin_db / 10.0)
    m = (f >= 1) & (f <= 450)
    ff = f[m]; pp = p[m]
    above = ff[pp > thr]
    return float(above.max()) if above.size else np.nan, float(floor)


if __name__ == '__main__':
    o, m = build_groups()
    print('built groups:', sorted(k.split('__')[0] for k in o if '__' in k))
    for g, mm in m.items():
        print(g, mm)
