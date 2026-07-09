"""sp_lines.py -- Task 2: narrowband line inventory.
Detect lines (local peak > 6 dB above local median baseline) in every real file
and in synth nothing/active groups. Classify: mains (50/100/150), rig (44-47/55),
synth machinery (random 4-80 Hz). Writes sp_out_lines.json.
"""
import os, glob, json
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import sp_lib as L

PROM_DB = 6.0
DF_HR = L.FS / L.NPERSEG_HR          # ~0.244 Hz
MEDWIN = int(round(6.0 / DF_HR)) | 1  # ~+/-3 Hz local-median kernel (odd)
FMIN, FMAX = 1.0, 450.0


def detect_lines(f, p):
    m = (f >= FMIN) & (f <= FMAX)
    ff = f[m]; pp = p[m]
    db = 10 * np.log10(pp + 1e-20)
    base = median_filter(db, size=MEDWIN, mode='nearest')
    resid = db - base
    dist = max(1, int(round(1.0 / DF_HR)))  # >=1 Hz apart
    idx, props = find_peaks(resid, height=PROM_DB, distance=dist)
    lines = []
    for i in idx:
        lines.append({'hz': round(float(ff[i]), 2), 'prom_db': round(float(resid[i]), 2)})
    lines.sort(key=lambda x: -x['prom_db'])
    return lines


def classify(lines):
    def near(h, targ, tol):
        return any(abs(h - t) <= tol for t in targ)
    tags = {'mains_50_100_150': [], 'rig_44_47_55': [], 'spurious_140_180': [],
            'machinery_4_80': [], 'other': []}
    for ln in lines:
        h = ln['hz']
        if near(h, [50, 100, 150, 200, 250, 300, 350, 400, 450], 1.5):
            tags['mains_50_100_150'].append(ln)
        elif 44 <= h <= 47 or 53 <= h <= 57:
            tags['rig_44_47_55'].append(ln)
        elif 140 <= h <= 180:
            tags['spurious_140_180'].append(ln)
        elif 4 <= h <= 80:
            tags['machinery_4_80'].append(ln)
        else:
            tags['other'].append(ln)
    return tags


out = {'config': {'prominence_db': PROM_DB, 'freq_res_hz': round(DF_HR, 3),
                  'local_median_kernel_hz': round(MEDWIN * DF_HR, 2),
                  'band_hz': [FMIN, FMAX]}, 'real_files': {}, 'synth_groups': {}}

# ---- real files (per file) ----
real_named = {
    'human.csv': os.path.join(L.REALDIR, 'human.csv'),
    'car.csv': os.path.join(L.REALDIR, 'car.csv'),
    'human_nothing.csv': os.path.join(L.REALDIR, 'human_nothing.csv'),
    'car_nothing.csv': os.path.join(L.REALDIR, 'car_nothing.csv'),
}
for name, path in real_named.items():
    x = L.load_real(path)
    f, p = L.hr_psd(x)
    lines = detect_lines(f, p)
    out['real_files'][name] = {'n_lines': len(lines), 'lines': lines[:30],
                               'classified': classify(lines)}

# root field files (2026-06 and 2026-05)
for path in sorted(glob.glob(os.path.join(L.BASE, 'geophone_2026*.csv'))):
    name = os.path.basename(path)
    x = L.load_real(path)
    if len(x) < L.NPERSEG_HR:
        f, p = L.hr_psd(x, nperseg=max(256, len(x) // 2), nover=max(128, len(x) // 4))
    else:
        f, p = L.hr_psd(x)
    lines = detect_lines(f, p)
    out['real_files'][name] = {'n_lines': len(lines), 'lines': lines[:20],
                               'classified': classify(lines)}

# ---- synth groups (median hr PSD from cache) ----
d = L.load_cache()
for g in ['synth_nothing_calm', 'synth_nothing_wind', 'synth_nothing_rain',
          'synth_human', 'synth_vehicle']:
    f = d[f'{g}__fhr']; p = d[f'{g}__phr']
    lines = detect_lines(f, p)
    out['synth_groups'][g] = {'n_lines': len(lines), 'lines': lines[:30],
                              'classified': classify(lines)}

# ---- PER-SCENE synth line detection (random machinery lines + 150Hz resonance
#      wash out in the median, so characterize their per-scene distribution) ----
import sqlite3
con = sqlite3.connect(L.DB); cur = con.cursor()


def per_scene_lines(where, n=150):
    ids = L.sample_scene_ids(cur, where, n=n)
    all_lines = []; scenes_with_150 = 0; p150 = []
    scenes_with_mach = 0
    for sid in ids:
        _, _, model_in = L.synth_scene_signals(cur, sid)
        f, p = L.hr_psd(model_in)
        lines = detect_lines(f, p)
        all_lines += lines
        mach = [l for l in lines if 4 <= l['hz'] <= 80 and not any(abs(l['hz']-t) <= 1.5 for t in [50])]
        if mach:
            scenes_with_mach += 1
        spur = [l for l in lines if 140 <= l['hz'] <= 180]
        if spur:
            scenes_with_150 += 1
            p150.append(max(s['prom_db'] for s in spur))
    freqs = np.array([l['hz'] for l in all_lines])
    proms = np.array([l['prom_db'] for l in all_lines])
    mach_f = freqs[(freqs >= 4) & (freqs <= 80) &
                   ~np.isin(np.round(freqs), [50])]
    return {
        'n_scenes': len(ids),
        'total_lines': len(all_lines),
        'lines_per_scene_mean': round(len(all_lines) / max(1, len(ids)), 2),
        'machinery_4_80_scene_frac': round(scenes_with_mach / max(1, len(ids)), 3),
        'machinery_freq_hz_p10_p50_p90': [round(float(np.percentile(mach_f, q)), 1) for q in (10, 50, 90)] if mach_f.size else None,
        'machinery_prom_db_median': round(float(np.median(proms[(freqs >= 4) & (freqs <= 80)])), 2) if freqs.size else None,
        'spurious_140_180_scene_frac': round(scenes_with_150 / max(1, len(ids)), 3),
        'spurious_140_180_prom_db_p50_p90': [round(float(np.percentile(p150, 50)), 2), round(float(np.percentile(p150, 90)), 2)] if p150 else None,
    }


out['synth_per_scene'] = {
    'machinery': per_scene_lines("coarse='nothing' and subkind='machinery'"),
    'calm': per_scene_lines("coarse='nothing' and noise_condition='calm'"),
    'all_nothing': per_scene_lines("coarse='nothing'"),
    'human': per_scene_lines("coarse='human' and target_snr_db>=12"),
    'vehicle': per_scene_lines("coarse='vehicle' and target_snr_db>=12"),
}
con.close()

with open(os.path.join(L.OUTDIR, 'sp_out_lines.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
print('=== SYNTH per-scene line stats ===')
for k, v in out['synth_per_scene'].items():
    print(f"{k:12s} lines/scene={v['lines_per_scene_mean']:5.2f} mach_frac={v['machinery_4_80_scene_frac']} "
          f"mach_f(p10/50/90)={v['machinery_freq_hz_p10_p50_p90']} "
          f"spur140-180_frac={v['spurious_140_180_scene_frac']} spur_prom={v['spurious_140_180_prom_db_p50_p90']}")

print('=== REAL line inventory ===')
for name, r in out['real_files'].items():
    c = r['classified']
    print(f"{name:28s} n={r['n_lines']:2d} mains={[l['hz'] for l in c['mains_50_100_150']]} "
          f"rig={[l['hz'] for l in c['rig_44_47_55']]} spur140-180={[l['hz'] for l in c['spurious_140_180']]}")
    top = r['lines'][:5]
    print('     top:', [(l['hz'], l['prom_db']) for l in top])
print('=== SYNTH line inventory ===')
for g, r in out['synth_groups'].items():
    c = r['classified']
    print(f"{g:20s} n={r['n_lines']:2d} mains={[l['hz'] for l in c['mains_50_100_150']]} "
          f"mach4-80={[(l['hz'],l['prom_db']) for l in c['machinery_4_80'][:6]]} "
          f"spur140-180={[(l['hz'],l['prom_db']) for l in c['spurious_140_180']]}")
