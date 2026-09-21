"""sp_coupling.py -- Task 3 (coupling resonance bump ~50-65 Hz per real session vs
synth realized coupling) + Task 4 (spurious 140-180 Hz resonance).
Method: fit smooth log-log baseline to each PSD, take dB residual, find the
dominant residual bump in a target band. Cross-checks coupling_verify.json and
queries scenes.coupling_fc. Writes sp_out_coupling.json.
"""
import os, json, sqlite3
import numpy as np
from scipy.ndimage import median_filter
import sp_lib as L


def residual_db(f, p, fit_lo=5.0, fit_hi=450.0, deg=4, suppress_lines=True):
    """dB residual over a smooth log-log baseline. If suppress_lines, first
    replace narrow spectral lines by a local-median envelope (~5 Hz kernel) so
    the residual reflects the BROAD coupling resonance, not mains/machinery lines."""
    m = (f >= fit_lo) & (f <= fit_hi) & (p > 0)
    ff = f[m]; db = 10 * np.log10(p[m])
    lf = np.log10(ff)
    if suppress_lines:
        df = np.median(np.diff(ff))
        k = int(round(5.0 / df)) | 1  # ~5 Hz median envelope kills narrow lines
        db = median_filter(db, size=max(3, k), mode='nearest')
    coef = np.polyfit(lf, db, deg)
    base = np.polyval(coef, lf)
    return ff, db - base


def find_bump(f, resid, band):
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3:
        return None
    ff = f[m]; rr = resid[m]
    i = int(np.argmax(rr))
    peak_hz = float(ff[i]); peak_db = float(rr[i])
    half = peak_db - 3.0
    lo = peak_hz; hi = peak_hz
    j = i
    while j > 0 and rr[j] >= half:
        lo = ff[j]; j -= 1
    j = i
    while j < len(rr) - 1 and rr[j] >= half:
        hi = ff[j]; j += 1
    return {'center_hz': round(peak_hz, 1), 'height_db': round(peak_db, 2),
            'halfwidth_hz': [round(lo, 1), round(hi, 1)],
            'width_hz': round(hi - lo, 1)}


d = L.load_cache()
out = {'method': 'log-log deg4 baseline, dB residual bump in target band',
       'coupling_band_hz': [25, 90], 'spurious_band_hz': [140, 180],
       'real': {}, 'synth': {}}

# real: use 1-Hz median PSDs for floors + active
real_groups = ['real_human_nothing', 'real_car_nothing', 'real_field',
               'real_human_active', 'real_car_active']
def group_bumps(f0, p0):
    fb, rb = residual_db(f0, p0, suppress_lines=True)   # broad coupling envelope
    fl, rl = residual_db(f0, p0, suppress_lines=False)  # keep narrow 150 line
    return {'coupling_bump_25_90': find_bump(fb, rb, (25, 90)),
            'spurious_bump_140_180': find_bump(fl, rl, (140, 180))}


for g in real_groups:
    out['real'][g] = group_bumps(d[f'{g}__f'], d[f'{g}__psd'])

# synth nothing groups
for g in ['synth_nothing_calm', 'synth_nothing_wind', 'synth_nothing_rain',
          'synth_human', 'synth_vehicle']:
    out['synth'][g] = group_bumps(d[f'{g}__f'], d[f'{g}__psd'])

# per-scene synth realized coupling bump (median washes it; measure per-scene)
con = sqlite3.connect(L.DB); cur = con.cursor()
ids = L.sample_scene_ids(cur, "coarse='nothing'", n=200)
centers = []; heights = []; spur_h = []
for sid in ids:
    _, _, mi = L.synth_scene_signals(cur, sid)
    fh, ph = L.hr_psd(mi)
    fb, rb = residual_db(fh, ph, suppress_lines=True)
    fl, rl = residual_db(fh, ph, suppress_lines=False)
    b = find_bump(fb, rb, (25, 90))
    if b and b['height_db'] > 3:
        centers.append(b['center_hz']); heights.append(b['height_db'])
    s = find_bump(fl, rl, (140, 180))
    if s:
        spur_h.append(s['height_db'])
centers = np.array(centers); heights = np.array(heights); spur_h = np.array(spur_h)

# coupling_fc parameter distribution (the intended realized coupling)
fcs = d['synth_coupling_fc'].astype(float)
out['synth_coupling_fc_param'] = {
    'n': int(fcs.size),
    'p10_p25_p50_p75_p90': [round(float(np.percentile(fcs, q)), 1) for q in (10, 25, 50, 75, 90)],
    'min': round(float(fcs.min()), 1), 'max': round(float(fcs.max()), 1),
    'frac_50_65hz': round(float(((fcs >= 50) & (fcs <= 65)).mean()), 3),
}
out['synth_per_scene_bump'] = {
    'n_scenes': len(ids),
    'coupling_bump_scene_frac': round(len(centers) / len(ids), 3),
    'coupling_center_hz_p25_p50_p75': [round(float(np.percentile(centers, q)), 1) for q in (25, 50, 75)] if centers.size else None,
    'coupling_height_db_p50_p90': [round(float(np.percentile(heights, q)), 2) for q in (50, 90)] if heights.size else None,
    'spurious_140_180_height_db_p50_p90': [round(float(np.percentile(spur_h, q)), 2) for q in (50, 90)] if spur_h.size else None,
}
con.close()

# cross-check coupling_verify.json
cvpath = os.path.join(L.BASE, 'dataset_validation', 'v4_plan', 'coupling_verify.json')
if os.path.exists(cvpath):
    cv = json.load(open(cvpath))
    out['coupling_verify_prior'] = {k: {'peak_hz': v.get('peak_hz'), 'residual_db': v.get('residual_db'),
                                        'halfwidth_hz': v.get('halfwidth_hz')}
                                    for k, v in cv['files'].items() if v}
    out['coupling_verify_FC_ANCHOR_HZ'] = cv.get('FC_ANCHOR_HZ')

with open(os.path.join(L.OUTDIR, 'sp_out_coupling.json'), 'w') as fh:
    json.dump(out, fh, indent=1)

print('=== REAL coupling / spurious bumps ===')
for g, r in out['real'].items():
    print(f"{g:20s} coupling={r['coupling_bump_25_90']}  spur140-180={r['spurious_bump_140_180']}")
print('=== SYNTH (median) bumps ===')
for g, r in out['synth'].items():
    print(f"{g:20s} coupling={r['coupling_bump_25_90']}  spur140-180={r['spurious_bump_140_180']}")
print('=== SYNTH coupling_fc param ===', out['synth_coupling_fc_param'])
print('=== SYNTH per-scene realized bump ===', out['synth_per_scene_bump'])
