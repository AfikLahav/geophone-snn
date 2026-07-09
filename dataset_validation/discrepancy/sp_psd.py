"""sp_psd.py -- Task 1 (per-class PSD comparison, band fractions, log-log slope,
spectral centroid) + Task 5 (HF rolloff / effective bandwidth).
Writes sp_out_psd.json and PNG overlays sp_psd_human/vehicle/nothing.png.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sp_lib as L

d = L.load_cache()
BAND_LO, BAND_HI = L.BAND_LO, L.BAND_HI


def norm_psd(f, p):
    m = (f >= BAND_LO) & (f <= BAND_HI)
    area = np.trapz(p[m], f[m])
    return p / area if area > 0 else p


def group_metrics(g):
    f = d[f'{g}__f']; p = d[f'{g}__psd']
    m = (f >= BAND_LO) & (f <= BAND_HI)
    bf, total = L.band_fractions(f, p)
    slope = L.loglog_slope(f, p)
    cen = L.spectral_centroid(f, p)
    bw, floor = L.hf_rolloff(f, p)
    return {
        'n_windows': int(d[f'{g}__n']),
        'band_fractions': {k: round(v, 4) for k, v in bf.items()},
        'abs_area_mv2': round(float(total), 4),
        'loglog_slope_2_20hz': round(slope, 3),
        'spectral_centroid_5_180hz': round(cen, 2),
        'eff_bandwidth_hz': round(bw, 1),
        'hf_noise_floor_mv2hz': float(f"{floor:.4g}"),
    }


GROUPS = [
    'real_human_active', 'synth_human', 'synth_human_clean',
    'real_car_active', 'synth_vehicle', 'synth_vehicle_clean',
    'real_human_nothing', 'real_car_nothing', 'real_field',
    'synth_nothing_calm', 'synth_nothing_wind', 'synth_nothing_rain',
]
metrics = {g: group_metrics(g) for g in GROUPS}

# ---- discrepancy deltas (synth - real) for the headline comparisons ----
def band_delta(real_g, synth_g):
    r = metrics[real_g]['band_fractions']; s = metrics[synth_g]['band_fractions']
    return {k: round(s[k] - r[k], 4) for k in r}


deltas = {
    'human_modelinput_vs_real': band_delta('real_human_active', 'synth_human'),
    'human_clean_vs_real': band_delta('real_human_active', 'synth_human_clean'),
    'vehicle_modelinput_vs_real': band_delta('real_car_active', 'synth_vehicle'),
    'vehicle_clean_vs_real': band_delta('real_car_active', 'synth_vehicle_clean'),
    'nothing_calm_vs_human_floor': band_delta('real_human_nothing', 'synth_nothing_calm'),
    'nothing_calm_vs_car_floor': band_delta('real_car_nothing', 'synth_nothing_calm'),
}

# upper-band fraction check (recompute DOMAIN_GAP_V4 metric: frac of 20-90 Hz above 55)
def upper_band_frac(g):
    f = d[f'{g}__f']; p = d[f'{g}__psd']
    lo = (f >= 20) & (f <= 90); hi = (f >= 55) & (f <= 90)
    tot = np.trapz(p[lo], f[lo]); top = np.trapz(p[hi], f[hi])
    return float(top / tot) if tot > 0 else np.nan


upper = {g: round(upper_band_frac(g), 3) for g in
         ['real_human_active', 'synth_human', 'synth_human_clean',
          'real_car_active', 'synth_vehicle', 'synth_vehicle_clean']}

# ---- signal-excess: isolate event spectrum by subtracting paired rig floor ----
# power of (signal+noise) - power of noise ~= signal power (independent components).
def signal_excess(active_g, floor_g):
    f = d[f'{active_g}__f']; pa = d[f'{active_g}__psd']; pf = d[f'{floor_g}__psd']
    ex = np.clip(pa - pf, 0, None)
    bf, total = L.band_fractions(f, ex)
    cen = L.spectral_centroid(f, ex)
    slope = L.loglog_slope(f, ex)
    # upper-band frac 20-90 above 55 on the excess
    lo = (f >= 20) & (f <= 90); hi = (f >= 55) & (f <= 90)
    ub = float(np.trapz(ex[hi], f[hi]) / np.trapz(ex[lo], f[lo]))
    return {'band_fractions': {k: round(v, 4) for k, v in bf.items()},
            'spectral_centroid_5_180hz': round(cen, 2),
            'loglog_slope_2_20hz': round(slope, 3),
            'upper_band_frac_20_90_above_55': round(ub, 3)}, f, ex


excess = {}
excess['real_human_signal_excess'], fxh, exh = signal_excess('real_human_active', 'real_human_nothing')
excess['real_car_signal_excess'], fxc, exc = signal_excess('real_car_active', 'real_car_nothing')

out = {'config': {'fs': L.FS, 'win_s': L.WIN_S, 'hop_s': L.HOP_S,
                  'nperseg': L.NPERSEG, 'band_hz': [BAND_LO, BAND_HI],
                  'bands': [f'{a}-{b}' for a, b in L.BANDS],
                  'synth_snr_min_db': L.SYNTH_SNR_MIN,
                  'clip_mv': L.CLIP},
       'metrics': metrics, 'band_deltas_synth_minus_real': deltas,
       'upper_band_frac_20_90_above_55': upper,
       'signal_excess_active_minus_floor': excess}

with open(os.path.join(L.OUTDIR, 'sp_out_psd.json'), 'w') as fh:
    json.dump(out, fh, indent=1)


# ---------------- overlays ----------------
def overlay(fname, title, groups, styles, extras=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    curves = [(d[f'{g}__f'], d[f'{g}__psd'], st) for g, st in zip(groups, styles)]
    if extras:
        curves += extras
    for f, p, (c, ls, lab) in curves:
        m = (f >= BAND_LO) & (f <= BAND_HI)
        axes[0].loglog(f[m], p[m], c, ls=ls, lw=1.6, label=lab)
        pn = norm_psd(f, p)
        axes[1].semilogy(f[m], pn[m], c, ls=ls, lw=1.6, label=lab)
    for ax in axes:
        for bl in [5, 25, 50, 55, 90, 150, 180]:
            ax.axvline(bl, color='0.85', lw=0.6, zorder=0)
    axes[0].set_title(f'{title} -- absolute PSD'); axes[0].set_xlabel('Hz'); axes[0].set_ylabel('mV^2/Hz')
    axes[1].set_title(f'{title} -- normalized (area=1)'); axes[1].set_xlabel('Hz'); axes[1].set_ylabel('norm PSD')
    axes[1].set_xlim(1, 450)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(L.OUTDIR, fname), dpi=110)
    plt.close(fig)


overlay('sp_psd_human.png', 'HUMAN',
        ['real_human_active', 'synth_human', 'synth_human_clean'],
        [('C0', '-', 'real human active'), ('C1', '-', 'synth human (model-in, SNR>=12)'),
         ('C1', '--', 'synth human clean-only')],
        extras=[(fxh, exh, ('C2', ':', 'real human signal-excess (active-floor)'))])
overlay('sp_psd_vehicle.png', 'VEHICLE',
        ['real_car_active', 'synth_vehicle', 'synth_vehicle_clean'],
        [('C0', '-', 'real car active'), ('C3', '-', 'synth vehicle (model-in, SNR>=12)'),
         ('C3', '--', 'synth vehicle clean-only')],
        extras=[(fxc, exc, ('C2', ':', 'real car signal-excess (active-floor)'))])
overlay('sp_psd_nothing.png', 'NOTHING / FLOOR',
        ['real_human_nothing', 'real_car_nothing', 'real_field',
         'synth_nothing_calm', 'synth_nothing_wind', 'synth_nothing_rain'],
        [('C0', '-', 'real human-rig floor'), ('C2', '-', 'real car-rig floor'),
         ('0.4', ':', 'real field 2026-06'),
         ('C1', '-', 'synth calm'), ('C4', '-', 'synth wind'), ('C5', '-', 'synth rain')])

print('sp_psd done. groups:', len(GROUPS))
for g in GROUPS:
    mm = metrics[g]
    print(f"{g:22s} slope={mm['loglog_slope_2_20hz']:+.2f} centroid={mm['spectral_centroid_5_180hz']:6.1f} "
          f"bw={mm['eff_bandwidth_hz']:5.0f} bands={mm['band_fractions']}")
print('upper_band_frac_20_90_above_55:', upper)
print('SIGNAL-EXCESS (real active minus paired rig floor):')
for k, v in excess.items():
    print(f"  {k}: centroid={v['spectral_centroid_5_180hz']} ub55={v['upper_band_frac_20_90_above_55']} bands={v['band_fractions']}")
