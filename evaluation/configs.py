"""Run definitions for the training campaign.

This file defines every model configuration that was trained. The evaluation
scripts look up run configurations by ID from the RUNS list. For new training,
use training/presets.yaml instead — this file exists for reproducibility of
the reported experiments.
"""
import copy

# the current production model, unchanged -- every other entry is this with one field moved
BASELINE = dict(
    kind="snn",
    widths=[512, 256, 128],
    neuron="plif",
    tau=2.0,
    T=4,
    norm="bn",
    dropout=0.1,
    readout="mean",
    v_threshold=1.0,
    reset="hard",
    surrogate="atan",
    surrogate_alpha=2.0,
    gate="on",
    heads="3head",
    input="static",
    stem_once=False,
    recurrent=None,
    attention=None,
    features="all132",
    window="3s",
    ann_passes=1,
    lr=5e-4,
    batch=4096,
    max_steps=40000,
    warmup=1000,
    weight_decay=1e-4,
    ema_decay=0.999,
    patience=12,
    eval_every=500,
)


def _cfg(run_id, group, label, **over):
    c = copy.deepcopy(BASELINE)
    c.update(over)
    c["run_id"] = run_id
    c["group"] = group
    c["label"] = label
    return c


RUNS = []


def add(run_id, group, label, **over):
    RUNS.append(_cfg(run_id, group, label, **over))


# ---------------------------------------------------------------- A. does spiking earn its place
# Run first: cheapest, and the answer reframes everything after it. The two ordinary-network
# controls are what make this section publishable rather than an assertion.
add("A1_baseline", "A", "current model, unchanged")
add("A2_ann_1pass", "A", "ordinary activation, one pass", kind="ann", T=1, ann_passes=1)
add("A3_snn_T1", "A", "spiking, one time step", T=1)
add("A4_T2", "A", "two time steps", T=2)
add("A5_T8", "A", "eight time steps", T=8)
add("A6_T16", "A", "sixteen time steps", T=16)
add("A7_ann_4pass", "A", "ordinary network, four passes averaged", kind="ann", ann_passes=4)
add("A8_stem_once", "A", "first layer computed once and broadcast", stem_once=True)

# ---------------------------------------------------------------- B. head structure
add("B1_3head", "B", "three independent heads, shared trunk")
add("B2_2head_neg", "B", "two heads, animals as negatives", heads="2head_neg")
add("B3_2head_noanimal", "B", "two heads, animal scenes removed", heads="2head_noanimal")
add("B4_fourway", "B", "one four-way mutually exclusive output", heads="4way")
add("B5_private", "B", "three heads, each with a private layer", heads="private")
add("B6_separate", "B", "three fully separate networks", heads="separate")
add("B7_uncertainty", "B", "learnable per-head loss weights", head_weighting="learned")
add("B8_corn", "B", "published conditional ordinal loss", ordinal_loss="corn")
add("B9_plus_any", "B", "extra 'something is there' head", heads="plus_any")
add("B10_veh_multi", "B", "vehicle single-versus-several re-measured", vehicle_ordinal=True)
add("B11_multi_report", "B", "person and animal single-versus-several re-measured",
    report_multiplicity=True)

# ---------------------------------------------------------------- D. neuron type
for nid, (nm, lbl) in enumerate({
    "if": "integrate-and-fire, no leak",
    "lif": "leaky, fixed leak",
    "klif": "K-leaky",
    "eif": "exponential",
    "qif": "quadratic",
    "izhikevich": "Izhikevich",
    "liaf": "analog output, spike-gated",
    "adaptive": "adaptive leaky",
}.items(), start=1):
    add(f"D{nid}_{nm}", "D", lbl, neuron=nm)


# ---------------------------------------------------------------- E. trunk and plumbing
add("E1_narrow", "E", "narrow trunk", widths=[256, 128, 64])
add("E2_wide", "E", "wide trunk", widths=[1024, 512, 256])
add("E3_depth2", "E", "two layers", widths=[512, 256])
add("E4_depth4", "E", "four layers", widths=[512, 512, 256, 128])
add("E5_depth5", "E", "five layers", widths=[512, 512, 256, 128, 64])
add("E6_tdbn", "E", "threshold-scaled normalization", norm="tdbn")
add("E7_groupnorm", "E", "group normalization", norm="gn")
add("E8_nonorm", "E", "no normalization", norm="none")
add("E9_gate_off", "E", "no per-feature scale", gate="off")
add("E10_gate_sparse", "E", "per-feature scale pushed toward sparsity", gate="sparse")
add("E11_attn_time", "E", "attention over time steps", attention="time")
add("E12_readout_last", "E", "read the last step only", readout="last")
add("E13_readout_count", "E", "read the spike count", readout="count")
add("E14_readout_vote", "E", "population vote readout", readout="vote")
add("E15_loss_per_step", "E", "loss applied at every step", loss_per_step=True)
add("E16_surr_alpha4", "E", "gradient stand-in, steepness four", surrogate_alpha=4.0)
add("E17_surr_sigmoid", "E", "sigmoid gradient stand-in", surrogate="sigmoid", surrogate_alpha=4.0)
add("E18_surr_piecewise", "E", "piecewise gradient stand-in", surrogate="piecewise")
add("E19_thr_low", "E", "firing threshold 0.5", v_threshold=0.5)
add("E20_thr_high", "E", "firing threshold 2.0", v_threshold=2.0)
add("E21_thr_learn", "E", "learnable firing threshold", v_threshold="learnable")
add("E22_soft_reset", "E", "soft reset", reset="soft")
add("E23_fire_penalty", "E", "firing-rate penalty toward 10%", fire_penalty=1e-3, fire_target=0.1)
add("E24_dropout0", "E", "no dropout", dropout=0.0)
add("E25_dropout2", "E", "dropout 0.2", dropout=0.2)
add("E26_tau1", "E", "faster leak", tau=1.2)
add("E27_tau4", "E", "slower leak", tau=4.0)

# ---------------------------------------------------------------- G. feature set
add("G1_sel104", "G", "the historical 104-feature selection", features="sel104")
add("G2_rank_scaled", "G", "rank-transformed ratio features", features="all132", scaling="rank")
for gi, grp in enumerate(["legacy32", "phys7", "cad12", "eng8", "wx2", "cep16", "wpe16", "ar8",
                          "nl1", "v2new30"], start=3):
    add(f"G{gi}_upto_{grp}", "G", f"cumulative feature groups up to {grp}",
        features=f"cumulative:{grp}")

# ---------------------------------------------------------------- H. fruit-fly front end
add("H1_flyhash", "H", "sparse random expansion plus top 5%, fixed", frontend="flyhash")
add("H2_flyhash_wta", "H", "same, with lateral inhibition over time steps",
    frontend="flyhash_wta", input="sequence", T=8)

# ---------------------------------------------------------------- C. make time mean something
# Needs the per-piece feature extraction, so these run after the cheap groups.
add("C1_pieces4", "C", "four pieces, four steps", input="sequence", T=4, window="p4")
add("C2_pieces8", "C", "eight pieces, eight steps", input="sequence", T=8, window="p8")
add("C3_pieces16", "C", "sixteen pieces, sixteen steps", input="sequence", T=16, window="p16")
add("C4_overlap", "C", "overlapping pieces", input="sequence", T=4, window="p4o")
add("C5_recur_linear", "C", "plus a recurrent trunk", input="sequence", T=4, window="p4",
    recurrent="linear")
add("C6_recur_elem", "C", "plus elementwise feedback", input="sequence", T=4, window="p4",
    recurrent="elementwise")
add("C7_adaptive", "C", "plus adaptive neurons", input="sequence", T=4, window="p4",
    neuron="adaptive")
add("C8_attn", "C", "plus attention over steps", input="sequence", T=4, window="p4",
    attention="time")

# ---------------------------------------------------------------- M. ordinary machine learning
# Minutes in total. Gives every neural result a floor and a ceiling, and the nearest-neighbour
# model doubles as the leak detector: if it scores near one, the split is leaking.
for mid, (nm, lbl) in enumerate({
    "logreg": "logistic regression",
    "svm_linear": "linear support-vector machine",
    "svm_rbf": "support-vector machine, radial kernel",
    "forest": "random forest",
    "boosted": "gradient-boosted trees",
    "knn": "nearest neighbours (leak detector)",
    "mlp": "small ordinary network, matched size",
}.items(), start=1):
    add(f"M{mid}_{nm}", "M", lbl, kind="classical", model=nm)

# ---------------------------------------------------------------- F. raw waveform
# Last. Roughly five hours each, so these run only if the window allows and are skipped cleanly
# if it does not.
add("F1_conv", "F", "convolution network on the raw signal", kind="conv", T=8, batch=256,
    conv_widths=[64, 128, 256, 512, 512], window="raw400")
add("F2_conv_adaptive", "F", "same, adaptive neurons", kind="conv", T=8, batch=256,
    neuron="adaptive", conv_widths=[64, 128, 256, 512, 512], window="raw400")
add("F3_conv_ann", "F", "ordinary convolution control", kind="conv", T=1, batch=256,
    spiking=False, conv_widths=[64, 128, 256, 512, 512], window="raw400")
add("F4_conv_hybrid", "F", "raw signal plus features", kind="conv", T=8, batch=256,
    hybrid=True, conv_widths=[64, 128, 256, 512, 512], window="raw400")


# ---------------------------------------------------------------- R. replicates: the noise floor
# Identical configurations trained again. Same seed on purpose: the review found seven accidental
# copies of the baseline spread 3.6 points at the same seed, so the floor to measure is the
# run-to-run variation the seed does not control.
for _i in range(1, 6):
    add(f"R_A1_rep{_i}", "R", f"baseline, replicate {_i}")
    add(f"R_B3_rep{_i}", "R", f"two heads, animal scenes removed, replicate {_i}",
        heads="2head_noanimal")
    add(f"R_G8_rep{_i}", "R", f"features up to cepstral, replicate {_i}",
        features="cumulative:cep16")
    add(f"R_A2_rep{_i}", "R", f"ordinary network, one pass, replicate {_i}",
        kind="ann", T=1, ann_passes=1)

# ---------------------------------------------------------------- S. size ladder
# Two heads (animal scenes removed) on the 77 cepstral-cut features, the two changes the review
# found to hold, with the trunk shrunk step by step. Fixed budget: 20,000 steps, schedule ending
# there, no early stopping, final weights kept. Three replicates per rung.
LADDER = [("16x8", [16, 8]), ("32x16", [32, 16]), ("64x32", [64, 32]), ("128x64", [128, 64]),
          ("256x128", [256, 128]), ("512x256x128", [512, 256, 128])]
for _i in range(1, 4):
    for _k, (_nm, _w) in enumerate(LADDER, start=1):
        add(f"S{_k}_{_nm}_rep{_i}", "S", f"size ladder {_nm}, replicate {_i}",
            widths=_w, heads="2head_noanimal", features="cumulative:cep16",
            max_steps=20000, early_stop=False, select="last")

# ---------------------------------------------------------------- T. tuning the 54k winner
# Everything the ladder held fixed, moved one at a time around the 256-128 rung: per-head feature
# sets (two separate small networks), fewer inputs, time steps, fixed leak, double budget, and
# depth at the same parameter count. Same fixed-budget protocol, three replicates each.
# TOP40 / TOP20: the cepstral-cut features ranked by single-feature separation on synthetic
# validation (max over the person and vehicle heads), kept in bank order. Synthetic data only.
TOP40 = ['energy_low_freq', 'energy_car_approach', 'energy_car_peak', 'energy_car_tail', 'energy_mid_gap', 'energy_human_peak', 'energy_human_tail', 'energy_high_freq', 'rms_total', 'peak_to_peak', 'variance', 'kurtosis', 'spectral_flatness', 'spectral_entropy', 'dominant_freq', 'activity_concentration', 'temporal_entropy', 'log_rms', 'frac_wind_1_5', 'frac_veh_5_25', 'frac_foot_20_90', 'ratio_veh_foot', 'cad_salience', 'cad_harm4', 'mod_peak_count', 'env_entropy', 'comb_salience', 'n_lines', 'line_stability', 'hop_band_frac', 'tonality_max', 'highband_kurtosis', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_6', 'lfcc_1', 'lfcc_2', 'lfcc_4', 'lfcc_7']
TOP20 = ['energy_low_freq', 'energy_car_approach', 'rms_total', 'variance', 'kurtosis', 'spectral_entropy', 'dominant_freq', 'temporal_entropy', 'log_rms', 'frac_wind_1_5', 'frac_foot_20_90', 'ratio_veh_foot', 'cad_salience', 'env_entropy', 'comb_salience', 'line_stability', 'highband_kurtosis', 'mfcc_3', 'lfcc_1', 'lfcc_2']
_LADDER = dict(heads="2head_noanimal", features="cumulative:cep16", widths=[256, 128],
               max_steps=20000, early_stop=False, select="last")
_TUNE = [
    ("T1_sep77", "two separate 128-64 networks, both on the 77 features",
     dict(_LADDER, heads="separate2", features="all132", widths=[128, 64],
          head_features={"human": "cumulative:cep16", "vehicle": "cumulative:cep16"})),
    ("T2_sep_veh132", "two separate 128-64 networks, vehicle on all 132 features",
     dict(_LADDER, heads="separate2", features="all132", widths=[128, 64],
          head_features={"human": "cumulative:cep16", "vehicle": "all132"})),
    ("T3_top40", "256-128 on the top 40 features", dict(_LADDER, features=TOP40)),
    ("T4_top20", "256-128 on the top 20 features", dict(_LADDER, features=TOP20)),
    ("T5_T8", "256-128, eight time steps", dict(_LADDER, T=8)),
    ("T6_T1", "256-128, one time step", dict(_LADDER, T=1)),
    ("T7_lif", "256-128, fixed leak", dict(_LADDER, neuron="lif")),
    ("T8_40k", "256-128, double budget", dict(_LADDER, max_steps=40000)),
    ("T9_128x128x64", "three layers at 35k", dict(_LADDER, widths=[128, 128, 64])),
    ("T10_128x128x128x64", "four layers at 51k", dict(_LADDER, widths=[128, 128, 128, 64])),
]
for _i in range(1, 4):
    for _rid, _lbl, _over in _TUNE:
        add(f"{_rid}_rep{_i}", "T", f"{_lbl}, replicate {_i}", **_over)

# ---------------------------------------------------------------- U. under 10k parameters
# How small the model can go while holding the 54k model's accuracy, without distillation:
# which of the six feature groups inside the 77 the 7.4k model can do without (leave one out),
# a low-rank first layer, small-network training hygiene, and shape at fixed size.
_SMALL = dict(heads="2head_noanimal", features="cumulative:cep16", widths=[64, 32],
              max_steps=20000, early_stop=False, select="last")
_LOO = {'legacy32': ['log_rms', 'frac_wind_1_5', 'frac_veh_5_25', 'frac_foot_20_90', 'frac_high_90_180', 'centroid_foot_band', 'ratio_veh_foot', 'cad_freq', 'cad_salience', 'cad_harm2', 'cad_harm4', 'footfall_rate', 'mod_peak_count', 'env_entropy', 'env_cv', 'env_ac_strength', 'iii_mean', 'iii_cv', 'gust_mod', 'comb_salience', 'comb_f0', 'n_lines', 'line_stability', 'hop_band_frac', 'beat_depth', 'line_mains', 'tonality_max', 'rain_impulse_rate', 'highband_kurtosis', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'lfcc_1', 'lfcc_2', 'lfcc_3', 'lfcc_4', 'lfcc_5', 'lfcc_6', 'lfcc_7', 'lfcc_8'], 'phys7': ['energy_low_freq', 'energy_car_approach', 'energy_car_peak', 'energy_car_tail', 'energy_mid_gap', 'energy_human_peak', 'energy_human_tail', 'energy_high_freq', 'rms_total', 'peak_to_peak', 'variance', 'zcr', 'kurtosis', 'skewness', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness', 'spectral_entropy', 'dominant_freq', 'sta_lta_ratio', 'event_count', 'mean_burst_len', 'activity_concentration', 'temporal_entropy', 'burst_efficiency', 'autocorr_250', 'autocorr_500', 'ratio_car_human', 'ratio_human_car', 'centroid_car_band', 'centroid_human_band', 'cad_freq', 'cad_salience', 'cad_harm2', 'cad_harm4', 'footfall_rate', 'mod_peak_count', 'env_entropy', 'env_cv', 'env_ac_strength', 'iii_mean', 'iii_cv', 'gust_mod', 'comb_salience', 'comb_f0', 'n_lines', 'line_stability', 'hop_band_frac', 'beat_depth', 'line_mains', 'tonality_max', 'rain_impulse_rate', 'highband_kurtosis', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'lfcc_1', 'lfcc_2', 'lfcc_3', 'lfcc_4', 'lfcc_5', 'lfcc_6', 'lfcc_7', 'lfcc_8'], 'cad12': ['energy_low_freq', 'energy_car_approach', 'energy_car_peak', 'energy_car_tail', 'energy_mid_gap', 'energy_human_peak', 'energy_human_tail', 'energy_high_freq', 'rms_total', 'peak_to_peak', 'variance', 'zcr', 'kurtosis', 'skewness', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness', 'spectral_entropy', 'dominant_freq', 'sta_lta_ratio', 'event_count', 'mean_burst_len', 'activity_concentration', 'temporal_entropy', 'burst_efficiency', 'autocorr_250', 'autocorr_500', 'ratio_car_human', 'ratio_human_car', 'centroid_car_band', 'centroid_human_band', 'log_rms', 'frac_wind_1_5', 'frac_veh_5_25', 'frac_foot_20_90', 'frac_high_90_180', 'centroid_foot_band', 'ratio_veh_foot', 'comb_salience', 'comb_f0', 'n_lines', 'line_stability', 'hop_band_frac', 'beat_depth', 'line_mains', 'tonality_max', 'rain_impulse_rate', 'highband_kurtosis', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'lfcc_1', 'lfcc_2', 'lfcc_3', 'lfcc_4', 'lfcc_5', 'lfcc_6', 'lfcc_7', 'lfcc_8'], 'eng8': ['energy_low_freq', 'energy_car_approach', 'energy_car_peak', 'energy_car_tail', 'energy_mid_gap', 'energy_human_peak', 'energy_human_tail', 'energy_high_freq', 'rms_total', 'peak_to_peak', 'variance', 'zcr', 'kurtosis', 'skewness', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness', 'spectral_entropy', 'dominant_freq', 'sta_lta_ratio', 'event_count', 'mean_burst_len', 'activity_concentration', 'temporal_entropy', 'burst_efficiency', 'autocorr_250', 'autocorr_500', 'ratio_car_human', 'ratio_human_car', 'centroid_car_band', 'centroid_human_band', 'log_rms', 'frac_wind_1_5', 'frac_veh_5_25', 'frac_foot_20_90', 'frac_high_90_180', 'centroid_foot_band', 'ratio_veh_foot', 'cad_freq', 'cad_salience', 'cad_harm2', 'cad_harm4', 'footfall_rate', 'mod_peak_count', 'env_entropy', 'env_cv', 'env_ac_strength', 'iii_mean', 'iii_cv', 'gust_mod', 'rain_impulse_rate', 'highband_kurtosis', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'lfcc_1', 'lfcc_2', 'lfcc_3', 'lfcc_4', 'lfcc_5', 'lfcc_6', 'lfcc_7', 'lfcc_8'], 'wx2': ['energy_low_freq', 'energy_car_approach', 'energy_car_peak', 'energy_car_tail', 'energy_mid_gap', 'energy_human_peak', 'energy_human_tail', 'energy_high_freq', 'rms_total', 'peak_to_peak', 'variance', 'zcr', 'kurtosis', 'skewness', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness', 'spectral_entropy', 'dominant_freq', 'sta_lta_ratio', 'event_count', 'mean_burst_len', 'activity_concentration', 'temporal_entropy', 'burst_efficiency', 'autocorr_250', 'autocorr_500', 'ratio_car_human', 'ratio_human_car', 'centroid_car_band', 'centroid_human_band', 'log_rms', 'frac_wind_1_5', 'frac_veh_5_25', 'frac_foot_20_90', 'frac_high_90_180', 'centroid_foot_band', 'ratio_veh_foot', 'cad_freq', 'cad_salience', 'cad_harm2', 'cad_harm4', 'footfall_rate', 'mod_peak_count', 'env_entropy', 'env_cv', 'env_ac_strength', 'iii_mean', 'iii_cv', 'gust_mod', 'comb_salience', 'comb_f0', 'n_lines', 'line_stability', 'hop_band_frac', 'beat_depth', 'line_mains', 'tonality_max', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'lfcc_1', 'lfcc_2', 'lfcc_3', 'lfcc_4', 'lfcc_5', 'lfcc_6', 'lfcc_7', 'lfcc_8'], 'cep16': ['energy_low_freq', 'energy_car_approach', 'energy_car_peak', 'energy_car_tail', 'energy_mid_gap', 'energy_human_peak', 'energy_human_tail', 'energy_high_freq', 'rms_total', 'peak_to_peak', 'variance', 'zcr', 'kurtosis', 'skewness', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness', 'spectral_entropy', 'dominant_freq', 'sta_lta_ratio', 'event_count', 'mean_burst_len', 'activity_concentration', 'temporal_entropy', 'burst_efficiency', 'autocorr_250', 'autocorr_500', 'ratio_car_human', 'ratio_human_car', 'centroid_car_band', 'centroid_human_band', 'log_rms', 'frac_wind_1_5', 'frac_veh_5_25', 'frac_foot_20_90', 'frac_high_90_180', 'centroid_foot_band', 'ratio_veh_foot', 'cad_freq', 'cad_salience', 'cad_harm2', 'cad_harm4', 'footfall_rate', 'mod_peak_count', 'env_entropy', 'env_cv', 'env_ac_strength', 'iii_mean', 'iii_cv', 'gust_mod', 'comb_salience', 'comb_f0', 'n_lines', 'line_stability', 'hop_band_frac', 'beat_depth', 'line_mains', 'tonality_max', 'rain_impulse_rate', 'highband_kurtosis']}
_UNDER = [
    *[(f"U1_no_{g}", f"7.4k without the {g} group", dict(_SMALL, features=_LOO[g]))
      for g in ["legacy32", "phys7", "cad12", "eng8", "wx2", "cep16"]],
    ("U2_lowrank16", "7.4k with the first layer factored through 16", dict(_SMALL, low_rank=16)),
    ("U3_lowrank8", "7.4k with the first layer factored through 8", dict(_SMALL, low_rank=8)),
    ("U4_nodrop_64x32", "7.4k, no dropout", dict(_SMALL, dropout=0.0)),
    ("U5_lr2e3_64x32", "7.4k, learning rate 2e-3", dict(_SMALL, lr=2e-3)),
    ("U6_nodrop_lr2e3_64x32", "7.4k, no dropout and learning rate 2e-3", dict(_SMALL, dropout=0.0, lr=2e-3)),
    ("U7_nodrop_16x8", "1.5k, no dropout", dict(_SMALL, widths=[16, 8], dropout=0.0)),
    ("U8_96x16", "96-16, about 9k", dict(_SMALL, widths=[96, 16])),
    ("U9_48x48", "48-48, about 6k", dict(_SMALL, widths=[48, 48])),
]
for _i in range(1, 4):
    for _rid, _lbl, _over in _UNDER:
        add(f"{_rid}_rep{_i}", "U", f"{_lbl}, replicate {_i}", **_over)

# ---------------------------------------------------------------- V. heads of the 4.8k model
# The 64-32/16 model with every way of laying out its heads: with and without the "more than
# one" output on each head, and the three ways of handling animal scenes -- removed from
# training (the base model), kept as negatives, or given their own head.
_TINY = dict(heads="2head_noanimal", features="cumulative:cep16", widths=[64, 32], low_rank=16,
             max_steps=20000, early_stop=False, select="last")
_HEADS = [
    ("V1_nomult", "person and vehicle, presence only",
     dict(_TINY, head_sizes={"human": 1, "vehicle": 1})),
    ("V2_mult_both", "person and vehicle, both with more-than-one",
     dict(_TINY, head_sizes={"human": 2, "vehicle": 2})),
    ("V3_animal_nomult", "person, vehicle and animal heads, presence only",
     dict(_TINY, heads="3head", head_sizes={"human": 1, "animal": 1, "vehicle": 1})),
    ("V4_animal_mult", "person, vehicle and animal heads, all with more-than-one",
     dict(_TINY, heads="3head", head_sizes={"human": 2, "animal": 2, "vehicle": 2})),
    ("V5_animal_neg", "person and vehicle, animal scenes kept as negatives",
     dict(_TINY, heads="2head_neg")),
]
for _i in range(1, 4):
    for _rid, _lbl, _over in _HEADS:
        add(f"{_rid}_rep{_i}", "V", f"{_lbl}, replicate {_i}", **_over)

# ---------------------------------------------------------------- W. the 4.8k model on more features
# The same 64-32/16 recipe fed all 132 features, and fed the catalog's approved selection
# (the historical scorecard: discriminative and representative), instead of the 77 cepstral cut.
_W = dict(heads="2head_noanimal", widths=[64, 32], low_rank=16, max_steps=20000, early_stop=False,
          select="last")
for _i in range(1, 4):
    add(f"W1_all132_rep{_i}", "W", f"64-32/16 on all 132 features, replicate {_i}", **dict(_W, features="all132"))
    add(f"W2_sel104_rep{_i}", "W", f"64-32/16 on the catalog's approved features, replicate {_i}",
        **dict(_W, features="sel104"))

# ---------------------------------------------------------------- X. conventional models
# The same recipe (77 features, two heads, fixed budget, three replicates) for non-spiking
# models at the three sizes of the spiking story: the 4.8k factored model, the 54k dense
# winner, and the 206k top rung. Twins keep the layer shape with ordinary units; the sequence
# models read the five consecutive windows (7.5 s of context, 6 s of it history) ending at the
# current one, which is the context hysteresis exploits after the fact.
_X = dict(heads="2head_noanimal", features="cumulative:cep16", max_steps=20000, early_stop=False,
          select="last")
_CONV = [
    ("X1_ann_64x32_lr16", "non-spiking twin of the 4.8k model (64-32, first layer factored through 16)",
     dict(_X, kind="ann", T=1, ann_passes=1, widths=[64, 32], low_rank=16)),
    ("X2_ann_256x128", "non-spiking twin of the 54k model (256-128)",
     dict(_X, kind="ann", T=1, ann_passes=1, widths=[256, 128])),
    ("X3_ann_512x256x128", "non-spiking twin of the 206k model (512-256-128)",
     dict(_X, kind="ann", T=1, ann_passes=1, widths=[512, 256, 128])),
    ("X4_gru16", "recurrent (GRU, state 16) over 5 windows, about 4.7k parameters",
     dict(_X, kind="gru", context=5, d_model=16, widths=[16])),
    ("X5_gru100", "recurrent (GRU, state 100) over 5 windows, about 54k parameters",
     dict(_X, kind="gru", context=5, d_model=100, widths=[100])),
    ("X6_gru225", "recurrent (GRU, state 225) over 5 windows, about 206k parameters",
     dict(_X, kind="gru", context=5, d_model=225, widths=[225])),
    ("X7_attn16", "transformer encoder (width 16, one layer) over 5 windows, about 4.8k parameters",
     dict(_X, kind="attn", context=5, d_model=16, layers=1, n_heads=2, widths=[16])),
    ("X8_attn64", "transformer encoder (width 64, one layer) over 5 windows, about 56k parameters",
     dict(_X, kind="attn", context=5, d_model=64, layers=1, n_heads=4, widths=[64])),
    ("X9_attn128", "transformer encoder (width 128, one layer) over 5 windows, about 210k parameters",
     dict(_X, kind="attn", context=5, d_model=128, layers=1, n_heads=4, widths=[128])),
]
for _i in range(1, 4):
    for _id, _lab, _over in _CONV:
        add(f"{_id}_rep{_i}", "X", f"{_lab}, replicate {_i}", **_over)

# ---------------------------------------------------------------- Y. the 4.8k model's own knobs
# The negative results (time steps, fixed leak, dropout) were measured on the 54k and 235k
# models; these repeat them on the deployed 4.8k model so the report's section on it is about it.
_Y = dict(heads="2head_noanimal", features="cumulative:cep16", widths=[64, 32], low_rank=16,
          max_steps=20000, early_stop=False, select="last")
for _i in range(1, 4):
    add(f"Y1_T1_rep{_i}", "Y", f"4.8k model with one time step, replicate {_i}", **dict(_Y, T=1))
    add(f"Y2_lif_rep{_i}", "Y", f"4.8k model with a fixed leak (LIF), replicate {_i}", **dict(_Y, neuron="lif"))
    add(f"Y3_nodrop_rep{_i}", "Y", f"4.8k model without dropout, replicate {_i}", **dict(_Y, dropout=0.0))

# ---------------------------------------------------------------- Z. matched-band arm (200 Hz)
# The three sizes of every model kind, trained on synthetic features whose input was decimated to
# 200 Hz and brought back to 1 kHz, exactly what the 200 Hz real datasets go through. Scored on the
# vehicle drive-bys, the savanna set, the rail passes, and the field recordings band-limited the
# same way. Same recipe otherwise. Spiking three replicates; the conventional kinds are
# deterministic, two.
_Z = dict(heads="2head_noanimal", features="cumulative:cep16", max_steps=20000, early_stop=False,
          select="last", window="3s_bl200")
_ZK = [
    ("Z1_snn_S", "SNN-S on the 200 Hz band (64-32, first layer factored through 16)", dict(_Z, widths=[64, 32], low_rank=16), 3),
    ("Z2_snn_M", "SNN-M on the 200 Hz band (256-128)", dict(_Z, widths=[256, 128]), 3),
    ("Z3_snn_L", "SNN-L on the 200 Hz band (512-256-128)", dict(_Z, widths=[512, 256, 128]), 3),
    ("Z4_ann_S", "ANN-S, non-spiking twin, 200 Hz band", dict(_Z, kind="ann", T=1, ann_passes=1, widths=[64, 32], low_rank=16), 2),
    ("Z5_ann_M", "ANN-M, non-spiking twin, 200 Hz band", dict(_Z, kind="ann", T=1, ann_passes=1, widths=[256, 128]), 2),
    ("Z6_ann_L", "ANN-L, non-spiking twin, 200 Hz band", dict(_Z, kind="ann", T=1, ann_passes=1, widths=[512, 256, 128]), 2),
    ("Z7_gru_S", "GRU-S over 5 windows, 200 Hz band", dict(_Z, kind="gru", context=5, d_model=16, widths=[16]), 2),
    ("Z8_gru_M", "GRU-M over 5 windows, 200 Hz band", dict(_Z, kind="gru", context=5, d_model=100, widths=[100]), 2),
    ("Z9_gru_L", "GRU-L over 5 windows, 200 Hz band", dict(_Z, kind="gru", context=5, d_model=225, widths=[225]), 2),
    ("Z10_trf_S", "TRF-S over 5 windows, 200 Hz band", dict(_Z, kind="attn", context=5, d_model=16, layers=1, n_heads=2, widths=[16]), 2),
    ("Z11_trf_M", "TRF-M over 5 windows, 200 Hz band", dict(_Z, kind="attn", context=5, d_model=64, layers=1, n_heads=4, widths=[64]), 2),
    ("Z12_trf_L", "TRF-L over 5 windows, 200 Hz band", dict(_Z, kind="attn", context=5, d_model=128, layers=1, n_heads=4, widths=[128]), 2),
]
# ---------------------------------------------------------------- ZH. human-only detector
# The 4.8k model on the 200 Hz band with the vehicle head removed, two ways: vehicle scenes kept
# as negatives for the person head, or dropped from training entirely. Motivated by the savanna
# result, where the vehicle head fires on 88-95% of every window because that set's energy sits
# below 25 Hz, and the person head is the only one carrying signal.
_ZH = dict(heads="1head_human", features="cumulative:cep16", widths=[64, 32], low_rank=16,
           max_steps=20000, early_stop=False, select="last", window="3s_bl200")
for _i in range(1, 4):
    add(f"ZH1_human_negveh_rep{_i}", "ZH",
        f"4.8k person-only detector, vehicles kept as negatives, 200 Hz band, replicate {_i}", **_ZH)
    add(f"ZH2_human_noveh_rep{_i}", "ZH",
        f"4.8k person-only detector, vehicles removed from training, 200 Hz band, replicate {_i}",
        **dict(_ZH, heads="1head_human_noveh"))

# medium and large person-only detectors, same two variants as the 4.8k pair above
_ZHB = dict(features="cumulative:cep16", max_steps=20000, early_stop=False, select="last",
            window="3s_bl200")
_ZH_SIZES = [("M", [256, 128], None), ("L", [512, 256, 128], None)]
_ZH_VARIANTS = [("negveh", "1head_human", "vehicles kept as negatives"),
                ("noveh", "1head_human_noveh", "vehicles removed from training")]
_zh_n = 3
for _sz, _w, _lr in _ZH_SIZES:
    for _vn, _hl, _vd in _ZH_VARIANTS:
        _zh_n += 1
        for _i in range(1, 4):
            _o = dict(_ZHB, heads=_hl, widths=_w)
            if _lr:
                _o["low_rank"] = _lr
            add(f"ZH{_zh_n}_human_{_vn}_{_sz}_rep{_i}", "ZH",
                f"SNN-{_sz} person-only detector, {_vd}, 200 Hz band, replicate {_i}", **_o)

for _id, _lab, _over, _nrep in _ZK:
    for _i in range(1, _nrep + 1):
        add(f"{_id}_rep{_i}", "Z", f"{_lab}, replicate {_i}", **_over)
# the classical models on the same band, scored on the same sets
for _mid, (_nm, _lbl) in enumerate({
    "logreg": "logistic regression", "svm_linear": "linear support-vector machine",
    "svm_rbf": "kernel support-vector machine", "forest": "random forest",
    "boosted": "boosted trees", "knn": "one-neighbour lookup",
    "mlp": "small ordinary network, matched size",
}.items(), start=1):
    add(f"MZ{_mid}_{_nm}", "MZ", f"{_lbl}, 200 Hz band", kind="classical", model=_nm,
        features="cumulative:cep16", heads="2head_noanimal", window="3s_bl200")

# ---------------------------------------------------------------- winners, stage three
# Not part of the sweep: the loop appends these once the sweep has picked winners.
WINNER_FOLLOWUPS = dict(
    longer_context=["5s", "10s"],          # the test that has never actually been run
    finetune_fractions=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00],
    finetune_datasets=["savanna", "vehicles", "revibe"],   # the sets with a disjoint split
    real_only_control=True,                # same architecture, real data, no synthetic start
)

# how the winners are chosen -- stated here rather than decided by judgment later
WINNER_RULE = dict(metric="accuracy", dataset="elbit", n=3, tie_break="energy_ratio")


# Kept in the list, marked unusable, so the report can say it was considered and why it was not
# run rather than silently omitting it. Two faults in the shipped library: the constructor
# asserts on an attribute it never defines, and the multi-step path charges the membrane before
# converting it from a float to a tensor. Neither is something this project should patch around.
SKIP = {
    "D7_liaf": "analog-fire neuron is broken in the installed library -- the constructor asserts "
               "on an attribute it never defines, and the multi-step path charges the membrane "
               "before converting it from a float to a tensor",
}
# The conventional models train deterministically: replicate 2 reproduced replicate 1 bit for bit
# (same validation score to four decimals for all nine), so a third identical run adds nothing.
for _id, _lab, _over in _CONV:
    SKIP[f"{_id}_rep3"] = ("non-spiking runs are deterministic on this setup -- replicate 2 "
                          "reproduced replicate 1 exactly, so a third run would be a copy")
for _r in RUNS:
    if _r["run_id"] in SKIP:
        _r["skip"] = SKIP[_r["run_id"]]


def by_id(run_id):
    for c in RUNS:
        if c["run_id"] == run_id:
            return c
    raise KeyError(run_id)


def summary():
    from collections import Counter
    c = Counter(r["group"] for r in RUNS)
    return {"total": len(RUNS), "by_group": dict(sorted(c.items()))}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=1))
    for r in RUNS:
        print(f"{r['run_id']:<24} {r['group']:<2} {r['label']}")
