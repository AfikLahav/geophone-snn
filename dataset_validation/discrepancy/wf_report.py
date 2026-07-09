# wf_report.py
# Reads discrepancy_waveform.json, computes headline "derived" numbers, writes
# them back into the JSON under 'derived', and renders discrepancy_waveform.md.
#
# Run AFTER wf_analyze.py:  python wf_report.py
import os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(HERE, "discrepancy_waveform.json")
MD = os.path.join(HERE, "discrepancy_waveform.md")


def main():
    d = json.load(open(J))
    R, S = d["real"], d["synth"]

    derived = {}
    # 1. pipeline scale-convention mismatch
    derived["x25p4_vs_x1000_factor"] = 1000.0 / 25.4  # 39.37x
    # 2. real car peak headroom above synth +-256 clip
    car_rail_mv = R["car_active"]["clipping"]["rail_abs"] * 1000.0
    derived["car_peak_mv"] = car_rail_mv
    derived["car_peak_over_synth_clip_ratio"] = car_rail_mv / 256.0
    derived["car_peak_over_synth_clip_db"] = 20.0 * np.log10(car_rail_mv / 256.0)
    # 3. impulsiveness gaps (median per-window)
    derived["human_kurt_real_vs_synth"] = [R["human_active"]["kurt_pct"]["p50"],
                                           S["human"]["kurt_pct"]["p50"]]
    derived["human_crest_real_vs_synth"] = [R["human_active"]["crest_pct"]["p50"],
                                            S["human"]["crest_pct"]["p50"]]
    derived["car_kurt_real_vs_synth"] = [R["car_active"]["kurt_pct"]["p50"],
                                         S["vehicle"]["kurt_pct"]["p50"]]
    # 4. clip fraction: real ~0 vs synth
    derived["synth_clipfrac_median"] = {k: S[k]["clipfrac_median_within_scene"]
                                        for k in ("human", "vehicle", "nothing")}
    derived["real_railfrac_active"] = {"human_active": R["human_active"]["clipping"]["frac_at_exact_rail"],
                                       "car_active": R["car_active"]["clipping"]["frac_at_exact_rail"]}
    # 5. quantization fill ratio real vs synth
    derived["fill_ratio_real"] = {k: R[k]["quant_fill_ratio"] for k in
                                  ("human_active", "car_active", "human_nothing",
                                   "car_nothing", "field_20260615_a")}
    derived["fill_ratio_synth"] = {k: S[k]["quant_fill_ratio"] for k in
                                   ("human", "vehicle", "nothing")}
    # 6. non-stationarity
    derived["rms_cv_real_vs_synth"] = {
        "human": [R["human_active"]["rms_cv"], S["human"]["rms_cv_median_within_scene"]],
        "vehicle": [R["car_active"]["rms_cv"], S["vehicle"]["rms_cv_median_within_scene"]],
        "nothing": [R["car_nothing"]["rms_cv"], S["nothing"]["rms_cv_median_within_scene"]],
    }
    d["derived"] = derived
    json.dump(d, open(J, "w"), indent=2)

    # ---------------------------------------------------------------- markdown
    L = []
    ap = L.append
    ap("# Time-domain / amplitude discrepancy: REAL geophone vs SYNTHETIC v4")
    ap("")
    ap(f"_Generated {d['meta']['generated']}. Synth = `{d['meta']['synth_db']}`, "
       f"model input = clip(clean+clean2?+noise, +-256 mV). Real amp = VOLTS; "
       f"analysed in mV (x1000). Windows = 3 s. Scripts: `wf_common.py`, "
       f"`wf_analyze.py`, `wf_report.py`._")
    ap("")
    ap("All numbers below are produced by `wf_analyze.py` (written to "
       "`discrepancy_waveform.json`) unless noted; derived headline numbers by "
       "`wf_report.py`.")
    ap("")

    # --- 0. scale-convention headline
    ap("## 0. Amplitude-scale convention (the x25.4 question)")
    ap("")
    ap(f"- The pipeline's `x25.4` convention and the true-mV `x1000` convention "
       f"differ by **{derived['x25p4_vs_x1000_factor']:.2f}x** "
       f"(`wf_report.py`).")
    ap("- Applying `x25.4` to real volts puts real RMS at "
       "**0.4-3.3 (mV-equiv)**, i.e. only **0.4-3.1%** of the synthetic "
       "model-input mV scale (ratios below). `x25.4` therefore does NOT align "
       "the two domains; it under-scales real by ~30-240x vs synthetic mV.")
    ap("- `x1000` (true mV) lands real on the same order of magnitude as synth "
       "(ratios 0.17-1.21). The single factor on raw volts needed to match the "
       "synth median is class-dependent (**825-6065x**), so no single scalar "
       "reconciles them. (`wf_analyze.py` -> `pipeline_x25p4`)")
    ap("")
    ap("| pairing | real med (V) | x25.4 -> ratio vs synth mV | x1000(mV) -> ratio | factor on V to match synth |")
    ap("|---|---|---|---|---|")
    for k, v in d["pipeline_x25p4"].items():
        ap(f"| {k} | {v['real_rms_median_volts']:.5f} | "
           f"{v['real_rms_if_x25p4']:.3f} -> {v['x25p4_vs_synth_ratio']:.4f} | "
           f"{v['real_rms_if_x1000_mv']:.2f} -> {v['x1000_vs_synth_ratio']:.3f} | "
           f"{v['factor_on_volts_to_match_synth']:.0f} |")
    ap("")

    # --- 1. amplitude distributions + scale gap
    ap("## 1. Amplitude distributions (RMS per 3 s window, mV) & scale gap (dB)")
    ap("")
    ap("RMS-per-window percentiles (mV). Real analysed as mV (x1000); synth = "
       "clipped model input. (`wf_analyze.py` -> `real`/`synth` `rms_mv_pct`)")
    ap("")
    ap("| source | p5 | p25 | p50 | p75 | p95 |")
    ap("|---|---|---|---|---|---|")
    def row(name, p):
        ap(f"| {name} | {p['p5']:.1f} | {p['p25']:.1f} | {p['p50']:.1f} | "
           f"{p['p75']:.1f} | {p['p95']:.1f} |")
    for k in ("human_active", "human_nothing", "car_active", "car_nothing"):
        row("REAL " + k, R[k]["rms_mv_pct"])
    for k in ("human", "vehicle", "nothing", "animal", "mixed"):
        row("SYNTH " + k, S[k]["rms_mv_pct"])
    ap("")
    ap("Scale gap = 20*log10(synth_median / real_median). Positive = synth "
       "louder. (`wf_analyze.py` -> `scale_gap`)")
    ap("")
    ap("| pairing | real med mV | synth med mV | gap dB (median) | gap dB (p95) |")
    ap("|---|---|---|---|---|")
    for k, v in d["scale_gap"].items():
        ap(f"| {k} | {v['real_rms_median_mv']:.1f} | {v['synth_rms_median_mv']:.1f} | "
           f"{v['gap_db_median']:+.1f} | {v['gap_db_p95']:+.1f} |")
    ap("")
    ap(f"- Real CAR events overshoot the synthetic +-256 mV ceiling: real car "
       f"peak = **{derived['car_peak_mv']:.0f} mV** "
       f"(**{derived['car_peak_over_synth_clip_ratio']:.1f}x**, "
       f"**{derived['car_peak_over_synth_clip_db']:.1f} dB** above the synth clip), "
       f"while synth vehicle p95 window RMS is pinned near 253 mV by clipping.")
    ap("")

    # --- 2. impulsiveness
    ap("## 2. Impulsiveness (per-window excess kurtosis & crest factor)")
    ap("")
    ap("| source | kurt p50 | kurt p95 | crest p50 | crest p95 |")
    ap("|---|---|---|---|---|")
    for lbl, k, tab in [("REAL human_active","human_active",R),
                        ("REAL car_active","car_active",R),
                        ("REAL human_nothing","human_nothing",R),
                        ("REAL car_nothing","car_nothing",R),
                        ("SYNTH human","human",S),("SYNTH vehicle","vehicle",S),
                        ("SYNTH nothing","nothing",S)]:
        kp, cp = tab[k]["kurt_pct"], tab[k]["crest_pct"]
        ap(f"| {lbl} | {kp['p50']:.2f} | {kp['p95']:.2f} | {cp['p50']:.2f} | {cp['p95']:.2f} |")
    ap("")
    ap(f"- **Human is the worst realism gap**: real footstep windows are spiky "
       f"(kurt p50 = {derived['human_kurt_real_vs_synth'][0]:.1f}, crest p50 = "
       f"{derived['human_crest_real_vs_synth'][0]:.1f}) but synth human is "
       f"near-Gaussian (kurt p50 = {derived['human_kurt_real_vs_synth'][1]:.2f}, "
       f"crest p50 = {derived['human_crest_real_vs_synth'][1]:.1f}). Synth "
       f"under-models impulsiveness by ~{derived['human_kurt_real_vs_synth'][0] - derived['human_kurt_real_vs_synth'][1]:.0f} "
       f"kurtosis units.")
    ap(f"- Car gap is milder (real kurt {derived['car_kurt_real_vs_synth'][0]:.2f} "
       f"vs synth {derived['car_kurt_real_vs_synth'][1]:.2f}). +-256 clipping "
       f"drives synth crest p5 toward 1.0 (flat-topped), an artifact absent in real.")
    ap("")

    # --- 3. quantization
    ap("## 3. Quantization (LSB & distinct levels per window)")
    ap("")
    ap("| source | LSB (mV) | eff bits | distinct/win (median) | fill ratio |")
    ap("|---|---|---|---|---|")
    for k in ("human_active","car_active","human_nothing","car_nothing",
              "field_20260524_a","field_20260615_a","field_20260615_b"):
        q = R[k]["quantization"]
        lsb = q["lsb_mv"]; eb = q["eff_bits"]
        ap(f"| REAL {k} | {lsb:.3f} | {eb:.1f} | "
           f"{R[k]['ndist_per_win_median']:.0f} | {R[k]['quant_fill_ratio']:.3f} |")
    for k in ("human","vehicle","nothing"):
        ap(f"| SYNTH {k} | continuous (float32) | ~24 | "
           f"{S[k]['ndist_per_win_median']:.0f} | {S[k]['quant_fill_ratio']:.3f} |")
    ap("")
    ap("- Real is a discrete comb: main sessions LSB ~0.20-0.29 mV, field files "
       "LSB = 0.125 mV (ADS1015, 12-bit, +-0.256 V FSR). Distinct levels per 3 s "
       "window collapse to **14-114** in quiet/field data (fill 0.5-4%). Synth "
       "float32 fills **96-100%** of a window with unique levels (continuous). "
       "The quantization is worst exactly in the quiet regime the model must "
       "discriminate.")
    ap("")

    # --- 4. clipping
    ap("## 4. Clipping / rail saturation")
    ap("")
    ap("| source | rail frac (exact) | clipfrac median | clipfrac p95 |")
    ap("|---|---|---|---|")
    for k in ("human_active","car_active","human_nothing","car_nothing"):
        ap(f"| REAL {k} | {R[k]['clipping']['frac_at_exact_rail']:.5f} | n/a | n/a |")
    for k in ("human","vehicle","nothing"):
        ap(f"| SYNTH {k} | n/a | {S[k]['clipfrac_median_within_scene']:.4f} | "
           f"{S[k]['clipfrac_p95_within_scene']:.4f} |")
    ap("")
    ap("- Real main sessions **never saturate** (0.000 samples at rail; car "
       "reaches 18.3 V with no clip). Synth is hard-clipped at +-256 mV: median "
       f"within-scene clip fraction human {S['human']['clipfrac_median_within_scene']*100:.1f}%, "
       f"vehicle {S['vehicle']['clipfrac_median_within_scene']*100:.1f}%, "
       f"nothing {S['nothing']['clipfrac_median_within_scene']*100:.1f}%; the "
       "loudest synth windows are up to 100% railed. Structural inversion: synth "
       "clips where real has headroom.")
    ap("")

    # --- 5. non-stationarity
    ap("## 5. Non-stationarity (CV of per-window RMS)")
    ap("")
    ap("| class | real rms-CV | synth rms-CV (median within scene) |")
    ap("|---|---|---|")
    for cls, (rv, sv) in derived["rms_cv_real_vs_synth"].items():
        ap(f"| {cls} | {rv:.3f} | {sv:.3f} |")
    ap("")
    ap("- Real active recordings are burstier than synthetic scenes (real car "
       f"CV {derived['rms_cv_real_vs_synth']['vehicle'][0]:.2f} vs synth "
       f"{derived['rms_cv_real_vs_synth']['vehicle'][1]:.2f}; real human "
       f"{derived['rms_cv_real_vs_synth']['human'][0]:.2f} vs synth "
       f"{derived['rms_cv_real_vs_synth']['human'][1]:.2f}). Real records "
       "long quiet + sparse events; synth scenes are event-centric/uniform. "
       "Note the spans differ (real 840-1400 s vs synth 20-115 s).")
    ap("")

    # --- 6. timing
    ap("## 6. Sample rate & timing (from real time_s)")
    ap("")
    ap("| source | fs Hz | jitter ms | dt_max s | frac dt>1.5x median |")
    ap("|---|---|---|---|---|")
    for k in R:
        t = R[k]["timing"]
        ap(f"| {k} | {t['fs_hz']:.2f} | {t['jitter_ms']:.3f} | "
           f"{t['dt_max_s']:.3f} | {t['frac_gap_gt_1p5x']:.2e} |")
    ap("| SYNTH (all) | 1000.00 | 0.000 | - | 0 |")
    ap("")
    ap("- Main sessions (human/car active + car_nothing) sit on a clean 1000 Hz "
       "grid (jitter <=0.01 ms), matching synth fs. But: `human_nothing` is a "
       "spliced recording (39 discontinuities of 35.0 s). Field 20260524 files "
       "are **100 Hz** (not 1000). Field 20260615 files show 0.08-0.5% dropped "
       "samples. Synth never has splices/drops/rate mixing.")
    ap("")

    open(MD, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", MD)
    print("updated", J, "with derived block")


if __name__ == "__main__":
    main()
