# wf_analyze.py
# Main driver: computes every time-domain / amplitude discrepancy number and
# writes gap_study/v4_plan/discrepancy/discrepancy_waveform.json
#
# Run: python wf_analyze.py
import os, json, sqlite3, time
import numpy as np
import wf_common as C

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "discrepancy_waveform.json")

# ------------------------------------------------------------- real file table
GD = os.path.join(C.BASE, "Goephone-Project", "geophone_data")
REAL_FILES = {
    # session -> (class, path)
    "human_active":  ("human",   os.path.join(GD, "human.csv")),
    "human_nothing": ("nothing", os.path.join(GD, "human_nothing.csv")),
    "car_active":    ("vehicle", os.path.join(GD, "car.csv")),
    "car_nothing":   ("nothing", os.path.join(GD, "car_nothing.csv")),
    "field_20260524_a": ("field", os.path.join(C.BASE, "geophone_20260524_180254.csv")),
    "field_20260524_b": ("field", os.path.join(C.BASE, "geophone_20260524_180403.csv")),
    "field_20260615_a": ("field", os.path.join(C.BASE, "geophone_20260615_065644.csv")),
    "field_20260615_b": ("field", os.path.join(C.BASE, "geophone_20260615_073558.csv")),
    "field_20260615_c": ("field", os.path.join(C.BASE, "geophone_20260615_082734.csv")),
}

SYNTH_CLASSES = {"human": 100, "vehicle": 100, "nothing": 100,
                 "animal": 60, "mixed": 60}


def analyze_real_file(session, cls, path):
    t, a_volts = C.load_real_volts(path)
    timing = C.measure_fs(t)
    fs = timing["fs_hz"]
    a_mv = a_volts * 1000.0            # primary unit
    # windowed metrics in mV
    wm = C.window_metrics(a_mv, fs)
    quant = C.estimate_lsb(a_volts)    # LSB in native volts
    rail = C.real_rail_fraction(a_volts)
    rec = {
        "session": session, "class": cls, "path": path,
        "timing": timing,
        "dc_mean_mv": float(np.mean(a_mv)),
        "global_std_mv": float(np.std(a_mv)),
        "global_std_volts": float(np.std(a_volts)),
        "quantization": {
            "lsb_volts": quant["lsb"],
            "lsb_mv": None if quant["lsb"] is None else quant["lsb"] * 1000.0,
            "n_distinct_total": quant["n_distinct"],
            "eff_bits": quant["eff_bits"],
            "range_volts": quant["range"],
        },
        "clipping": rail,  # rail_abs is in volts
    }
    if wm is not None:
        rec["rms_mv_pct"] = C.pct(wm["rms"])
        rec["kurt_pct"] = C.pct(wm["kurt"])
        rec["crest_pct"] = C.pct(wm["crest"])
        rec["ndist_per_win_pct"] = C.pct(wm["ndist"])
        rec["ndist_per_win_median"] = float(np.median(wm["ndist"]))
        rec["quant_fill_ratio"] = float(np.median(wm["ndist"]) / wm["win_len"])
        rec["rms_cv"] = C.cv_of_rms(wm["rms"])
        rec["n_windows"] = wm["n_win"]
        rec["_rms_median_mv"] = rec["rms_mv_pct"]["p50"]
    return rec


def analyze_synth_class(cur, coarse, n):
    ids = C.synth_scene_ids(cur, coarse, n, seed=0)
    rms_all, kurt_all, crest_all, ndist_all = [], [], [], []
    cv_list = []
    clipfrac_scene = []
    n_scene = 0
    for sid in ids:
        r = C.synth_model_input(cur, sid)
        if r is None:
            continue
        clipped, unclipped = r
        wm = C.window_metrics(clipped, 1000.0)
        if wm is None:
            continue
        rms_all.append(wm["rms"])
        kurt_all.append(wm["kurt"])
        crest_all.append(wm["crest"])
        ndist_all.append(wm["ndist"])
        cv_list.append(C.cv_of_rms(wm["rms"]))
        clipfrac_scene.append(C.synth_clip_fraction(unclipped))
        n_scene += 1
    rms_all = np.concatenate(rms_all)
    kurt_all = np.concatenate(kurt_all)
    crest_all = np.concatenate(crest_all)
    ndist_all = np.concatenate(ndist_all)
    return {
        "coarse": coarse, "n_scenes": n_scene, "n_windows": int(len(rms_all)),
        "rms_mv_pct": C.pct(rms_all),
        "kurt_pct": C.pct(kurt_all),
        "crest_pct": C.pct(crest_all),
        "ndist_per_win_pct": C.pct(ndist_all),
        "ndist_per_win_median": float(np.median(ndist_all)),
        "quant_fill_ratio": float(np.median(ndist_all) / 3000.0),
        "rms_cv_median_within_scene": float(np.median(cv_list)),
        "clipfrac_median_within_scene": float(np.median(clipfrac_scene)),
        "clipfrac_p95_within_scene": float(np.percentile(clipfrac_scene, 95)),
        "_rms_median_mv": float(np.percentile(rms_all, 50)),
    }


def db_gap(real_med, synth_med):
    if not real_med or not synth_med or real_med <= 0 or synth_med <= 0:
        return None
    return 20.0 * np.log10(synth_med / real_med)


def main():
    t0 = time.time()
    out = {"meta": {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synth_db": C.SYNTH_DB, "clip_mv": C.CLIP_MV, "win_s": C.WIN_S,
        "notes": "REAL amp is VOLTS; mV=volts*1000; pipeline convention=volts*25.4. "
                 "SYNTH model input = clip(clean+clean2?+noise, +-256 mV).",
    }, "real": {}, "synth": {}, "scale_gap": {}, "pipeline_x25p4": {}}

    for sess, (cls, path) in REAL_FILES.items():
        print("real:", sess, flush=True)
        out["real"][sess] = analyze_real_file(sess, cls, path)

    con = sqlite3.connect(C.SYNTH_DB)
    cur = con.cursor()
    for coarse, n in SYNTH_CLASSES.items():
        print("synth:", coarse, flush=True)
        out["synth"][coarse] = analyze_synth_class(cur, coarse, n)
    con.close()

    # ---- scale gap (dB) for the primary class pairings (real mV vs synth mV)
    pairings = [
        ("human_active", "human"),
        ("car_active", "vehicle"),
        ("human_nothing", "nothing"),
        ("car_nothing", "nothing"),
    ]
    for sess, coarse in pairings:
        rmed = out["real"][sess].get("_rms_median_mv")
        smed = out["synth"][coarse]["_rms_median_mv"]
        # also compare high-energy tail (p95) which reflects events
        rp95 = out["real"][sess].get("rms_mv_pct", {}).get("p95")
        sp95 = out["synth"][coarse]["rms_mv_pct"]["p95"]
        out["scale_gap"][f"{sess}__vs__{coarse}"] = {
            "real_rms_median_mv": rmed, "synth_rms_median_mv": smed,
            "gap_db_median": db_gap(rmed, smed),
            "real_rms_p95_mv": rp95, "synth_rms_p95_mv": sp95,
            "gap_db_p95": db_gap(rp95, sp95),
        }

    # ---- does x25.4 align the medians? evaluate on the two ambient/baseline
    #      files (cleanest, no sparse-event dilution) and the active files.
    for sess, coarse in pairings:
        std_v = out["real"][sess]["global_std_volts"]
        rmed_mv = out["real"][sess].get("_rms_median_mv")  # = volts*1000 median rms
        rmed_volts = rmed_mv / 1000.0 if rmed_mv else None
        smed = out["synth"][coarse]["_rms_median_mv"]
        # median rms if pipeline scale (volts*25.4) were used:
        rms_x254 = rmed_volts * 25.4 if rmed_volts else None
        rms_x1000 = rmed_mv
        factor_to_align = (smed / rmed_volts) if (rmed_volts and rmed_volts > 0) else None
        out["pipeline_x25p4"][f"{sess}__vs__{coarse}"] = {
            "real_rms_median_volts": rmed_volts,
            "synth_rms_median_mv": smed,
            "real_rms_if_x25p4": rms_x254,
            "real_rms_if_x1000_mv": rms_x1000,
            "x25p4_vs_synth_ratio": (rms_x254 / smed) if (rms_x254 and smed) else None,
            "x1000_vs_synth_ratio": (rms_x1000 / smed) if (rms_x1000 and smed) else None,
            "factor_on_volts_to_match_synth": factor_to_align,
        }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT_JSON, "in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
