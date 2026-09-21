"""Second test sets, evaluated from stored per-window scores. Writes SECOND_TESTS.html.

Sets: the vehicle drive-bys (four cars, four surfaces, 200 Hz), the savanna wildlife set (people,
animals, quiet, distance and station on every window, 200 Hz), the rail passes near Lyon (all
"nothing", a heavy-vehicle confuser, 250 Hz decimated to 200), and the field recordings decimated
to 200 Hz. Two model arms: the 1 kHz campaign models (band mismatch on the 200 Hz sets, shown as
the "mismatched band" reference) and the matched-band arm trained on synthetic features whose
input was decimated to 200 Hz (group Z, and MZ for the classical models).

Thresholds always come from each model's own synthetic background at 0.1% false alarms.

    python campaign/second_tests.py
"""
import glob
import html
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from training import db

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))
DATASETS_ROOT = os.environ.get("GEO_REAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "geophone_datasets"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"snn": "#2a78d6", "ann": "#7b858c", "cls": "#4a3aa7", "ink": "#14171a", "ink2": "#4a5359", "ink3": "#7b858c"}
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6eae8", "axes.axisbelow": True, "legend.frameon": False, "svg.fonttype": "none"})

# model families: label -> (run ids, kind)
REF = {  # 1 kHz campaign models
    "SNN-S": ([f"U2_lowrank16_rep{i}" for i in (1, 2, 3)], "snn"), "SNN-M": ([f"S5_256x128_rep{i}" for i in (1, 2, 3)], "snn"),
    "SNN-L": ([f"S6_512x256x128_rep{i}" for i in (1, 2, 3)], "snn"),
    "ANN-S": ([f"X1_ann_64x32_lr16_rep{i}" for i in (1, 2)], "ann"), "ANN-M": ([f"X2_ann_256x128_rep{i}" for i in (1, 2)], "ann"),
    "ANN-L": ([f"X3_ann_512x256x128_rep{i}" for i in (1, 2)], "ann"),
    "logistic regression": (["M1_logreg"], "cls"), "linear SVM": (["M2_svm_linear"], "cls"), "kernel SVM": (["M3_svm_rbf"], "cls"),
    "random forest": (["M4_forest"], "cls"), "boosted trees": (["M5_boosted"], "cls"), "small ordinary net": (["M7_mlp"], "cls"),
}
ARM = {  # matched-band arm
    "SNN-S": ([f"Z1_snn_S_rep{i}" for i in (1, 2, 3)], "snn"), "SNN-M": ([f"Z2_snn_M_rep{i}" for i in (1, 2, 3)], "snn"), "SNN-L": ([f"Z3_snn_L_rep{i}" for i in (1, 2, 3)], "snn"),
    "ANN-S": ([f"Z4_ann_S_rep{i}" for i in (1, 2)], "ann"), "ANN-M": ([f"Z5_ann_M_rep{i}" for i in (1, 2)], "ann"), "ANN-L": ([f"Z6_ann_L_rep{i}" for i in (1, 2)], "ann"),
    "logistic regression": (["MZ1_logreg"], "cls"), "linear SVM": (["MZ2_svm_linear"], "cls"), "kernel SVM": (["MZ3_svm_rbf"], "cls"),
    "random forest": (["MZ4_forest"], "cls"), "boosted trees": (["MZ5_boosted"], "cls"), "small ordinary net": (["MZ7_mlp"], "cls"),
}


def auc(y, s):
    """Area under the receiver operating curve by rank statistic; NaN if one class is missing."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_scores(con, run, ds):
    S, heads = db.get_scores(con, run, ds)
    if S is None:
        return None
    tau = {h: db.threshold_at_far(con, run, h, 0.001) for h in heads}
    if tau.get("human") is None or tau.get("vehicle") is None:
        return None
    return {h: S[:, i] for i, h in enumerate(heads)}, tau


def window_extra(con, ds):
    rows = con.execute("SELECT extra_json FROM windows WHERE dataset=? ORDER BY window_id", (ds,)).fetchall()
    return [json.loads(r[0]) if r[0] else {} for r in rows]


# ---------------------------------------------------------------- per-dataset evaluation

def eval_vehicles(con, run, W, dist):
    got = load_scores(con, run, "vehicles")
    if got is None:
        return None
    sc, tau = got
    y = W["label"]; veh = y == "vehicle"; quiet = y == "nothing"
    von = sc["vehicle"] > tau["vehicle"]; hon = sc["human"] > tau["human"]
    out = dict(auc=auc(veh, sc["vehicle"]), recall=float(von[veh].mean()), fa_any=float((von | hon)[quiet].mean()),
               called_person=float((hon & ~von)[veh].mean()))
    if dist is not None:
        # a quarter of the far-end windows still have the car inside 50 m, where it is plainly
        # detectable, so the honest negative is the part of the far end beyond 100 m
        far = quiet & np.isfinite(dist) & (dist > 100.0)
        near = veh & np.isfinite(dist) & (dist < 20.0)
        if far.sum() >= 50:
            out["fa_beyond_100m"] = float((von | hon)[far].mean())
            if near.sum() >= 50:
                m = near | far
                out["auc_clean"] = auc(near[m], sc["vehicle"][m])
                out["recall_within_20m"] = float(von[near].mean())
    for terr in np.unique(W["terrain"]):
        m = veh & (W["terrain"] == terr)
        if m.sum():
            out[f"recall@{terr}"] = float(von[m].mean())
    if dist is not None:
        # the car is somewhere on the site in every window of this set, near or far, so the curve
        # runs over all windows with a distance rather than over the near-labeled ones only
        d = dist
        for a, b in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 120), (120, 999)):
            m = np.isfinite(d) & (d >= a) & (d < b)
            if m.sum() >= 30:
                out[f"recall@dist:{a}-{b}"] = float(von[m].mean())
    return out


def eval_savanna(con, run, W):
    got = load_scores(con, run, "savanna")
    if got is None:
        return None
    sc, tau = got
    y = W["label"]; hum = y == "human"; ani = y == "animal"; quiet = y == "nothing"
    hon = sc["human"] > tau["human"]; von = sc["vehicle"] > tau["vehicle"]
    out = dict(auc_person=auc(hum[hum | quiet], sc["human"][hum | quiet]), recall_person=float(hon[hum].mean()),
               fa_any=float((hon | von)[quiet].mean()), animal_called_person=float(hon[ani].mean()))
    d0 = W["distance_m"]
    near = hum & np.isfinite(d0) & (d0 < 25.0)          # inside the range the physics allows
    if near.sum() >= 200 and quiet.sum() >= 200:
        m = near | quiet
        out["auc_person_within_25m"] = auc(near[m], sc["human"][m])
        out["recall_person_within_25m"] = float(hon[near].mean())
    if "animal" in sc and tau.get("animal") is not None and np.isfinite(sc["animal"]).all():
        aon = sc["animal"] > tau["animal"]
        out["auc_animal"] = auc(ani[ani | quiet], sc["animal"][ani | quiet]); out["recall_animal"] = float(aon[ani].mean())
    d = W["distance_m"]
    for a, b in ((0, 25), (25, 50), (50, 100), (100, 150)):
        m = hum & (d >= a) & (d < b)
        if m.sum() >= 50:
            out[f"recall_person@dist:{a}-{b}"] = float(hon[m].mean())
    # spread over stations: person separation per station
    st = W["station"]
    per = []
    for s in np.unique(st):
        m = (st == s) & (hum | quiet)
        if (hum & m).sum() >= 50 and (quiet & m).sum() >= 50:
            per.append(auc(hum[m], sc["human"][m]))
    if per:
        out["auc_person_station_min"] = float(np.min(per)); out["auc_person_station_median"] = float(np.median(per))
    return out


def rail_in_pass(W, span=180.0, pass_s=120.0):
    """True where a window's centre falls inside the catalogued pass, from the window's own start
    time. The flag stored at preparation time is wrong and is ignored."""
    lead = (span - pass_s) / 2.0
    c = W["t0_s"] + 1.5
    return (c >= lead) & (c <= lead + pass_s)


def eval_rail(con, run, W, inpass):
    got = load_scores(con, run, "rail")
    if got is None:
        return None
    sc, tau = got
    hon = sc["human"] > tau["human"]; von = sc["vehicle"] > tau["vehicle"]
    hours = len(hon) * 1.5 / 3600.0
    out = dict(person_fa_share=float(hon.mean()), vehicle_share=float(von.mean()), nothing_share=float((~hon & ~von).mean()),
               person_fa_per_hour=float(hon.sum() / hours), hours=hours)
    for nm, m in (("in_pass", inpass), ("lead_tail", ~inpass)):
        out[f"person_fa_share@{nm}"] = float(hon[m].mean()); out[f"vehicle_share@{nm}"] = float(von[m].mean())
    return out


def eval_field(con, run, ds, W):
    got = load_scores(con, run, ds)
    if got is None:
        return None
    sc, tau = got
    hon = sc["human"] > tau["human"]; von = sc["vehicle"] > tau["vehicle"]
    d = np.where(hon & von, np.where(sc["human"] >= sc["vehicle"], "human", "vehicle"), np.where(hon, "human", np.where(von, "vehicle", "nothing")))
    return dict(accuracy=float((d == W["label"]).mean()))


def vehicle_distances(W):
    """Distance at each vehicle window's centre, from the source distance files."""
    try:
        import polars as pl
    except ImportError:
        return None
    base = os.path.join(DATASETS_ROOT, "m3n_vc")
    cache = {}
    out = np.full(len(W["label"]), np.nan)
    for i, (rec, t0) in enumerate(zip(W["recording"], W["t0_s"])):
        scene, stem, _tag = rec.split("/")
        if (scene, stem) not in cache:
            try:
                g = pl.read_parquet(os.path.join(base, scene, stem + "_geo.parquet"), columns=["timestamp"])
                d = pl.read_parquet(os.path.join(base, scene, stem + "_dis.parquet"))
                ts = g["timestamp"].to_numpy().astype(np.float64)
                if np.nanmedian(ts) > 1e11:
                    ts = ts / 1000.0
                dt = d["time"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
                dd = d["distance"].to_numpy().astype(np.float64)
                ok = np.isfinite(dd)
                cache[(scene, stem)] = (dt[ok] - ts[0], dd[ok])
            except Exception:
                cache[(scene, stem)] = None
        c = cache[(scene, stem)]
        if c is not None and len(c[0]) > 1:
            out[i] = np.interp(t0 + 1.5, c[0], c[1])
    return out if np.isfinite(out).any() else None


# ---------------------------------------------------------------- family aggregation

def family_eval(con, fams, fn):
    rows = {}
    for label, (runs, kind) in fams.items():
        vals = [v for v in (fn(r) for r in runs) if v is not None]
        if not vals:
            continue
        keys = sorted({k for v in vals for k in v})
        rows[label] = dict(kind=kind, n=len(vals), **{k: float(np.nanmean([v[k] for v in vals if k in v])) for k in keys})
    return rows


def bar_chart(rows, key, ylabel, title, ylim=None, fmt="{:.3f}"):
    labels = [l for l in rows if key in rows[l] and not np.isnan(rows[l][key])]
    if not labels:
        return ""
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels) + 2), 3.6))
    vals = [rows[l][key] for l in labels]; cols = [C[rows[l]["kind"]] for l in labels]
    ax.bar(np.arange(len(labels)), vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.004 if (ylim is None or ylim[1] <= 1.05) else 0.02 * ylim[1]), fmt.format(v), ha="center", fontsize=8, color=C["ink2"], rotation=90 if len(labels) > 10 else 0)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10, loc="left", color=C["ink"]); ax.grid(axis="x", visible=False)
    if ylim: ax.set_ylim(*ylim)
    buf = io.StringIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    s = buf.getvalue(); return s[s.find("<svg"):]


def dist_chart(rows, prefix, title, ylabel="detection rate at the synthetic threshold"):
    keys = sorted({k for r in rows.values() for k in r if k.startswith(prefix)}, key=lambda k: float(k.split("@dist:")[1].split("-")[0]))
    labels = [l for l in rows if any(k in rows[l] for k in keys)]
    if not keys or not labels:
        return ""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for l in labels:
        ys = [rows[l].get(k, np.nan) for k in keys]
        ax.plot(range(len(keys)), ys, "-o", color=C[rows[l]["kind"]], alpha=0.85, ms=4, lw=1.3, label=l)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.split("@dist:")[1].replace("-999", "+").replace("-", " to ") + " m" for k in keys], fontsize=8.5)
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=10, loc="left", color=C["ink"]); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    buf = io.StringIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    s = buf.getvalue(); return s[s.find("<svg"):]


def table(rows, cols, names):
    h = "<table><thead><tr><th>model</th><th>runs</th>" + "".join(f"<th>{html.escape(n)}</th>" for n in names) + "</tr></thead><tbody>"
    for l, r in rows.items():
        h += f"<tr><td>{html.escape(l)}</td><td class=num>{r['n']}</td>" + "".join(
            f"<td class=num>{r[c]:.3f}</td>" if c in r and not np.isnan(r[c]) else "<td class=num>–</td>" for c in cols) + "</tr>"
    return h + "</tbody></table>"


def main():
    con = db.connect(os.path.join(OUT, "campaign.sqlite"))
    sections = []
    Wv = db.dataset_windows(con, "vehicles"); Ws = db.dataset_windows(con, "savanna")
    Wr = db.dataset_windows(con, "rail"); Wf = db.dataset_windows(con, "elbit_bl200")
    dist = vehicle_distances(Wv) if Wv is not None else None
    inpass = None
    if Wr is not None:
        inpass = rail_in_pass(Wr)

    # ---- vehicles
    for arm_name, fams in (("matched-band arm (trained on 200 Hz band)", ARM), ("1 kHz models, band mismatch (reference)", REF)):
        rows = family_eval(con, fams, lambda r: eval_vehicles(con, r, Wv, dist)) if Wv is not None else {}
        if not rows:
            sections.append(f"<h2>Vehicle drive-bys · {html.escape(arm_name)}</h2><p>No scored runs yet.</p>"); continue
        terr = sorted({k for r in rows.values() for k in r if k.startswith("recall@") and not k.startswith("recall@dist:")})
        sections.append(f"<h2>Vehicle drive-bys · {html.escape(arm_name)}</h2>"
                        f"<p>{len(Wv['label']):,} windows: near-pass windows labeled vehicle, far-end windows of the same recording labeled quiet. Four cars, six scenes, four road surfaces. Thresholds at 0.1% false alarms on each model's synthetic background.</p>"
                        "<p><b>The dataset's own negative is contaminated.</b> Reading the satellite track back, the near windows sit at 1.4 to 53 m (median 12) and the far windows at 6.8 to 192 m (median 86), but a quarter of the far windows still have the car inside 50 m, where it is plainly detectable. The first three columns of the table use a clean split instead: vehicle inside 20 m against quiet beyond 100 m. The remaining columns use the dataset's own split and are the pessimistic reading.</p>"
                        + bar_chart(rows, "auc", "separation, vehicle vs quiet (ROC area)", "How well the vehicle score separates passes from the far end", (0.5, 1.0))
                        + bar_chart(rows, "recall", "share of near-pass windows detected", "Detection rate at the synthetic threshold", (0, 1.0))
                        + bar_chart(rows, "fa_any", "share of quiet windows with any head on", "False alarms on the far-end windows", (0, 1.0))
                        + dist_chart(rows, "recall@dist:", "Vehicle head on, against the car's distance from the satellite track. Every window of this set has the car somewhere on the site, so this is a detection-against-range curve, not a false-alarm curve.")
                        + table(rows, ["auc_clean", "recall_within_20m", "fa_beyond_100m", "auc", "recall", "fa_any", "called_person"] + terr,
                                ["separation, clean split", "detection within 20 m", "false alarm beyond 100 m", "separation, all", "detection, all", "false alarm, all far end", "vehicle called person"] + [t.split("@")[1].replace("_", " ") for t in terr]))
    # ---- savanna
    for arm_name, fams in (("matched-band arm", ARM), ("1 kHz models, band mismatch (reference)", REF)):
        rows = family_eval(con, fams, lambda r: eval_savanna(con, r, Ws)) if Ws is not None else {}
        if not rows:
            sections.append(f"<h2>Savanna wildlife set · {html.escape(arm_name)}</h2><p>No scored runs yet.</p>"); continue
        sections.append(f"<h2>Savanna wildlife set · {html.escape(arm_name)}</h2>"
                        f"<p>{len(Ws['label']):,} windows from about 20 stations: people, animals, and a third class labeled noise, with the camera-trap distance on every window.</p>"
                        "<p><b>This set cannot measure detection, and the distances say why.</b> The labels mark what a camera saw, not what the sensor could feel. The person windows sit at a median of 92 m with only a tenth inside 36 m, while the physics of Part 3 puts a walking person out of reach past roughly 50 m, so most of the positives contain no recoverable signal. The negative is worse: the noise class carries a camera distance of its own, median 57 m, the same distribution as the animals, so it marks windows where something was present and unclassified rather than windows that are quiet. With both ends broken, separation below 0.5 is what a working detector produces here. The close-range columns restrict the positives to inside 25 m; the negative stays contaminated, so even those are a floor rather than a measurement.</p>"
                        + bar_chart(rows, "auc_person", "separation, person vs quiet (ROC area)", "Person separation, all distances pooled", (0.4, 1.0))
                        + dist_chart(rows, "recall_person@dist:", "Person detection rate against camera-trap distance")
                        + bar_chart(rows, "fa_any", "share of quiet windows with any head on", "False alarms on the labeled quiet windows", (0, 1.0))
                        + bar_chart(rows, "animal_called_person", "share of animal windows called person", "Animals mistaken for people", (0, 1.0))
                        + table(rows, ["auc_person_within_25m", "recall_person_within_25m", "auc_person", "recall_person", "fa_any", "animal_called_person", "auc_person_station_min", "auc_person_station_median"],
                                ["person separation within 25 m", "person detection within 25 m", "person separation, all", "person detection, all", "share of noise-class windows with a head on", "animal called person", "worst station", "median station"]))
    # ---- rail
    if Wr is not None:
        rows = family_eval(con, ARM, lambda r: eval_rail(con, r, Wr, inpass))
        if rows:
            hrs = next(iter(rows.values()))["hours"]
            sections.append(f"<h2>Rail passes near Lyon · matched-band arm</h2>"
                            f"<p>{len(Wr['label']):,} windows, {hrs:.0f} hours of ground beside a railway, 2,574 catalogued train passes as 180-second cuts with the catalogued 120 seconds in the middle. Nothing here is a person, so every person detection is a false alarm; the vehicle head firing during a pass is a heavy vehicle read as a vehicle, which is the behavior the detector is built for.</p>"
                            + bar_chart(rows, "person_fa_per_hour", "person false alarms per hour", "Person false alarms per hour of ground", None, "{:.1f}")
                            + bar_chart(rows, "vehicle_share", "share of windows with the vehicle head on", "Trains read as vehicles", (0, 1.0))
                            + table(rows, ["person_fa_per_hour", "person_fa_share@in_pass", "person_fa_share@lead_tail", "vehicle_share@in_pass", "vehicle_share@lead_tail", "nothing_share"],
                                    ["person false alarms per hour", "person share, during pass", "person share, lead and tail", "vehicle share, during pass", "vehicle share, lead and tail", "called nothing"]))
        else:
            sections.append("<h2>Rail passes near Lyon</h2><p>No scored runs yet.</p>")
    # ---- field, band-limited
    if Wf is not None:
        rows = family_eval(con, ARM, lambda r: eval_field(con, r, "elbit_bl200", Wf))
        if rows:
            sections.append("<h2>Field recordings, band-limited to 100 Hz · matched-band arm</h2>"
                            "<p>The four field recordings decimated to 200 Hz and back, scored by the matched-band models. Three-way accuracy, one threshold per window, all 2,353 windows. Compare with the 1 kHz models on the full band in the main report.</p>"
                            + bar_chart(rows, "accuracy", "three-way accuracy", "Field accuracy with the band above 100 Hz removed", (0.5, 1.0))
                            + table(rows, ["accuracy"], ["three-way accuracy"]))
    css = ("<title>Second Test Sets</title><style>body{font-family:'Source Sans 3','Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:28px 24px;color:#14171a;background:#fbfbfa}"
           "h1{font-family:Archivo,'Arial Narrow',sans-serif;font-size:32px;margin:0 0 6px}h2{font-size:20px;margin:36px 0 8px;padding-top:12px;border-top:1px solid #cfd5d2}"
           "p{max-width:80ch;color:#48515a}table{border-collapse:collapse;font-size:13px;margin:10px 0 18px}th,td{padding:5px 10px;border-bottom:1px solid #e3e7e4;text-align:left}"
           "th{font-family:'IBM Plex Mono',monospace;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#7b858c}td.num{font-variant-numeric:tabular-nums;text-align:right}"
           "svg{max-width:100%;height:auto;display:block;margin:8px 0}.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#7b858c}</style>")
    intro = ("<div class=eyebrow>Geophone detector · second test sets</div><h1>Second Test Sets</h1>"
             "<p>Results on the public datasets beyond the four field recordings, from stored per-window scores. Model names: SNN spiking, ANN its non-spiking twin; S, M, L the three sizes (about 4.7k, 54k, 206k parameters). "
             "The matched-band arm is trained on synthetic features whose input was decimated to 200 Hz and brought back to 1 kHz, the same path the 200 Hz real datasets take, so synthetic and real windows share the empty band above 100 Hz. The 1 kHz models are shown on the same sets as a band-mismatch reference.</p>")
    open(os.path.join(OUT, "SECOND_TESTS.html"), "w", encoding="utf-8").write(css + intro + "".join(sections)
                                                                                + "<p class=eyebrow>evaluation/second_tests.py</p>")
    print("wrote", os.path.join(OUT, "SECOND_TESTS.html"))


if __name__ == "__main__":
    main()
