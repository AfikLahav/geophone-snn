"""Re-analysis of the campaign from stored scores. Nothing is retrained and no model is re-run.

Why this exists: the campaign summary ranked models by one number -- accuracy on the field
recordings at a threshold set on synthetic background -- and the review found that number
mostly measures how that threshold transferred to real ground, plus a scoring artifact: a
three-head model may answer "animal" on a dataset that contains no animals, and it does so on
about one window in seven. This script rebuilds every comparison from the per-window scores in
the database, so the corrected picture, the noise floor, and any later chart come from one place.

Tables written. All idempotent -- re-run any time, rows are replaced, nothing accumulates.

  review_metrics    per run x dataset x variant x operating point x class
                      variant   active  the animal head may decide (how the campaign scored)
                                masked  the animal head is ignored at decision time
                      op_kind   synth_far  threshold from synthetic background quantiles (zero-shot)
                                real_far   threshold from the dataset's own quiet windows
                                           (diagnostic: uses real labels, matched across models)
                                auc        threshold-free, op_value 0
                      cls       overall | human | vehicle | animal | nothing | present
                                'present' collapses the decision to something-versus-nothing
  review_bootstrap  confidence intervals on the field recordings, 30 s blocks inside a recording
  review_cleaning   how the label cleaning overlaps with each model's errors
  review_run_flags  runs that carry a caveat: identical to the baseline, discrete output, rule
  noise_floor       spread across identical configurations, per metric and operating point
  review_meta       campaign-level statistics and when this last ran

One rule is registered like a model: L0_loudness, a single loudness threshold set on synthetic
background, deciding "present -> human". It is the floor every model has to clear.

    python -m campaign.review            # rebuild everything and print the corrected table
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simgeo", "simgeo_v42"))

import features as F                                           # noqa: E402
from training import data as D, db
from evaluation import prepare_real

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))
DB_PATH = os.path.join(OUT, "campaign.sqlite")

HEADS = ["human", "animal", "vehicle"]          # column order of every score block
SYNTH_FAR = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
REAL_FAR = [0.01, 0.02, 0.05, 0.1]
VARIANTS = ["active", "masked"]
N_BOOT = 1000
BLOCK_S = 30.0
BOOT_OPS = [("synth_far", 0.001), ("synth_far", 0.01), ("real_far", 0.05)]
CLEAN_OPS = [("synth_far", 0.001), ("synth_far", 0.01)]

# Identical configurations. The first seven of the baseline family were accidental: the review
# found that their differing configuration fields never reached the model.
FAMILIES = {
    "baseline": ["A1_baseline", "B1_3head", "B8_corn", "B10_veh_multi", "B11_multi_report",
                 "G12_upto_v2new30", "E21_thr_learn"] + [f"R_A1_rep{i}" for i in range(1, 6)],
    "two_head": ["B3_2head_noanimal"] + [f"R_B3_rep{i}" for i in range(1, 6)],
    "cepstral_cut": ["G8_upto_cep16"] + [f"R_G8_rep{i}" for i in range(1, 6)],
    "ordinary": ["A2_ann_1pass"] + [f"R_A2_rep{i}" for i in range(1, 6)],
    **{f"ladder_{nm}": [f"S{k}_{nm}_rep{i}" for i in range(1, 4)]
       for k, nm in enumerate(["16x8", "32x16", "64x32", "128x64", "256x128", "512x256x128"], start=1)},
    **{f"tune_{nm}": [f"{nm}_rep{i}" for i in range(1, 4)]
       for nm in ["T1_sep77", "T2_sep_veh132", "T3_top40", "T4_top20", "T5_T8", "T6_T1", "T7_lif",
                  "T8_40k", "T9_128x128x64", "T10_128x128x128x64"]},
    **{f"under_{nm}": [f"{nm}_rep{i}" for i in range(1, 4)]
       for nm in ["U1_no_legacy32", "U1_no_phys7", "U1_no_cad12", "U1_no_eng8", "U1_no_wx2",
                  "U1_no_cep16", "U2_lowrank16", "U3_lowrank8", "U4_nodrop_64x32", "U5_lr2e3_64x32",
                  "U6_nodrop_lr2e3_64x32", "U7_nodrop_16x8", "U8_96x16", "U9_48x48"]},
    **{f"prune_{lvl}_{v}": [f"P{lvl}_{v}_rep{i}" for i in range(1, 4)]
       for lvl in ["0500", "0800", "0900", "0950", "0975"] for v in ["raw", "ft"]},
    **{f"heads_{nm}": [f"{nm}_rep{i}" for i in range(1, 4)]
       for nm in ["V1_nomult", "V2_mult_both", "V3_animal_nomult", "V4_animal_mult", "V5_animal_neg"]},
    "feats_W1_all132": [f"W1_all132_rep{i}" for i in range(1, 4)],
    "feats_W2_sel104": [f"W2_sel104_rep{i}" for i in range(1, 4)],
    # conventional models: deterministic, so two replicates (the third was skipped as a copy)
    **{f"conv_{nm}": [f"{nm}_rep{i}" for i in range(1, 3)]
       for nm in ["X1_ann_64x32_lr16", "X2_ann_256x128", "X3_ann_512x256x128", "X4_gru16",
                  "X5_gru100", "X6_gru225", "X7_attn16", "X8_attn64", "X9_attn128"]},
    **{f"small_{nm}": [f"{nm}_rep{i}" for i in range(1, 4)]
       for nm in ["Y1_T1", "Y2_lif", "Y3_nodrop"]},
}

FLAGS = [
    ("B1_3head", "identical_to_baseline", "no field differs from the baseline"),
    ("B8_corn", "identical_to_baseline",
     "both branches of the ordinal loss are the same expression; the setting never took effect"),
    ("B10_veh_multi", "identical_to_baseline", "its configuration key is read nowhere"),
    ("B11_multi_report", "identical_to_baseline", "its configuration key is read nowhere"),
    ("G12_upto_v2new30", "identical_to_baseline", "cumulative up to the last group is all 132 features"),
    ("E21_thr_learn", "identical_to_baseline",
     "the threshold is converted to a plain number before the gradient sees it; it stayed at 1.0 "
     "in every layer"),
    ("B8_corn", "no_op_config", "ordinal_loss=corn is not implemented"),
    ("B10_veh_multi", "no_op_config", "vehicle_ordinal is not implemented"),
    ("B11_multi_report", "no_op_config", "report_multiplicity is not implemented"),
    ("E21_thr_learn", "no_op_config", "learnable threshold receives no gradient"),
    ("E13_readout_count", "discrete_score",
     "the output takes five values at four steps; the 0.1% and 1% synthetic thresholds coincide"),
    ("A7_ann_4pass", "inference_identical",
     "at inference the four passes are identical (dropout off); its energy is that of one pass"),
    ("A8_stem_once", "inference_identical",
     "mathematically identical to the baseline; only the energy accounting differs"),
    ("L0_loudness", "rule_baseline", "one loudness threshold, present -> human"),
    ("M6_knn", "discrete_score",
     "one-neighbour output is 0 or 1; no synthetic threshold exists below its maximum, so it "
     "answers nothing everywhere. Kept as the leak detector it was meant to be"),
]
for _i in range(1, 6):
    FLAGS.append((f"R_A1_rep{_i}", "identical_to_baseline", "deliberate replicate"))
    FLAGS.append((f"R_B3_rep{_i}", "identical_to_two_head", "deliberate replicate"))
    FLAGS.append((f"R_G8_rep{_i}", "identical_to_cepstral_cut", "deliberate replicate"))
    FLAGS.append((f"R_A2_rep{_i}", "identical_to_ordinary", "deliberate replicate"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_metrics (
    run_id TEXT NOT NULL, dataset TEXT NOT NULL, variant TEXT NOT NULL,
    op_kind TEXT NOT NULL, op_value REAL NOT NULL, cls TEXT NOT NULL,
    n INTEGER, accuracy REAL, recall REAL, precision_ REAL, f1 REAL, auc REAL, extra_json TEXT,
    PRIMARY KEY (run_id, dataset, variant, op_kind, op_value, cls));
CREATE INDEX IF NOT EXISTS ix_rm_ds ON review_metrics(dataset, variant, op_kind, op_value, cls);
CREATE TABLE IF NOT EXISTS review_bootstrap (
    run_id TEXT NOT NULL, dataset TEXT NOT NULL, variant TEXT NOT NULL,
    op_kind TEXT NOT NULL, op_value REAL NOT NULL, metric TEXT NOT NULL,
    point REAL, lo REAL, hi REAL, n_boot INTEGER, block_s REAL,
    PRIMARY KEY (run_id, dataset, variant, op_kind, op_value, metric));
CREATE TABLE IF NOT EXISTS review_cleaning (
    run_id TEXT NOT NULL, dataset TEXT NOT NULL, variant TEXT NOT NULL,
    op_kind TEXT NOT NULL, op_value REAL NOT NULL,
    n_removed INTEGER, n_kept INTEGER, err_removed REAL, err_kept REAL, acc_all REAL, acc_kept REAL,
    PRIMARY KEY (run_id, dataset, variant, op_kind, op_value));
CREATE TABLE IF NOT EXISTS review_run_flags (
    run_id TEXT NOT NULL, flag TEXT NOT NULL, note TEXT, PRIMARY KEY (run_id, flag));
CREATE TABLE IF NOT EXISTS noise_floor (
    family TEXT NOT NULL, dataset TEXT NOT NULL, variant TEXT NOT NULL,
    op_kind TEXT NOT NULL, op_value REAL NOT NULL, metric TEXT NOT NULL,
    n INTEGER, mean REAL, std REAL, min REAL, max REAL, members_json TEXT,
    PRIMARY KEY (family, dataset, variant, op_kind, op_value, metric));
CREATE TABLE IF NOT EXISTS review_meta (key TEXT PRIMARY KEY, value TEXT);
"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ decisions and metrics

def decide(S, taus, use):
    """Strongest head above its own threshold, otherwise nothing. Same rule as the campaign."""
    n = S.shape[0]
    out = np.full(n, "nothing", dtype=object)
    best = np.full(n, -np.inf)
    for h in use:
        s = S[:, HEADS.index(h)]
        m = (s > taus[h]) & (s > best)
        out[m] = h
        best[m] = s[m]
    return out


def heads_of(con, run_id):
    """Heads a run actually has: the ones with a stored background distribution."""
    return [r[0] for r in con.execute(
        "SELECT head FROM background_quantiles WHERE run_id=?", (run_id,)) if r[0] in HEADS]


def taus_synth(con, run_id, heads, far):
    return {h: db.threshold_at_far(con, run_id, h, far) for h in heads}


def taus_real(S, y, heads, far):
    quiet = y == "nothing"
    return {h: float(np.quantile(S[quiet, HEADS.index(h)], 1.0 - far)) for h in heads}


def prf(tp, fp, fn):
    rec = tp / (tp + fn) if tp + fn else None
    pre = tp / (tp + fp) if tp + fp else None
    f1 = (2 * pre * rec / (pre + rec)) if (pre and rec) else None
    return rec, pre, f1


def class_rows(y, pred, classes):
    """(cls, n, accuracy, recall, precision, f1, extra) for overall, each class, nothing, present."""
    rows = [("overall", int(len(y)), float((pred == y).mean()), None, None, None, None)]
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        rec, pre, f1 = prf(tp, fp, fn)
        rows.append((c, int((y == c).sum()), None, rec, pre, f1, None))
    nn_ = int((y == "nothing").sum())
    if nn_:
        rec = float(((pred == "nothing") & (y == "nothing")).sum() / nn_)
        rows.append(("nothing", nn_, None, rec, None, None, {"false_alarm_rate": 1.0 - rec}))
    yp, pp = y != "nothing", pred != "nothing"
    tp = int((yp & pp).sum()); fp = int((~yp & pp).sum()); fn = int((yp & ~pp).sum())
    rec, pre, f1 = prf(tp, fp, fn)
    far = float(fp / (~yp).sum()) if (~yp).sum() else None
    rows.append(("present", int(len(y)), float((yp == pp).mean()), rec, pre, f1,
                 {"false_alarm_rate": far}))
    return rows


def auc(y_bin, s):
    from sklearn.metrics import roc_auc_score
    if y_bin.min() == y_bin.max():
        return None
    return float(roc_auc_score(y_bin, s))


# ------------------------------------------------------------------ writers

def put_rm(con, run_id, ds, variant, op_kind, op_value, rows):
    con.executemany(
        "INSERT OR REPLACE INTO review_metrics (run_id,dataset,variant,op_kind,op_value,cls,n,"
        "accuracy,recall,precision_,f1,auc,extra_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(run_id, ds, variant, op_kind, float(op_value), c, n, a, r, p, f, None,
          json.dumps(x) if x else None) for (c, n, a, r, p, f, x) in rows])


def put_auc(con, run_id, ds, variant, cls, value, n, extra=None):
    con.execute(
        "INSERT OR REPLACE INTO review_metrics (run_id,dataset,variant,op_kind,op_value,cls,n,"
        "accuracy,recall,precision_,f1,auc,extra_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, ds, variant, "auc", 0.0, cls, int(n), None, None, None, None, value,
         json.dumps(extra) if extra else None))


# ------------------------------------------------------------------ the loudness rule

def register_loudness(con, real_sets, verbose=True):
    """One threshold on total loudness, calibrated on synthetic background like every model."""
    import pandas as pd
    run_id = "L0_loudness"
    d = os.path.join(D.SYNTH_ROOT, D.WINDOW_DIRS["3s"])
    shards = sorted(glob.glob(os.path.join(d, "features_shard_*.parquet")))
    cols = ["rms_total", "split", "human_level", "vehicle_level", "animal_level"]
    df = pd.concat([pd.read_parquet(s, columns=cols) for s in shards], ignore_index=True)
    bg = (df["split"] == "val").to_numpy().copy()
    for c in ("human_level", "vehicle_level", "animal_level"):
        lv = df[c].map(D.LVL)
        lv = lv.fillna(pd.to_numeric(df[c], errors="coerce")).to_numpy()
        bg &= (lv == 0)
    bg_rms = df["rms_total"].to_numpy(np.float64)[bg]
    del df
    db.upsert_run(con, run_id, grp="L", label="loudness threshold only, zero-shot",
                  config_json=json.dumps({"kind": "rule", "feature": "rms_total",
                                          "decision": "above threshold -> human"}),
                  status="ok", params=0, macs=0, sops=0.0, energy_uj=0.0, energy_ratio=None,
                  steps=0, wall_min=0.0, best_val=None,
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    con.execute("DELETE FROM background_quantiles WHERE run_id=?", (run_id,))
    db.put_background_quantiles(con, run_id, "human", bg_rms)
    j = F.FEATURE_NAMES.index("rms_total")
    for ds, (X, rows, info) in real_sets.items():
        block = np.zeros((len(X), 3), np.float32)
        block[:, 0] = X[:, j]
        db.put_scores(con, run_id, ds, block, heads=HEADS)
    if verbose:
        log(f"loudness rule registered: {len(bg_rms):,} synthetic background windows for its threshold")


# ------------------------------------------------------------------ bootstrap machinery

class BlockBoot:
    """Resampling plan for one dataset: 30 s blocks inside each recording, recordings kept
    separate so the class mix of the original set is preserved. Built once, reused per model."""

    def __init__(self, W, n_boot=N_BOOT, block_s=BLOCK_S, seed=0):
        key = np.array([f"{r}|{int(t // block_s)}" for r, t in zip(W["recording"], W["t0_s"])],
                       dtype=object)
        uniq, self.blk = np.unique(key, return_inverse=True)
        self.nb = np.bincount(self.blk, minlength=len(uniq)).astype(np.float64)
        rec_of_block = np.array([u.split("|")[0] for u in uniq], dtype=object)
        rng = np.random.default_rng(seed)
        self.Wb = np.zeros((n_boot, len(uniq)), np.float64)
        for r in np.unique(rec_of_block):
            ids = np.where(rec_of_block == r)[0]
            for b in range(n_boot):
                np.add.at(self.Wb[b], rng.choice(ids, len(ids), replace=True), 1.0)
        self.n_boot, self.block_s = n_boot, block_s

    def ci(self, correct):
        cb = np.bincount(self.blk, weights=correct.astype(np.float64), minlength=len(self.nb))
        acc = (self.Wb @ cb) / (self.Wb @ self.nb)
        return float(correct.mean()), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


# ------------------------------------------------------------------ per-run analysis

def review_run(con, run_id, datasets, boot=None, cleaning=None, verbose=False):
    heads = heads_of(con, run_id)
    if not heads:
        return False
    for ds, W in datasets.items():
        S, cols = db.get_scores(con, run_id, ds)
        if S is None:
            continue
        assert cols == HEADS, (run_id, ds, cols)
        y = W["label"]
        classes = [c for c in sorted(set(y.tolist())) if c != "nothing"]
        real_ok = bool((y == "nothing").any())
        for variant in VARIANTS:
            use = [h for h in heads if not (variant == "masked" and h == "animal")]
            ops = [("synth_far", f, taus_synth(con, run_id, use, f)) for f in SYNTH_FAR]
            if real_ok:
                ops += [("real_far", f, taus_real(S, y, use, f)) for f in REAL_FAR]
            for op_kind, op_value, taus in ops:
                pred = decide(S, taus, use)
                rows = class_rows(y, pred, classes)
                rows[0] = rows[0][:6] + ({"taus": {h: float(t) for h, t in taus.items()}},)
                put_rm(con, run_id, ds, variant, op_kind, op_value, rows)
                if boot is not None and ds in boot:
                    for (bk, bv) in BOOT_OPS:
                        if (bk, bv) == (op_kind, op_value):
                            yp, pp = y != "nothing", pred != "nothing"
                            for metric, corr in (("accuracy", pred == y), ("present_accuracy", yp == pp)):
                                pt, lo, hi = boot[ds].ci(corr)
                                con.execute(
                                    "INSERT OR REPLACE INTO review_bootstrap VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                    (run_id, ds, variant, op_kind, float(op_value), metric, pt, lo, hi,
                                     boot[ds].n_boot, boot[ds].block_s))
                if cleaning is not None and ds in cleaning and (op_kind, op_value) in CLEAN_OPS:
                    rm = cleaning[ds]
                    corr = (pred == y)
                    con.execute(
                        "INSERT OR REPLACE INTO review_cleaning VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, ds, variant, op_kind, float(op_value), int(rm.sum()), int((~rm).sum()),
                         float(1 - corr[rm].mean()) if rm.any() else None,
                         float(1 - corr[~rm].mean()), float(corr.mean()), float(corr[~rm].mean())))
            # threshold-free
            sc = np.max(S[:, [HEADS.index(h) for h in use]], axis=1) if use else np.zeros(len(y))
            a = auc((y != "nothing").astype(int), sc)
            if a is not None:
                put_auc(con, run_id, ds, variant, "present", a, len(y))
        # per-head separation does not depend on the variant
        for h in heads:
            s = S[:, HEADS.index(h)]
            if (y == h).any():
                a = auc((y == h).astype(int), s)
                if a is not None:
                    put_auc(con, run_id, ds, "active", h, a, len(y), {"positive": h, "negative": "all others"})
                if real_ok:
                    m = (y == h) | (y == "nothing")
                    a2 = auc((y[m] == h).astype(int), s[m])
                    if a2 is not None:
                        put_auc(con, run_id, ds, "active", f"{h}_vs_nothing", a2, int(m.sum()),
                                {"positive": h, "negative": "nothing"})
        con.commit()
    return True


# ------------------------------------------------------------------ noise floor and stats

METRIC_COLS = {"accuracy": "accuracy", "recall": "recall", "auc": "auc"}
FLOOR_METRICS = [("overall", "accuracy"), ("present", "accuracy"), ("nothing", "recall"),
                 ("human", "recall"), ("vehicle", "recall"), ("present", "auc"),
                 ("human", "auc"), ("vehicle", "auc"), ("human_vs_nothing", "auc"),
                 ("vehicle_vs_nothing", "auc")]


def build_noise_floor(con):
    con.execute("DELETE FROM noise_floor")
    keys = con.execute("SELECT DISTINCT dataset, variant, op_kind, op_value FROM review_metrics").fetchall()
    n_rows = 0
    for family, members in FAMILIES.items():
        for ds, variant, op_kind, op_value in keys:
            for cls, col in FLOOR_METRICS:
                vals, who = [], []
                for m in members:
                    v = con.execute(
                        f"SELECT {col} FROM review_metrics WHERE run_id=? AND dataset=? AND variant=? "
                        "AND op_kind=? AND op_value=? AND cls=?", (m, ds, variant, op_kind, op_value, cls)
                    ).fetchone()
                    if v and v[0] is not None:
                        vals.append(float(v[0])); who.append(m)
                if len(vals) < 2:
                    continue
                a = np.array(vals)
                con.execute("INSERT OR REPLACE INTO noise_floor VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (family, ds, variant, op_kind, float(op_value), f"{cls}:{col}", len(a),
                             float(a.mean()), float(a.std(ddof=1)), float(a.min()), float(a.max()),
                             json.dumps(who)))
                n_rows += 1
    con.commit()
    return n_rows


# ------------------------------------------------------------------ energy, recomputed

def build_energy(con):
    """Energy per window from the stored firing rates, three ways.

      as_run       the campaign's accounting: the first dense layer recomputed at every time step
      first_once   that layer computed once and broadcast -- what any deployment does with a
                   static input, and mathematically the same network
      inference    what a deployed copy pays: first_once for a spiking network, one pass for an
                   ordinary one (the "four passes" control collapses to one pass once dropout is off)

    Also spikes per activation per inference (rate x steps, per layer), the quantity the
    break-even argument in the hardware literature is written in.
    """
    from training import model as M
    con.execute("""CREATE TABLE IF NOT EXISTS review_energy (
        run_id TEXT PRIMARY KEY, kind TEXT, steps INTEGER, params INTEGER, macs INTEGER,
        energy_uj_as_run REAL, ratio_as_run REAL, energy_uj_first_once REAL, ratio_first_once REAL,
        energy_uj_inference REAL, ratio_inference REAL, spikes_per_activation_json TEXT, note TEXT)""")
    n = 0
    for r in db.runs_table(con):
        if r["status"] != "ok" or not r["config_json"]:
            continue
        cfg = json.loads(r["config_json"])
        if "widths" not in cfg or cfg.get("kind") == "classical":
            continue
        fire = json.loads(r["fire_json"]) if r["fire_json"] else []
        if cfg.get("pruned"):
            # priced over the surviving connections by campaign/prune.py; the same number
            # applies to every column because a pruned first layer is still computed once
            T = int(cfg.get("T", 4))
            spa = [float(x) * T for x in (fire or [])[:len(cfg["widths"])]]
            con.execute("INSERT OR REPLACE INTO review_energy VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["run_id"], "snn", T, r["params"], r["macs"], r["energy_uj"], r["energy_ratio"],
                         r["energy_uj"], r["energy_ratio"], r["energy_uj"], r["energy_ratio"],
                         json.dumps(spa), f"pruned to {100 * (1 - cfg['sparsity']):g}% of the weights"))
            n += 1
            continue
        fire = fire or [0.1]
        n_in = len(D.select_features(cfg.get("features", "all132")))
        c0 = M.cost_model(cfg, n_in, fire)
        c1 = M.cost_model(dict(cfg, stem_once=True), n_in, fire)
        note = None
        if cfg.get("kind") == "ann":
            ci = M.cost_model(dict(cfg, ann_passes=1), n_in, fire)
            if int(cfg.get("ann_passes", 1)) > 1:
                note = "passes are identical at inference; charged as one"
        else:
            ci = c1
        T = int(cfg.get("T", 4))
        spa = [float(x) * T for x in fire[:len(cfg["widths"])]] if cfg.get("kind") == "snn" else []
        con.execute("INSERT OR REPLACE INTO review_energy VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["run_id"], cfg.get("kind", "snn"), T, r["params"], c0["macs"],
                     c0["energy_uj"], c0["energy_ratio"], c1["energy_uj"], c1["energy_ratio"],
                     ci["energy_uj"], ci["energy_ratio"], json.dumps(spa), note))
        n += 1
    con.commit()
    return n


# ------------------------------------------------------------------ post-processing rules

def build_postprocess(con, dataset="elbit", far=0.001):
    """Hysteresis and N-of-M for every model with scores on the field recordings, per head,
    from the stored scores. Per-head binary presence (this head's label against everything
    else), so the rows read directly as a detector's recall / precision / false-alarm rate.

    rule       plain        one threshold, the campaign's reporting protocol
               hysteresis   param1 = lower threshold as a fraction of the upper one
               n_of_m       param1 = N, param2 = M
    """
    from evaluation import postprocess as P
    con.execute("""CREATE TABLE IF NOT EXISTS review_postprocess (
        run_id TEXT NOT NULL, dataset TEXT NOT NULL, head TEXT NOT NULL, rule TEXT NOT NULL,
        param1 REAL NOT NULL, param2 REAL NOT NULL, op_kind TEXT, op_value REAL,
        recall REAL, precision_ REAL, f1 REAL, false_alarm_rate REAL,
        PRIMARY KEY (run_id, dataset, head, rule, param1, param2))""")
    run_ids = [r[0] for r in con.execute(
        "SELECT DISTINCT run_id FROM score_blocks WHERE dataset=?", (dataset,))]
    n = 0
    for rid in run_ids:
        for head in ("human", "vehicle"):
            if head not in heads_of(con, rid):
                continue
            rows = []
            for r in P.hysteresis_sweep(con, rid, dataset, head=head, far_on=far):
                rule = "plain" if r["off_fraction"] == 1.0 else "hysteresis"
                rows.append((rid, dataset, head, rule, float(r["off_fraction"]), 0.0, "synth_far", far,
                             r["recall"], r["precision"], r["f1"], r["false_alarm_rate"]))
            for r in P.n_of_m_sweep(con, rid, dataset, head=head, far=far,
                                    grid=((2, 3), (3, 5), (4, 7), (5, 9))):
                rec, pre = r["recall"], r["precision"]
                f1 = (2 * pre * rec / (pre + rec)) if (pre and rec) else None
                rows.append((rid, dataset, head, "n_of_m", float(r["n"]), float(r["m"]), "synth_far", far,
                             rec, pre, f1, r["false_alarm_rate"]))
            con.executemany("INSERT OR REPLACE INTO review_postprocess VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            n += len(rows)
        con.commit()
    return n


def cleaning_mask(W):
    """Boolean keep-mask over the field windows from the rebuilt cleaning list, or None."""
    cp = os.path.join(OUT, "elbit_cleaning.json")
    if not os.path.exists(cp):
        return None
    ex = json.load(open(cp, encoding="utf-8"))
    rm = np.zeros(len(W["label"]), bool)
    for f, items in ex.items():
        ts = np.array([i["t0_s"] for i in items])
        if len(ts):
            for k in np.where(W["recording"] == f)[0]:
                if np.min(np.abs(ts - W["t0_s"][k])) < 1e-6:
                    rm[k] = True
    return ~rm


# ------------------------------------------------------------------ hysteresis, three-way

HYST_ENTRIES = (0.001, 0.002, 0.005, 0.01)
HYST_EXITS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)


def build_hysteresis_3way(con, dataset="elbit"):
    """Three-way accuracy with hysteresis on every head, over a grid of entry threshold
    (synthetic false-alarm rate) and exit threshold (fraction of the entry one).

    A head switches on when its score crosses the entry threshold and stays on until it drops
    below the exit one, reading consecutive windows inside each recording. The decision is then
    the strongest head that is on, otherwise nothing -- the same rule as everywhere else.
    exit_frac 1.0 with entry 0.001 is the plain per-window number.
    """
    from evaluation import postprocess as P
    con.execute("DROP TABLE IF EXISTS review_hysteresis")
    con.execute("""CREATE TABLE review_hysteresis (
        run_id TEXT NOT NULL, dataset TEXT NOT NULL, variant TEXT NOT NULL, subset TEXT NOT NULL,
        entry_far REAL NOT NULL, exit_frac REAL NOT NULL,
        accuracy REAL, human_recall REAL, human_precision REAL, vehicle_recall REAL, vehicle_precision REAL,
        nothing_recall REAL, present_accuracy REAL, present_recall REAL, present_precision REAL,
        PRIMARY KEY (run_id, dataset, variant, subset, entry_far, exit_frac))""")
    W = db.dataset_windows(con, dataset)
    y_all = W["label"]
    keep = cleaning_mask(W) if dataset == "elbit" else None
    subsets = [("all", np.ones(len(y_all), bool))] + ([("cleaned", keep)] if keep is not None else [])
    groups = []
    for rec in np.unique(W["recording"]):
        m = W["recording"] == rec
        groups.append(np.where(m)[0][np.argsort(W["t0_s"][m])])
    run_ids = [r[0] for r in con.execute("SELECT DISTINCT run_id FROM score_blocks WHERE dataset=?", (dataset,))]
    n = 0
    for rid in run_ids:
        heads = heads_of(con, rid)
        if not heads:
            continue
        S, cols = db.get_scores(con, rid, dataset)
        rows = []
        for variant in VARIANTS:
            use = [h for h in heads if not (variant == "masked" and h == "animal")]
            for entry in HYST_ENTRIES:
                taus = taus_synth(con, rid, use, entry)
                for frac in HYST_EXITS:
                    on = {}
                    for h in use:
                        col = S[:, HEADS.index(h)]
                        o = np.zeros(len(y_all), bool)
                        for idx in groups:
                            o[idx] = P.hysteresis(col[idx], taus[h], taus[h] * frac)
                        on[h] = o
                    pred_all = np.full(len(y_all), "nothing", dtype=object)
                    best = np.full(len(y_all), -np.inf)
                    for h in use:
                        col = S[:, HEADS.index(h)]
                        m = on[h] & (col > best)
                        pred_all[m] = h
                        best[m] = col[m]
                    # the rule runs on the full sequence; the cleaned subset only changes which
                    # windows are counted, never what the rule saw
                    for sub_name, sub in subsets:
                        pred, y = pred_all[sub], y_all[sub]
                        rec_of = lambda c: float(((pred == c) & (y == c)).sum() / max(1, (y == c).sum()))
                        pre_of = lambda c: (float(((pred == c) & (y == c)).sum() / (pred == c).sum())
                                            if (pred == c).sum() else None)
                        yp, pp = y != "nothing", pred != "nothing"
                        rows.append((rid, dataset, variant, sub_name, entry, frac, float((pred == y).mean()),
                                     rec_of("human"), pre_of("human"), rec_of("vehicle"), pre_of("vehicle"),
                                     rec_of("nothing"), float((yp == pp).mean()),
                                     float((yp & pp).sum() / yp.sum()),
                                     float((yp & pp).sum() / pp.sum()) if pp.sum() else None))
        con.executemany("INSERT OR REPLACE INTO review_hysteresis VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        n += len(rows)
    return n


def campaign_stats(con):
    """What drives the ranking: rank correlations across every scored model on the field set."""
    from scipy.stats import spearmanr
    q = lambda variant, kind, val, cls, col: dict(con.execute(
        f"SELECT run_id, {col} FROM review_metrics WHERE dataset='elbit' AND variant=? AND op_kind=? "
        f"AND op_value=? AND cls=? AND {col} IS NOT NULL", (variant, kind, val, cls)).fetchall())
    acc_a = q("active", "synth_far", 0.001, "overall", "accuracy")
    acc_m = q("masked", "synth_far", 0.001, "overall", "accuracy")
    acc_m1 = q("masked", "synth_far", 0.01, "overall", "accuracy")
    acc_r5 = q("masked", "real_far", 0.05, "overall", "accuracy")
    bgr = q("masked", "synth_far", 0.001, "nothing", "recall")
    auc_p = q("masked", "auc", 0.0, "present", "auc")
    bv = dict(con.execute("SELECT run_id, best_val FROM runs WHERE best_val IS NOT NULL").fetchall())
    flagged = {r[0] for r in con.execute("SELECT run_id FROM review_run_flags WHERE flag IN "
                                         "('rule_baseline','discrete_score')")}
    ids = [r for r in acc_m if r in bgr and r in auc_p and r not in flagged]
    v = lambda d: np.array([d[r] for r in ids], dtype=float)
    out = {"n_runs": len(ids), "note": "field recordings, animal head masked unless stated"}
    if len(ids) > 3:
        out["spearman_accuracy_vs_background_recall"] = float(spearmanr(v(acc_m), v(bgr))[0])
        out["spearman_accuracy_vs_present_auc"] = float(spearmanr(v(acc_m), v(auc_p))[0])
        out["spearman_accuracy_0.1pct_vs_1pct"] = float(spearmanr(v(acc_m), v(acc_m1))[0])
        out["spearman_accuracy_zero_shot_vs_matched_real_5pct"] = float(spearmanr(v(acc_m), v(acc_r5))[0])
        ib = [r for r in ids if r in bv]
        out["spearman_accuracy_vs_synthetic_best_val"] = float(
            spearmanr([acc_m[r] for r in ib], [bv[r] for r in ib])[0])
        out["spearman_active_vs_masked_accuracy"] = float(spearmanr(v(acc_a), v(acc_m))[0])
        out["std_accuracy_masked"] = float(v(acc_m).std())
        out["std_present_auc"] = float(v(auc_p).std())
        out["std_background_recall"] = float(v(bgr).std())
    con.execute("INSERT OR REPLACE INTO review_meta VALUES ('campaign_stats', ?)", (json.dumps(out),))
    con.commit()
    return out


def leaderboard(con, limit=None):
    rows = con.execute("""
        SELECT r.run_id,
          (SELECT accuracy FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='active' AND op_kind='synth_far' AND op_value=0.001 AND cls='overall') AS act,
          (SELECT accuracy FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='masked' AND op_kind='synth_far' AND op_value=0.001 AND cls='overall') AS msk,
          (SELECT accuracy FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='masked' AND op_kind='synth_far' AND op_value=0.01 AND cls='overall') AS msk1,
          (SELECT accuracy FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='masked' AND op_kind='real_far' AND op_value=0.05 AND cls='overall') AS real5,
          (SELECT auc FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='masked' AND op_kind='auc' AND cls='present') AS aucp,
          (SELECT auc FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='active' AND op_kind='auc' AND cls='human_vs_nothing') AS auch,
          (SELECT auc FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='active' AND op_kind='auc' AND cls='vehicle_vs_nothing') AS aucv,
          (SELECT recall FROM review_metrics m WHERE m.run_id=r.run_id AND dataset='elbit' AND variant='masked' AND op_kind='synth_far' AND op_value=0.001 AND cls='nothing') AS bgr,
          (SELECT group_concat(flag) FROM review_run_flags f WHERE f.run_id=r.run_id) AS flags
        FROM runs r WHERE r.status='ok' ORDER BY msk DESC""").fetchall()
    rows = [r for r in rows if r[2] is not None]
    return rows[:limit] if limit else rows


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-loudness", action="store_true")
    ap.add_argument("--no-boot", action="store_true")
    args = ap.parse_args()

    con = db.connect(DB_PATH)
    con.executescript(SCHEMA)
    con.executemany("INSERT OR REPLACE INTO review_run_flags VALUES (?,?,?)", FLAGS)
    con.commit()

    ds_names = [r[0] for r in con.execute("SELECT dataset FROM dataset_manifest")]
    datasets = {ds: db.dataset_windows(con, ds) for ds in ds_names}
    log(f"datasets: " + ", ".join(f"{k} {len(v['label']):,}" for k, v in datasets.items()))

    if not args.no_loudness:
        real_sets = {ds: prepare_real.cached(ds) for ds in ds_names}
        real_sets = {k: v for k, v in real_sets.items() if v is not None}
        register_loudness(con, real_sets)

    boot = None if args.no_boot else {"elbit": BlockBoot(datasets["elbit"])} if "elbit" in datasets else None
    cleaning = None
    if "elbit" in datasets:
        keep = cleaning_mask(datasets["elbit"])
        if keep is not None:
            cleaning = {"elbit": ~keep}
            log(f"cleaning list: {(~keep).sum()} of {len(keep)} field windows excluded")

    run_ids = [r[0] for r in con.execute("SELECT run_id FROM runs WHERE status='ok' ORDER BY run_id")]
    if args.only:
        run_ids = [r for r in run_ids if r in set(args.only)]
    t0 = time.time()
    done = 0
    for rid in run_ids:
        if review_run(con, rid, datasets, boot=boot, cleaning=cleaning):
            done += 1
    log(f"re-analysed {done} runs in {time.time() - t0:.0f}s")
    n = build_noise_floor(con)
    log(f"noise floor: {n} rows")
    log(f"energy recomputed for {build_energy(con)} runs")
    log(f"post-processing rules: {build_postprocess(con)} rows")
    log(f"hysteresis, three-way: {build_hysteresis_3way(con)} rows")
    stats = campaign_stats(con)
    con.execute("INSERT OR REPLACE INTO review_meta VALUES ('run_at', ?)",
                (time.strftime("%Y-%m-%dT%H:%M:%S"),))
    con.commit()
    print(json.dumps(stats, indent=1))

    print("\ncorrected table -- field recordings, animal head masked, sorted by zero-shot accuracy at 0.1%")
    print(f"{'run':<20} {'active':>7} {'masked':>7} {'@1%':>7} {'real5%':>7} {'AUCpres':>7} {'AUCh':>6} {'AUCv':>6} {'bgrec':>6}  flags")
    for r in leaderboard(con):
        f = lambda x: "  -   " if x is None else f"{x:.4f}"
        print(f"{r[0]:<20} {f(r[1]):>7} {f(r[2]):>7} {f(r[3]):>7} {f(r[4]):>7} {f(r[5]):>7} "
              f"{f(r[6])[:5]:>6} {f(r[7])[:5]:>6} {f(r[8])[:5]:>6}  {r[9] or ''}")
    print("\nnoise floor, field recordings, masked, zero-shot 0.1%:")
    for r in con.execute("SELECT family, metric, n, mean, std, min, max FROM noise_floor WHERE dataset='elbit' "
                         "AND variant='masked' AND op_kind='synth_far' AND op_value=0.001 "
                         "AND metric IN ('overall:accuracy','present:accuracy','nothing:recall')"):
        print(f"  {r[0]:<10} {r[1]:<18} n={r[2]:<3} mean {r[3]:.4f} std {r[4]:.4f} range {r[5]:.4f}-{r[6]:.4f}")
    for r in con.execute("SELECT family, metric, n, mean, std, min, max FROM noise_floor WHERE dataset='elbit' "
                         "AND variant='active' AND op_kind='auc' AND metric LIKE '%vs_nothing:auc'"):
        print(f"  {r[0]:<10} {r[1]:<26} n={r[2]:<3} mean {r[3]:.4f} std {r[4]:.4f} range {r[5]:.4f}-{r[6]:.4f}")
    print("\nenergy, ratio against the same network run as an ordinary one (higher is better):")
    print(f"{'run':<20} {'steps':>5} {'as run':>7} {'1st once':>8} {'infer':>7}  spikes/activation per layer")
    for r in con.execute("SELECT run_id, steps, ratio_as_run, ratio_first_once, ratio_inference, "
                         "spikes_per_activation_json, note FROM review_energy WHERE run_id IN "
                         "('A1_baseline','A3_snn_T1','A4_T2','A5_T8','A6_T16','A8_stem_once','A2_ann_1pass',"
                         "'A7_ann_4pass','E1_narrow','E2_wide','B3_2head_noanimal','G8_upto_cep16') ORDER BY steps, run_id"):
        print(f"{r[0]:<20} {r[1]:>5} {r[2]:>7.2f} {r[3]:>8.2f} {r[4]:>7.2f}  {[round(x,2) for x in json.loads(r[5])]}  {r[6] or ''}")


if __name__ == "__main__":
    main()
