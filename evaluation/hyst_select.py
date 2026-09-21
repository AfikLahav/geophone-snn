"""Choose each model's hysteresis exit fraction on SYNTHETIC validation streams, then read off
the field result at that setting from the stored field sweep. Nothing about the field touches
the choice, so the field number stays zero-shot.

For every run listed: rebuild the model, score synthetic validation, take the entry threshold
from the stored background quantiles (0.1% false alarms), run hysteresis along each validation
scene in time order for exits 1.0 .. 0.3, and keep the exit with the best three-way accuracy
on validation (animal scenes excluded, mixed scenes count as their louder class). Writes

  review_hyst_select(run_id, exit_frac, synth_acc_plain, synth_acc_hyst,
                     field_acc_plain, field_acc_hyst, field_acc_hyst_cleaned,
                     human_recall, human_precision, vehicle_recall, vehicle_precision)

    python campaign/hyst_select.py [run_id ...]
"""
import json
import os
import sys

import numpy as np
import torch
from spikingjelly.activation_based import functional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simgeo", "simgeo_v42"))
from training import data as D, db, model as M, evaluate as S
from evaluation import postprocess as P

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))
EXITS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

DEFAULT = ([f"U2_lowrank16_rep{i}" for i in (1, 2, 3)] + [f"S5_256x128_rep{i}" for i in (1, 2, 3)]
           + [f"S6_512x256x128_rep{i}" for i in (1, 2, 3)]
           + [f"{n}_rep{i}" for n in ("X1_ann_64x32_lr16", "X2_ann_256x128", "X3_ann_512x256x128",
                                       "X4_gru16", "X5_gru100", "X6_gru225", "X7_attn16", "X8_attn64",
                                       "X9_attn128") for i in (1, 2)]
           + [f"{n}_rep{i}" for n in ("Y1_T1", "Y2_lif", "Y3_nodrop") for i in (1, 2, 3)])


def hyst_fast(s, t_on, t_off, stream_start):
    """Vectorized hysteresis over concatenated streams. on[i] = s>t_on, or (s>t_off and on[i-1]);
    a stream start resets the state. Since t_off <= t_on, a window is on exactly when it sits in a
    run of consecutive above-t_off windows (not crossing a stream start) that has already
    contained an above-t_on window."""
    n = len(s); idx = np.arange(n)
    a = s > t_on; b = s > t_off
    run_start = b & (np.r_[True, ~b[:-1]] | stream_start)
    run_first = np.maximum.accumulate(np.where(run_start, idx, 0))
    last_a = np.maximum.accumulate(np.where(a, idx, -1))
    return b & (last_a >= run_first)


def three_way(sc_h, sc_v, on_h, on_v):
    d = np.where(sc_h >= sc_v, "human", "vehicle")
    return np.where(on_h & on_v, d, np.where(on_h, "human", np.where(on_v, "vehicle", "nothing")))


def main(run_ids):
    con = db.connect(os.path.join(OUT, "campaign.sqlite"))
    con.execute("""CREATE TABLE IF NOT EXISTS review_hyst_select (
        run_id TEXT PRIMARY KEY, exit_frac REAL, synth_acc_plain REAL, synth_acc_hyst REAL,
        field_acc_plain REAL, field_acc_hyst REAL, field_acc_hyst_cleaned REAL,
        human_recall REAL, human_precision REAL, vehicle_recall REAL, vehicle_precision REAL)""")
    syn = D.SyntheticData(window="3s", device="cuda")
    for run_id in run_ids:
        row = con.execute("SELECT config_json FROM runs WHERE run_id=? AND status='ok'", (run_id,)).fetchone()
        if not row:
            print("  skip", run_id); continue
        cfg = json.loads(row[0])
        prepared = syn.prepare(cfg)
        net = M.build(cfg, len(prepared["features"])).cuda()
        net.load_state_dict(torch.load(os.path.join(OUT, "models", run_id, "model_ema.pt"), map_location="cuda"))
        functional.set_step_mode(net, "m")
        sv = S.model_scores(net, prepared["Xva"].cpu().numpy())
        mv = prepared["meta_val"]
        coarse = mv["coarse"]; scene = mv["scene_id"]; t0 = mv["t0"]
        keep = coarse != "animal"
        # truth: present classes from the level labels (mixed scenes -> the class that is present)
        lh = prepared["labels"]["human"]["va_lvl"].cpu().numpy() > 0
        lv = prepared["labels"]["vehicle"]["va_lvl"].cpu().numpy() > 0
        truth = np.where(lh & lv, "mixed", np.where(lh, "human", np.where(lv, "vehicle", "nothing")))
        tau = {h: db.threshold_at_far(con, run_id, h, 0.001) for h in ("human", "vehicle")}
        # streams: validation rows are in scene, t0 order; group by scene
        order = np.lexsort((t0, scene))
        sc_h, sc_v = sv["human"][order], sv["vehicle"][order]
        sc_id = scene[order]; tr = truth[order]; kp = keep[order]
        starts = np.r_[True, sc_id[1:] != sc_id[:-1]]
        best = None
        accs = {}
        for ex in EXITS:
            on_h = hyst_fast(sc_h, tau["human"], tau["human"] * ex, starts)
            on_v = hyst_fast(sc_v, tau["vehicle"], tau["vehicle"] * ex, starts)
            pred = three_way(sc_h, sc_v, on_h, on_v)
            ok = (pred == tr) | ((tr == "mixed") & (pred != "nothing"))
            accs[ex] = float(ok[kp].mean())
            if best is None or accs[ex] > accs[best] + 1e-9:
                best = ex
        f = con.execute("""SELECT accuracy, human_recall, human_precision, vehicle_recall, vehicle_precision
                           FROM review_hysteresis WHERE run_id=? AND dataset='elbit' AND variant='masked'
                           AND subset='all' AND entry_far=0.001 AND exit_frac=?""", (run_id, best)).fetchone()
        fc = con.execute("""SELECT accuracy FROM review_hysteresis WHERE run_id=? AND dataset='elbit' AND
                            variant='masked' AND subset='cleaned' AND entry_far=0.001 AND exit_frac=?""",
                         (run_id, best)).fetchone()
        fp = con.execute("""SELECT accuracy FROM review_hysteresis WHERE run_id=? AND dataset='elbit' AND
                            variant='masked' AND subset='all' AND entry_far=0.001 AND exit_frac=1.0""",
                         (run_id,)).fetchone()
        vals = (run_id, best, accs[1.0], accs[best], fp[0] if fp else None, f[0] if f else None,
                fc[0] if fc else None, *(f[1:] if f else (None,) * 4))
        db._write(con, "INSERT OR REPLACE INTO review_hyst_select VALUES (?,?,?,?,?,?,?,?,?,?,?)", vals)
        db._commit(con)
        print(f"  {run_id:26s} exit {best:.1f}  synth {accs[1.0]:.3f}->{accs[best]:.3f}  "
              f"field {vals[4]:.3f}->{vals[5]:.3f} cleaned {vals[6]:.3f}" if f else f"  {run_id}: no field sweep row")
        del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
