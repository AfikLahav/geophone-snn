"""Post-processing, from the database alone.

Nothing here loads a model. Every function takes stored scores and stored window times and
returns numbers, which is the whole reason the scores are kept per window rather than collapsed
into a metric.

What lives here:

  threshold sweep   accuracy across the full range of operating points, derived from the stored
                    background distribution, so any false-alarm rate can be chosen after the fact

  hysteresis        two thresholds instead of one: a high one to start calling something present
                    and a lower one to keep calling it, so a detection does not flicker off for a
                    single quiet window. Reads consecutive windows inside each recording, which is
                    why the start time is stored.

  N-of-M            call it present when at least N of the last M windows are above threshold.
                    The cheap alternative to hysteresis, and worth comparing against it.

  event scoring     collapse runs of consecutive windows into events, so a long detection counts
                    once rather than forty times -- closer to what a deployment actually cares
                    about than per-window accuracy.

The per-window number stays the headline. Smoothing was dropped from the reporting protocol
because it costs detection latency, so anything here is a separate, clearly-labelled search.
"""
import numpy as np

from training import db


def _ordered(con, run_id, dataset):
    """Scores and window records for one model on one dataset, grouped by recording and in time
    order inside each -- the shape every rule below needs."""
    w = db.dataset_windows(con, dataset)
    s, heads = db.get_scores(con, run_id, dataset)
    if w is None or s is None:
        return None
    groups = {}
    for rec in np.unique(w["recording"]):
        m = w["recording"] == rec
        order = np.argsort(w["t0_s"][m])
        idx = np.where(m)[0][order]
        groups[rec] = idx
    return w, s, heads, groups


def threshold_sweep(con, run_id, dataset, head="human", fars=None):
    """Accuracy and recall across operating points, without retraining or rescoring."""
    fars = fars or [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    got = _ordered(con, run_id, dataset)
    if got is None:
        return []
    w, s, heads, _ = got
    hi = heads.index(head)
    y = (w["label"] == head)
    out = []
    for far in fars:
        t = db.threshold_at_far(con, run_id, head, far)
        if t is None:
            continue
        pred = s[:, hi] > t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        out.append(dict(far_target=far, threshold=float(t),
                        recall=tp / (tp + fn) if tp + fn else None,
                        precision=tp / (tp + fp) if tp + fp else None,
                        realized_far=float((s[~y, hi] > t).mean())))
    return out


def hysteresis(scores, t_on, t_off):
    """Once it is on, keep it on until the score drops below the lower threshold.

    Removes the single-window flicker that a one-threshold rule produces at the edge of
    detection, at the cost of holding a detection slightly too long."""
    out = np.zeros(len(scores), dtype=bool)
    on = False
    for i, v in enumerate(scores):
        on = (v > t_on) if not on else (v > t_off)
        out[i] = on
    return out


def n_of_m(scores, t, n=3, m=5):
    """Present when at least n of the last m windows are above the threshold."""
    above = (scores > t).astype(np.int32)
    if len(above) < m:
        return above.astype(bool)
    c = np.convolve(above, np.ones(m, dtype=np.int32), mode="full")[:len(above)]
    return c >= n


def hysteresis_sweep(con, run_id, dataset, head="human", far_on=0.001, offs=None):
    """Search the lower threshold, holding the upper one at a fixed false-alarm rate.

    The search is sparse on purpose: a fine grid over both thresholds overfits a 2,353-window set
    long before it finds anything real."""
    got = _ordered(con, run_id, dataset)
    if got is None:
        return []
    w, s, heads, groups = got
    hi = heads.index(head)
    t_on = db.threshold_at_far(con, run_id, head, far_on)
    if t_on is None:
        return []
    offs = offs or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    y = (w["label"] == head)
    rows = []
    for frac in offs:
        t_off = t_on * frac
        pred = np.zeros(len(s), dtype=bool)
        for rec, idx in groups.items():
            pred[idx] = hysteresis(s[idx, hi], t_on, t_off)
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        rec_ = tp / (tp + fn) if tp + fn else None
        pre = tp / (tp + fp) if tp + fp else None
        rows.append(dict(t_on=float(t_on), t_off=float(t_off), off_fraction=frac,
                         recall=rec_, precision=pre,
                         f1=(2 * pre * rec_ / (pre + rec_)) if (pre and rec_) else None,
                         false_alarm_rate=float(pred[~y].mean())))
    return rows


def n_of_m_sweep(con, run_id, dataset, head="human", far=0.001, grid=((2, 3), (3, 5), (4, 7))):
    got = _ordered(con, run_id, dataset)
    if got is None:
        return []
    w, s, heads, groups = got
    hi = heads.index(head)
    t = db.threshold_at_far(con, run_id, head, far)
    if t is None:
        return []
    y = (w["label"] == head)
    rows = []
    for n, m in grid:
        pred = np.zeros(len(s), dtype=bool)
        for rec, idx in groups.items():
            pred[idx] = n_of_m(s[idx, hi], t, n, m)
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        rows.append(dict(n=n, m=m, recall=tp / (tp + fn) if tp + fn else None,
                         precision=tp / (tp + fp) if tp + fp else None,
                         false_alarm_rate=float(pred[~y].mean())))
    return rows


def events(con, run_id, dataset, head="human", far=0.001, min_windows=2):
    """Collapse consecutive detections into events.

    A deployment cares whether an intruder was reported once, not whether forty consecutive
    windows each said so -- and a single false window that never repeats is not a false alarm in
    the operational sense."""
    got = _ordered(con, run_id, dataset)
    if got is None:
        return {}
    w, s, heads, groups = got
    hi = heads.index(head)
    t = db.threshold_at_far(con, run_id, head, far)
    if t is None:
        return {}
    detected, missed, false_events = 0, 0, 0
    for rec, idx in groups.items():
        pred = s[idx, hi] > t
        truth = (w["label"][idx] == head)
        runs, cur = [], None
        for i, p in enumerate(pred):
            if p and cur is None:
                cur = i
            elif not p and cur is not None:
                runs.append((cur, i))
                cur = None
        if cur is not None:
            runs.append((cur, len(pred)))
        runs = [r for r in runs if r[1] - r[0] >= min_windows]
        for a, b in runs:
            if truth[a:b].any():
                detected += 1
            else:
                false_events += 1
        if truth.any() and not any(truth[a:b].any() for a, b in runs):
            missed += 1
    hours = float(np.ptp(w["t0_s"])) / 3600.0 if len(w["t0_s"]) else 0.0
    return dict(events_detected=detected, events_missed=missed,
                false_events=false_events, min_windows=min_windows,
                false_events_per_hour=(false_events / hours) if hours > 0 else None)


def summary(con, run_id, dataset="elbit", head="human"):
    """Everything the post-processing search produces for one model, in one call."""
    return dict(
        thresholds=threshold_sweep(con, run_id, dataset, head),
        hysteresis=hysteresis_sweep(con, run_id, dataset, head),
        n_of_m=n_of_m_sweep(con, run_id, dataset, head),
        events=events(con, run_id, dataset, head),
    )


if __name__ == "__main__":
    import argparse
    import json
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--dataset", default="elbit")
    ap.add_argument("--head", default="human")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.environ.get("GEO_CAMPAIGN_OUT",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "campaign_out"))
    con = db.connect(os.path.join(root, "campaign.sqlite"))
    res = summary(con, a.run_id, a.dataset, a.head)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
