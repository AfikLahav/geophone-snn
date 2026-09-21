"""Score trained models and compute thresholded and threshold-free metrics."""
import json
import os

import numpy as np
import torch
from spikingjelly.activation_based import functional

from . import db

FAR_POINTS = {"far_1pct": 0.01, "far_0.1pct": 0.001}
CLASS_OF_HEAD = {"human": "human", "vehicle": "vehicle", "animal": "animal"}


def _presence(logits):
    """Presence probability of a head: sigmoid of the first output. Never a softmax."""
    return torch.sigmoid(logits[:, 0]).float().cpu().numpy()


def model_scores(net, X, device="cuda", batch=8192, heads=None):
    """Raw per-head presence scores for a feature matrix."""
    net.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch]).to(device)
            functional.reset_net(net)
            o = net(xb)
            for h, v in o.items():
                if h == "fourway":
                    p = torch.softmax(v, -1).float().cpu().numpy()
                    out.setdefault(h, []).append(p)
                else:
                    out.setdefault(h, []).append(_presence(v))
    return {h: np.concatenate(v) for h, v in out.items()}


def fourway_to_heads(p4):
    """Turn the mutually-exclusive control's output into the same three presence numbers, so it
    can be compared on equal terms with everything else."""
    return {"human": p4[:, 1], "vehicle": p4[:, 2], "animal": p4[:, 3]}


def background_scores(net, Xva, bg_mask, device="cuda"):
    """Scores on synthetic windows containing no source at all -- where thresholds come from."""
    s = model_scores(net, Xva[bg_mask], device=device)
    if "fourway" in s:
        s = fourway_to_heads(s["fourway"])
    return s


def decide(scores, taus, heads=("human", "vehicle", "animal")):
    """Three-way decision: the strongest head above its own threshold, otherwise nothing."""
    present = {h: scores[h] > taus[h] for h in heads if h in scores}
    n = len(next(iter(scores.values())))
    out = np.array(["nothing"] * n, dtype=object)
    best = np.full(n, -np.inf)
    for h in heads:
        if h not in scores:
            continue
        m = present[h] & (scores[h] > best)
        out[m] = h
        best[m] = scores[h][m]
    return out


def metrics_for(y_true, y_pred, classes):
    """Accuracy, and per-class recall / precision / F1. Plain counting, nothing clever."""
    out = {"overall": {"accuracy": float((y_pred == y_true).mean()), "n": int(len(y_true))}}
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        rec = tp / (tp + fn) if tp + fn else None
        pre = tp / (tp + fp) if tp + fp else None
        f1 = (2 * pre * rec / (pre + rec)) if (pre and rec) else None
        out[c] = {"recall": rec, "precision": pre, "f1": f1, "n": int((y_true == c).sum())}
    return out


def auc_for(y_true, score, positive):
    """Threshold-free separation. Recorded because it was asked for, and flagged diagnostic:
    it hides where the operating point sits, which on a session-confounded set is exactly the
    thing that matters."""
    from sklearn.metrics import roc_auc_score
    y = (y_true == positive).astype(int)
    if y.min() == y.max():
        return None
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return None


def score_run(con, run_id, net, prepared, real_sets, bg_mask, device="cuda", verbose=True):
    """Score one network everywhere and write everything the report and post-processing need."""
    # a run may use a feature subset; the cached real tables always hold all 132, so the same
    # columns the model was trained on are selected here by name
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                      "simgeo_v42"))
    import features as _F
    feat_idx = [_F.FEATURE_NAMES.index(n) for n in prepared["features"]]

    heads = [h for h in ("human", "vehicle", "animal") if h in prepared["labels"]]
    if not heads:                      # four-way control: its output is mapped back
        heads = ["human", "vehicle", "animal"]

    # 1. thresholds, from synthetic background only
    bg = background_scores(net, prepared["Xva"], bg_mask, device=device)

    # 2. every real dataset
    real_scored = {}
    for name, (X, rows, info) in real_sets.items():
        # the cached real features are RAW; the network expects them through the same scaler the
        # synthetic side was fitted with. Feeding raw values saturates every head to zero and
        # still produces varying scores, so nothing downstream notices -- it has to be done here.
        Xs = prepared["scaler"].transform(X[:, feat_idx])
        K = int(prepared.get("context", 1) or 1)
        if K > 1:
            # sequence models: the K consecutive windows of the same recording ending at each one
            from .data import context_index, SeqView
            S = context_index([r["recording"] for r in rows], [r["pos"] for r in rows], K, step=1.0)
            Xs = SeqView(torch.as_tensor(np.asarray(Xs, dtype=np.float32)), torch.as_tensor(S))
        s = model_scores(net, Xs, device=device)
        if "fourway" in s:
            s = fourway_to_heads(s["fourway"])
        real_scored[name] = s

    # 3. synthetic validation, for the in-domain number and the recall-against-level curve
    sv = model_scores(net, prepared["Xva"].cpu().numpy(), device=device)
    if "fourway" in sv:
        sv = fourway_to_heads(sv["fourway"])
    taus = record_scores(con, run_id, bg, real_scored, sv, real_sets, prepared, heads, verbose)
    multiplicity_metrics(con, run_id, net, prepared, device=device)
    return taus


def multiplicity_metrics(con, run_id, net, prepared, device="cuda", batch=8192):
    """Separation of "more than one" against "exactly one", among synthetic validation windows
    where the source is present, for every head that has a second output. The field recordings
    carry no such label, so synthetic validation is the only place this output can be scored."""
    from sklearn.metrics import roc_auc_score
    net.eval()
    outs = {}
    with torch.no_grad():
        Xva = prepared["Xva"]
        for i in range(0, Xva.shape[0], batch):
            functional.reset_net(net)
            o = net(Xva[i:i + batch])
            for h, v in o.items():
                if h != "fourway" and v.shape[1] >= 2:
                    outs.setdefault(h, []).append(v[:, 1].float().cpu().numpy())
    for h, parts in outs.items():
        if h not in prepared["labels"]:
            continue
        lvl = prepared["labels"][h]["va_lvl"].cpu().numpy()
        s2 = np.concatenate(parts)
        m = lvl >= 1
        y = (lvl[m] >= 2).astype(int)
        if y.min() == y.max():
            continue
        db.put_metric(con, run_id, "synthetic_val", "multiplicity", h,
                      auc=float(roc_auc_score(y, s2[m])), n=int(m.sum()),
                      extra={"positive": "more than one", "negative": "exactly one",
                             "share_positive": float(y.mean())})


def record_scores(con, run_id, bg, real_scored, sv, real_sets, prepared, heads, verbose=True):
    """Write one model's scores, thresholds and metrics. Shared by the networks and the ordinary
    machine-learning models, so both land in the same tables and mean the same thing.

    bg           {head: scores on synthetic background windows}
    real_scored  {dataset: {head: scores on that dataset's windows}}
    sv           {head: scores on the whole synthetic validation split}, or None to skip the
                 recall-against-level entry
    """
    for h, v in bg.items():
        db.put_background_quantiles(con, run_id, h, v)
    taus = {name: {h: db.threshold_at_far(con, run_id, h, far) for h in bg}
            for name, far in FAR_POINTS.items()}

    for name, (X, rows, info) in real_sets.items():
        s = real_scored[name]
        order = ["human", "animal", "vehicle"]
        block = np.stack([s.get(h, np.zeros(len(X), np.float32)) for h in order], 1)
        db.put_scores(con, run_id, name, block, heads=order)

        y_true = np.array([r["label"] for r in rows], dtype=object)
        classes = sorted(set(y_true.tolist()))
        for op, tau in taus.items():
            y_pred = decide(s, tau, heads=tuple(h for h in order if h in s))
            mm = metrics_for(y_true, y_pred, [c for c in classes if c != "nothing"])
            db.put_metric(con, run_id, name, op, "overall",
                          accuracy=mm["overall"]["accuracy"], n=mm["overall"]["n"],
                          extra={"taus": {k: float(v) for k, v in tau.items() if v is not None}})
            for c in classes:
                if c == "nothing":
                    nn_ = int((y_true == "nothing").sum())
                    rec = float(((y_pred == "nothing") & (y_true == "nothing")).sum() / nn_) if nn_ else None
                    db.put_metric(con, run_id, name, op, "nothing", recall=rec, n=nn_)
                elif c in mm:
                    db.put_metric(con, run_id, name, op, c, n=mm[c]["n"], recall=mm[c]["recall"],
                                  precision=mm[c]["precision"], f1=mm[c]["f1"])
        for h in order:
            if h in s and h in set(classes):
                a = auc_for(y_true, s[h], h)
                if a is not None:
                    db.put_metric(con, run_id, name, "threshold_free", h, auc=a,
                                  extra={"diagnostic_only": True})
        if verbose:
            acc = {op: round(float((decide(s, tau, tuple(h for h in order if h in s)) == y_true).mean()), 4)
                   for op, tau in taus.items()}
            print(f"    {name:<10} {len(X):>7,} windows  {acc}", flush=True)

    if sv is not None:
        for h in heads:
            if h not in sv or h not in prepared["labels"]:
                continue                  # the mutually-exclusive control has no per-class head
            lvl = prepared["labels"][h]["va_lvl"].cpu().numpy()
            snr = prepared["labels"][h]["va_snr"].cpu().numpy()
            det = (lvl > 0) & (snr > 0)
            for op, far in FAR_POINTS.items():
                t = db.threshold_at_far(con, run_id, h, far)
                if t is None or det.sum() == 0:
                    continue
                db.put_metric(con, run_id, "synthetic_val", op, h,
                              recall=float((sv[h][det] > t).mean()), n=int(det.sum()),
                              extra={"subset": "detectable (present and above the noise)"})
    return taus


# ---------------------------------------------------------------- CLI entry point
if __name__ == "__main__":
    import argparse, json, torch
    from . import model as M
    from .data import SyntheticData

    parser = argparse.ArgumentParser(description="Evaluate a trained model on a feature table.")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument("--config", default=None, help="Path to config.json (default: same directory as checkpoint)")
    parser.add_argument("--data", default=None, help="Path to a features parquet directory (default: synthetic validation)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckdir = os.path.dirname(args.checkpoint)
    cfg_path = args.config or os.path.join(ckdir, "config.json")
    cfg = json.load(open(cfg_path))

    syn = SyntheticData(window=cfg.get("window", "3s"), device=args.device, data_dir=args.data)
    prepared = syn.prepare(cfg)

    net = M.build(cfg, len(prepared["features"])).to(args.device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    net.eval()

    from .train import evaluate as eval_fn, val_score
    with torch.no_grad():
        cat, fire, dead = eval_fn(net, prepared["Xva"], prepared["labels"], list(prepared["labels"].keys()))
    v = val_score(cat, prepared["labels"], list(prepared["labels"].keys()))
    print(f"synthetic validation: {v:.4f}")
    print(f"firing rates: {fire}")
    print(f"dead units: {dead}")
