"""Score models that already finished, on a dataset that was not ready when they ran.

This exists because preparing a dataset and training a model are independent work on different
hardware -- one is processor-bound, the other uses the graphics card -- and waiting for the first
before starting the second wastes hours. So the campaign starts on whatever is cached, and a
dataset that arrives later is applied afterwards to the saved weights.

Scoring is seconds per model. Training is minutes. Nothing is retrained.

    python -m campaign.rescore savanna
    python -m campaign.rescore savanna --only A1_baseline B6_separate
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from spikingjelly.activation_based import functional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simgeo", "simgeo_v42"))

import features as F                                                        # noqa: E402
from training import data, db, model as M, evaluate as score
from evaluation import configs, prepare_real

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))


class _Scaler:
    """Rebuilds a saved scaler from its stored form, so a model is scored exactly as trained."""

    def __init__(self, blob):
        self.kind = blob.get("kind", "standard")
        if self.kind == "rank":
            self.edges = np.asarray(blob["edges"], np.float32)
            self.n_q = blob["n_q"]
        else:
            self.mu = np.asarray(blob["mean"], np.float32)
            self.sd = np.asarray(blob["std"], np.float32)
            self.clip = blob.get("clip", 8.0)

    def transform(self, X):
        if self.kind == "rank":
            out = np.empty_like(X, dtype=np.float32)
            for j in range(X.shape[1]):
                out[:, j] = np.searchsorted(self.edges[:, j], X[:, j])
            out /= (self.n_q - 1)
            return (out * 2.0 - 1.0).astype(np.float32)
        return np.clip((X - self.mu) / self.sd, -self.clip, self.clip).astype(np.float32)


def rescore(dataset, only=None, device="cuda", verbose=True):
    con = db.connect(os.path.join(OUT, "campaign.sqlite"))
    got = prepare_real.cached(dataset)
    if got is None:
        raise SystemExit(f"{dataset} is not prepared yet -- run: python -m campaign.prepare_real {dataset}")
    X, rows, info = got
    db.add_windows(con, dataset, rows, info)

    runs = [r for r in db.runs_table(con) if r["status"] == "ok"]
    if only:
        runs = [r for r in runs if r["run_id"] in set(only)]
    if verbose:
        print(f"{dataset}: {len(X):,} windows, {len(runs)} models to score")

    y_true = np.array([r["label"] for r in rows], dtype=object)
    classes = sorted(set(y_true.tolist()))
    done = 0
    for r in runs:
        rid = r["run_id"]
        d = os.path.join(OUT, "models", rid)
        wp, sp = os.path.join(d, "model_ema.pt"), os.path.join(d, "scaler.json")
        if not (os.path.exists(wp) and os.path.exists(sp)):
            continue
        blob = json.load(open(sp, encoding="utf-8"))
        cfg = json.loads(r["config_json"])
        names = blob["features"]
        idx = [F.FEATURE_NAMES.index(n) for n in names]
        sc = _Scaler(blob)

        net = M.build(cfg, len(names))
        functional.set_step_mode(net, "m")
        net.load_state_dict(torch.load(wp, map_location="cpu"))
        net = net.to(device).eval()

        s = score.model_scores(net, sc.transform(X[:, idx]), device=device)
        if "fourway" in s:
            s = score.fourway_to_heads(s["fourway"])
        order = ["human", "animal", "vehicle"]
        db.put_scores(con, rid, dataset,
                      np.stack([s.get(h, np.zeros(len(X), np.float32)) for h in order], 1),
                      heads=order)

        for op, far in score.FAR_POINTS.items():
            tau = {h: db.threshold_at_far(con, rid, h, far) for h in order}
            tau = {h: t for h, t in tau.items() if t is not None}
            if not tau:
                continue
            pred = score.decide(s, tau, heads=tuple(h for h in order if h in s and h in tau))
            mm = score.metrics_for(y_true, pred, [c for c in classes if c != "nothing"])
            db.put_metric(con, rid, dataset, op, "overall",
                          accuracy=mm["overall"]["accuracy"], n=mm["overall"]["n"],
                          extra={"taus": {k: float(v) for k, v in tau.items()},
                                 "scored_after_the_fact": True})
            for c in classes:
                if c == "nothing":
                    nn_ = int((y_true == "nothing").sum())
                    db.put_metric(con, rid, dataset, op, "nothing", n=nn_,
                                  recall=float(((pred == "nothing") & (y_true == "nothing")).sum() / nn_)
                                  if nn_ else None)
                elif c in mm:
                    db.put_metric(con, rid, dataset, op, c, n=mm[c]["n"], recall=mm[c]["recall"],
                                  precision=mm[c]["precision"], f1=mm[c]["f1"])
        for h in order:
            if h in s and h in set(classes):
                a = score.auc_for(y_true, s[h], h)
                if a is not None:
                    db.put_metric(con, rid, dataset, "threshold_free", h, auc=a,
                                  extra={"diagnostic_only": True})
        done += 1
        if verbose:
            acc = {op: round(float((score.decide(
                s, {h: db.threshold_at_far(con, rid, h, far) for h in order if
                    db.threshold_at_far(con, rid, h, far) is not None}) == y_true).mean()), 4)
                for op, far in score.FAR_POINTS.items()}
            print(f"  {rid:<24} {acc}", flush=True)
        del net
        torch.cuda.empty_cache()
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    n = rescore(a.dataset, only=a.only, device=a.device)
    print(f"scored {n} models on {a.dataset} in {time.time() - t0:.0f}s")
