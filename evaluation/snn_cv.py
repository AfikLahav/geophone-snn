"""The 4.8k spiking model trained on the field recordings themselves, five-fold.

Two starting points: the synthetic-pretrained weights (three replicates of U2_lowrank16) and
a fresh network with the same architecture (three seeds). Two ways of folding the 2,353
field windows:
  random 5-fold      -- windows shuffled; neighbours overlap by half, so every test window has
                        a near-copy in training
  contiguous 5-fold  -- each recording cut into five consecutive blocks; fold k tests block k of
                        every recording; training windows within 2 windows of a test block are
                        dropped so no sample is shared across the split
Features: the 77 cepstral-cut features through the synthetic scaler, as the model expects.
Decision on the held-out fold: strongest head above 0.5, otherwise nothing. Rows go to the
`within_dataset` table; separation per head is stored in the per-class column.

    python -m campaign.snn_cv
"""
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
import features as F                                                       # noqa: E402
from training import db, train as train_one
from evaluation import rescore
from training import model as M

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))
MODELS = os.path.join(OUT, "models")
BASES = [f"U2_lowrank16_rep{i}" for i in range(1, 4)]
FT = dict(steps=1500, lr=1e-4, batch=256)          # from the pretrained weights
SCRATCH = dict(steps=4000, lr=1e-3, batch=256)     # from a fresh network


def contiguous_folds(rec, t0, k=5, purge=2):
    folds = []
    order = {r: np.where(rec == r)[0][np.argsort(t0[rec == r])] for r in np.unique(rec)}
    for f in range(k):
        test = np.zeros(len(rec), bool); drop = np.zeros(len(rec), bool)
        for r, m in order.items():
            n = len(m); a, b = f * n // k, (f + 1) * n // k
            test[m[a:b]] = True
            drop[m[max(0, a - purge):a]] = True
            drop[m[b:min(n, b + purge)]] = True
        folds.append((np.where(~test & ~drop)[0], np.where(test)[0]))
    return folds


def train(net, X, lab, cfg, plan, device="cuda", seed=0):
    Xtr = torch.as_tensor(X, device=device)
    lvl = {h: torch.as_tensor(lab[h], device=device) for h in lab}
    soft = {h: lvl[h].float() for h in lab}
    opt = torch.optim.AdamW(net.parameters(), lr=plan["lr"], weight_decay=1e-4)
    g = torch.Generator(device=device).manual_seed(seed)
    net.train()
    for step in range(plan["steps"]):
        for grp in opt.param_groups:
            grp["lr"] = train_one.cosine_lr(step, plan["steps"], plan["lr"], 100)
        idx = torch.randint(0, Xtr.shape[0], (plan["batch"],), generator=g, device=device)
        functional.reset_net(net)
        out = net(Xtr[idx])
        loss = sum(train_one.head_loss(out[h], lvl[h][idx], soft[h][idx]) for h in lab) + net.gate_penalty()
        if not torch.isfinite(loss):
            raise RuntimeError("loss stopped being a real number")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()


def predict(net, X, device="cuda"):
    with torch.no_grad():
        functional.reset_net(net)
        o = net(torch.as_tensor(X, device=device))
    return {h: torch.sigmoid(v[:, 0]).cpu().numpy() for h, v in o.items()}


def main():
    from sklearn.metrics import roc_auc_score
    con = db.connect(os.path.join(OUT, "campaign.sqlite"))
    con.executescript("""CREATE TABLE IF NOT EXISTS within_dataset (
        model TEXT NOT NULL, features TEXT NOT NULL, task TEXT NOT NULL, split TEXT NOT NULL,
        fold INTEGER NOT NULL, n_train INTEGER, n_test INTEGER, accuracy REAL, per_class_json TEXT,
        note TEXT, PRIMARY KEY (model, features, task, split, fold));"""); con.commit()
    W = db.dataset_windows(con, "elbit")
    y, rec, t0 = W["label"], W["recording"], W["t0_s"]
    Xraw = np.load(os.path.join(ROOT, "campaign", "real_cache", "elbit_X.npy"))
    blob = json.load(open(os.path.join(MODELS, BASES[0], "scaler.json"), encoding="utf-8"))
    names = blob["features"]; idx = [F.FEATURE_NAMES.index(n) for n in names]
    X = rescore._Scaler(blob).transform(Xraw[:, idx]).astype(np.float32)
    lab = {"human": (y == "human").astype(np.int64), "vehicle": (y == "vehicle").astype(np.int64)}
    cfg = json.loads(con.execute("SELECT config_json FROM runs WHERE run_id=?", (BASES[0],)).fetchone()[0])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    random_folds = [(np.setdiff1d(np.arange(len(y)), perm[f::5]), perm[f::5]) for f in range(5)]
    splits = {"random 5-fold": random_folds, "contiguous 5-fold, purged": contiguous_folds(rec, t0)}
    t_all = time.time()
    for start in ("pretrained on synthetic", "from scratch"):
        for r in range(3):
            model_name = f"4.8k SNN, {start}, replicate {r + 1}"
            for sname, folds in splits.items():
                pred_all = np.full(len(y), "nothing", dtype=object)
                ph, pv = np.zeros(len(y)), np.zeros(len(y))
                for fi, (a, b) in enumerate(folds):
                    net = M.build(cfg, len(names)).cuda()
                    if start.startswith("pretrained"):
                        net.load_state_dict(torch.load(os.path.join(MODELS, BASES[r], "model_ema.pt"), map_location="cuda"))
                        plan, seed = FT, r
                    else:
                        torch.manual_seed(100 + r); net = M.build(cfg, len(names)).cuda()
                        plan, seed = SCRATCH, 100 + r
                    functional.set_step_mode(net, "m")
                    train(net, X[a], {h: v[a] for h, v in lab.items()}, cfg, plan, seed=seed)
                    p = predict(net, X[b])
                    ph[b], pv[b] = p["human"], p["vehicle"]
                    d = np.where(p["human"] > p["vehicle"], "human", "vehicle")
                    on = np.maximum(p["human"], p["vehicle"]) > 0.5
                    pred_all[b] = np.where(on, d, "nothing")
                    acc = float((pred_all[b] == y[b]).mean())
                    con.execute("INSERT OR REPLACE INTO within_dataset VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (model_name, "77 cepstral cut, 3 s windows, 50% overlap", "three-way", sname, fi,
                                 int(len(a)), int(len(b)), acc, None, f"{plan['steps']} steps, lr {plan['lr']}"))
                    del net; torch.cuda.empty_cache()
                acc = float((pred_all == y).mean())
                per = {c: float(((pred_all == c) & (y == c)).sum() / (y == c).sum()) for c in ("human", "vehicle", "nothing")}
                per["sep_person_vs_quiet"] = float(roc_auc_score((y[y != "vehicle"] == "human").astype(int), ph[y != "vehicle"]))
                per["sep_vehicle_vs_quiet"] = float(roc_auc_score((y[y != "human"] == "vehicle").astype(int), pv[y != "human"]))
                con.execute("INSERT OR REPLACE INTO within_dataset VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (model_name, "77 cepstral cut, 3 s windows, 50% overlap", "three-way", sname, -1,
                             None, int(len(y)), acc, json.dumps(per), "all folds pooled"))
                con.commit()
                print(f"  {model_name:<44} {sname:<28} accuracy {acc:.4f}  person R {per['human']:.3f}  vehicle R {per['vehicle']:.3f}  quiet R {per['nothing']:.3f}  sep {per['sep_person_vs_quiet']:.3f}/{per['sep_vehicle_vs_quiet']:.3f}", flush=True)
    print(f"[snn_cv] done in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
