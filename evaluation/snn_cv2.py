"""Five-fold field training of the 4.8k model, with a rehearsal start and forgetting measured.

Three starting points, three replicates each, two fold schemes (see snn_cv.py):
  pretrained on synthetic          -- the synthetic weights, fine-tuned on field windows only
  pretrained with rehearsal        -- the synthetic weights, every batch half field windows and
                                      half synthetic training windows, the same loss on both
  from scratch                     -- a fresh network, field windows only
For every fold the synthetic validation score (mean presence separation over the two heads on
a fixed 60,000-window subsample, the campaign's model-selection number) is recorded before and
after training. The difference is the forgetting. Rows go to `within_dataset`; the pooled row's
per-class column carries recalls, separations, and the synthetic score before/after.

    python -m campaign.snn_cv2
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
from training import data, db, train as train_one
from evaluation import rescore, snn_cv
from training import model as M

OUT = os.environ.get("GEO_CAMPAIGN_OUT", os.path.join(ROOT, "campaign_out"))
MODELS = os.path.join(OUT, "models")
BASES = [f"U2_lowrank16_rep{i}" for i in range(1, 4)]
PLANS = {
    "pretrained on synthetic": dict(steps=1500, lr=1e-4, batch=256, rehearse=False),
    "pretrained with rehearsal": dict(steps=1500, lr=1e-4, batch=256, rehearse=True),
    "from scratch": dict(steps=4000, lr=1e-3, batch=256, rehearse=False),
}


def train(net, X, lab, plan, syn_prep, device="cuda", seed=0):
    Xtr = torch.as_tensor(X, device=device)
    lvl = {h: torch.as_tensor(lab[h], device=device) for h in lab}
    soft = {h: lvl[h].float() for h in lab}
    heads = list(lab)
    opt = torch.optim.AdamW(net.parameters(), lr=plan["lr"], weight_decay=1e-4)
    g = torch.Generator(device=device).manual_seed(seed)
    b_field = plan["batch"] // 2 if plan["rehearse"] else plan["batch"]
    Xs, Ls = syn_prep["Xtr"], syn_prep["labels"]
    net.train()
    for step in range(plan["steps"]):
        for grp in opt.param_groups:
            grp["lr"] = train_one.cosine_lr(step, plan["steps"], plan["lr"], 100)
        idx = torch.randint(0, Xtr.shape[0], (b_field,), generator=g, device=device)
        functional.reset_net(net)
        out = net(Xtr[idx])
        loss = sum(train_one.head_loss(out[h], lvl[h][idx], soft[h][idx]) for h in heads)
        if plan["rehearse"]:
            j = torch.randint(0, Xs.shape[0], (plan["batch"] - b_field,), generator=g, device=device)
            functional.reset_net(net)
            o2 = net(Xs[j])
            loss = loss + sum(train_one.head_loss(o2[h], Ls[h]["tr_lvl"][j], Ls[h]["tr_soft"][j]) for h in heads)
        loss = loss + net.gate_penalty()
        if not torch.isfinite(loss):
            raise RuntimeError("loss stopped being a real number")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()


def synth_score(net, prep, sub):
    cat, _, _ = train_one.evaluate(net, prep["Xva"], prep["labels"], prep["heads"], sub=sub)
    return float(train_one.val_score(cat, prep["labels"], prep["heads"], sub=sub))


def main():
    from sklearn.metrics import roc_auc_score
    con = db.connect(os.path.join(OUT, "campaign.sqlite"))
    W = db.dataset_windows(con, "elbit")
    y, rec, t0 = W["label"], W["recording"], W["t0_s"]
    Xraw = np.load(os.path.join(ROOT, "campaign", "real_cache", "elbit_X.npy"))
    blob = json.load(open(os.path.join(MODELS, BASES[0], "scaler.json"), encoding="utf-8"))
    names = blob["features"]; idx = [F.FEATURE_NAMES.index(n) for n in names]
    X = rescore._Scaler(blob).transform(Xraw[:, idx]).astype(np.float32)
    lab = {"human": (y == "human").astype(np.int64), "vehicle": (y == "vehicle").astype(np.int64)}
    cfg = json.loads(con.execute("SELECT config_json FROM runs WHERE run_id=?", (BASES[0],)).fetchone()[0])
    syn = data.SyntheticData(window="3s", device="cuda", verbose=True)
    prep = syn.prepare(cfg)
    gsub = torch.Generator(device="cpu").manual_seed(0)
    sub = torch.randperm(prep["Xva"].shape[0], generator=gsub)[:60000].cuda()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    splits = {"random 5-fold": [(np.setdiff1d(np.arange(len(y)), perm[f::5]), perm[f::5]) for f in range(5)],
              "contiguous 5-fold, purged": snn_cv.contiguous_folds(rec, t0)}
    t_all = time.time()
    for start, plan in PLANS.items():
        for r in range(3):
            model_name = f"4.8k SNN, {start}, replicate {r + 1}"
            for sname, folds in splits.items():
                pred_all = np.full(len(y), "nothing", dtype=object)
                ph, pv = np.zeros(len(y)), np.zeros(len(y))
                before, after = [], []
                for fi, (a, b) in enumerate(folds):
                    if start.startswith("pretrained"):
                        net = M.build(cfg, len(names)).cuda()
                        net.load_state_dict(torch.load(os.path.join(MODELS, BASES[r], "model_ema.pt"), map_location="cuda"))
                        seed = r
                    else:
                        torch.manual_seed(100 + r); net = M.build(cfg, len(names)).cuda(); seed = 100 + r
                    functional.set_step_mode(net, "m")
                    s0 = synth_score(net, prep, sub)
                    train(net, X[a], {h: v[a] for h, v in lab.items()}, plan, prep, seed=seed)
                    s1 = synth_score(net, prep, sub)
                    before.append(s0); after.append(s1)
                    p = snn_cv.predict(net, X[b])
                    ph[b], pv[b] = p["human"], p["vehicle"]
                    d = np.where(p["human"] > p["vehicle"], "human", "vehicle")
                    pred_all[b] = np.where(np.maximum(p["human"], p["vehicle"]) > 0.5, d, "nothing")
                    con.execute("INSERT OR REPLACE INTO within_dataset VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (model_name, "77 cepstral cut, 3 s windows, 50% overlap", "three-way", sname, fi,
                                 int(len(a)), int(len(b)), float((pred_all[b] == y[b]).mean()),
                                 json.dumps({"synthetic_val_before": s0, "synthetic_val_after": s1}),
                                 f"{plan['steps']} steps, lr {plan['lr']}, rehearsal {plan['rehearse']}"))
                    del net; torch.cuda.empty_cache()
                acc = float((pred_all == y).mean())
                per = {c: float(((pred_all == c) & (y == c)).sum() / (y == c).sum()) for c in ("human", "vehicle", "nothing")}
                per["sep_person_vs_quiet"] = float(roc_auc_score((y[y != "vehicle"] == "human").astype(int), ph[y != "vehicle"]))
                per["sep_vehicle_vs_quiet"] = float(roc_auc_score((y[y != "human"] == "vehicle").astype(int), pv[y != "human"]))
                per["synthetic_val_before"] = float(np.mean(before)); per["synthetic_val_after"] = float(np.mean(after))
                con.execute("INSERT OR REPLACE INTO within_dataset VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (model_name, "77 cepstral cut, 3 s windows, 50% overlap", "three-way", sname, -1,
                             None, int(len(y)), acc, json.dumps(per), "all folds pooled"))
                con.commit()
                print(f"  {model_name:<46} {sname:<28} accuracy {acc:.4f}  person R {per['human']:.3f}  vehicle R {per['vehicle']:.3f}  "
                      f"quiet R {per['nothing']:.3f}  synthetic val {per['synthetic_val_before']:.3f} -> {per['synthetic_val_after']:.3f}", flush=True)
    print(f"[snn_cv2] done in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
