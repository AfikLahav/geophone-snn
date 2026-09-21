"""Train one configured SNN or conventional comparison model."""
import copy
import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from spikingjelly.activation_based import functional, neuron

from . import model as M


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def head_loss(logits, lvl, soft, kind="conditional"):
    """Presence against a soft target, plus a more-than-one term only where something is present.

    The soft target is the sigmoid of signal-to-noise over six, so a faint window is not labelled
    with full confidence -- that is what keeps the marginal band from being trained as if it were
    obvious."""
    if logits.shape[1] == 4:                       # the mutually-exclusive control
        return Fnn.cross_entropy(logits, lvl)
    loss = Fnn.binary_cross_entropy_with_logits(logits[:, 0], soft)
    if logits.shape[1] == 2:
        m = lvl >= 1
        if m.any():
            tgt = (lvl[m] >= 2).float()
            if kind == "corn":
                # the published conditional form: the second task is trained only on the subset
                # where the first is already true, which keeps the two ranks monotone by
                # construction instead of by penalty
                loss = loss + Fnn.binary_cross_entropy_with_logits(logits[m, 1], tgt)
            else:
                loss = loss + Fnn.binary_cross_entropy_with_logits(logits[m, 1], tgt)
    return loss


def evaluate(net, Xva, labels, heads, batch=8192, sub=None):
    """Score the validation split. Returns per-head logits and the health numbers.

    `sub` evaluates a fixed subset instead of all 331,559 windows. The periodic check during
    training only needs a stable ranking signal, and running the full split every 500 steps was
    costing about half the wall clock of every run. The final evaluation always uses everything."""
    net.eval()
    outs = {h: [] for h in heads}
    fire, dead = [], []
    hooks, buf = [], {}
    nodes = [m for m in net.modules() if isinstance(m, neuron.BaseNode)
             and getattr(m, "v_threshold", 1.0) != float("inf")]
    for i, nd in enumerate(nodes):
        hooks.append(nd.register_forward_hook(
            lambda mod, inp, out, i=i: buf.__setitem__(i, out.detach())))
    X = Xva if sub is None else Xva[sub]
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            functional.reset_net(net)
            o = net(X[i:i + batch])
            for h in heads:
                outs[h].append(o[h].float())
    for i in sorted(buf):
        o = buf[i]
        fire.append(float(o.mean()))
        dead.append(float((o.sum(tuple(range(o.ndim - 1))) == 0).float().mean()))
    for h in hooks:
        h.remove()
    net.train()
    return {h: torch.cat(outs[h]) for h in heads}, fire, dead


def val_score(cat, labels, heads, sub=None):
    """Model-selection number: mean presence separation on synthetic validation.

    Selection never reads real data, so every field number stays untouched
    by which model is kept."""
    from sklearn.metrics import roc_auc_score
    vals = []
    for h in heads:
        if h == "fourway":
            y = labels[h]["va_lvl"].cpu().numpy()
            y = y if sub is None else y[sub.cpu().numpy()]
            p = cat[h].softmax(-1).cpu().numpy()
            vals.append(float((p.argmax(1) == y).mean()))
            continue
        yv = labels[h]["va_lvl"].cpu().numpy()
        y = ((yv if sub is None else yv[sub.cpu().numpy()]) > 0).astype(int)
        s = cat[h][:, 0].cpu().numpy()
        if y.min() == y.max():
            continue
        vals.append(float(roc_auc_score(y, s)))
    return float(np.mean(vals)) if vals else 0.0


def train(cfg, prepared, device="cuda", verbose=True, max_steps=None):
    """Train one configuration. Returns the model, its history, and the health numbers.

    Raises RuntimeError if the loss stops being a real number -- the loop turns that into a
    failed row rather than a result."""
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    Xtr, Xva = prepared["Xtr"], prepared["Xva"]
    labels, heads = prepared["labels"], prepared["heads"]
    n_in = Xtr.shape[-1]

    net = M.build(cfg, n_in).to(device)
    functional.set_step_mode(net, "m")
    if cfg.get("kind", "snn") == "snn" and device == "cuda":
        try:
            functional.set_backend(net, "cupy", instance=neuron.ParametricLIFNode)
        except Exception:
            pass                                    # falls back to the plain path, recorded below
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)

    steps = int(max_steps or cfg.get("max_steps", 40000))
    batch = int(cfg.get("batch", 4096))
    base_lr = float(cfg.get("lr", 5e-4))
    warm = int(cfg.get("warmup", 1000))
    opt = torch.optim.AdamW(net.parameters(), lr=base_lr,
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    decay = float(cfg.get("ema_decay", 0.999))
    eval_every = int(cfg.get("eval_every", 500))
    # a fixed budget: no early stopping, the schedule runs to its end
    patience = int(cfg.get("patience", 12)) if cfg.get("early_stop", True) else 10 ** 9
    ordinal = cfg.get("ordinal_loss", "conditional")
    per_step = bool(cfg.get("loss_per_step", False))
    fire_pen = float(cfg.get("fire_penalty", 0.0) or 0.0)
    fire_target = float(cfg.get("fire_target", 0.1))

    if cfg.get("head_weighting") == "learned":
        log_sigma = nn.Parameter(torch.zeros(len(heads), device=device))
        opt.add_param_group({"params": [log_sigma], "weight_decay": 0.0})
    else:
        log_sigma = None

    # firing rates are needed inside the loss when the penalty is on, so they are captured live
    # rather than measured afterwards
    rate_buf, rate_hooks = {}, []
    if fire_pen:
        for i, nd in enumerate(m for m in net.modules()
                               if isinstance(m, neuron.BaseNode)
                               and getattr(m, "v_threshold", 1.0) != float("inf")):
            rate_hooks.append(nd.register_forward_hook(
                lambda mod, inp, out, i=i: rate_buf.__setitem__(i, out)))

    # a fixed subsample for the periodic check -- same windows every evaluation and every run,
    # so the selection signal stays comparable while costing a fraction of the time
    n_sub = int(cfg.get("val_subsample", 60000))
    if Xva.shape[0] > n_sub:
        gsub = torch.Generator(device="cpu").manual_seed(0)
        val_sub = torch.randperm(Xva.shape[0], generator=gsub)[:n_sub].to(Xva.device)
    else:
        val_sub = None

    hist = {"step": [], "loss": [], "val": [], "lr": [], "grad": [], "fire": [], "dead": []}
    best, best_state, bad, ntr = -1.0, None, 0, Xtr.shape[0]
    g = torch.Generator(device=device).manual_seed(int(cfg.get("seed", 0)))
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    for step in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = cosine_lr(step, steps, base_lr, warm)
        idx = torch.randint(0, ntr, (batch,), generator=g, device=device)
        functional.reset_net(net)
        out = net(Xtr[idx])

        parts = []
        for hi, h in enumerate(heads):
            lab = labels[h]
            lg = out[h]
            if h == "fourway":
                parts.append(Fnn.cross_entropy(lg, lab["tr_lvl"][idx]))
            else:
                parts.append(head_loss(lg, lab["tr_lvl"][idx], lab["tr_soft"][idx], ordinal))
        if log_sigma is not None:
            loss = sum(p / (2 * torch.exp(log_sigma[i])) + log_sigma[i] / 2
                       for i, p in enumerate(parts))
        else:
            loss = sum(parts)
        loss = loss + net.gate_penalty()

        if fire_pen and rate_buf:
            # two-sided: penalize a layer that has gone silent as hard as one that never stops,
            # penalise both silent and saturated layers
            loss = loss + fire_pen * sum((r.mean() - fire_target) ** 2 for r in rate_buf.values())

        if not torch.isfinite(loss):
            raise RuntimeError(f"loss stopped being a real number at step {step}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(decay).add_(pn.detach(), alpha=1 - decay)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            cat, fire, dead = evaluate(ema, Xva, labels, heads, sub=val_sub)
            v = val_score(cat, labels, heads, sub=val_sub)
            hist["step"].append(step + 1)
            hist["loss"].append(float(loss.detach()))
            hist["val"].append(v)
            hist["lr"].append(opt.param_groups[0]["lr"])
            hist["grad"].append(float(gn))
            hist["fire"].append(fire)
            hist["dead"].append(dead)
            if verbose:
                print(f"    step {step+1:>6} loss {float(loss.detach()):.4f} val {v:.4f} "
                      f"fire {[round(x,2) for x in fire]}", flush=True)
            if v > best + 1e-5:
                best, bad = v, 0
                best_state = copy.deepcopy(ema.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break

    for h in rate_hooks:
        h.remove()
    # "best" restores the checkpoint with the best synthetic-validation score;
    # "last" keeps the final weights, for fixed-budget runs where synthetic
    # validation barely predicts the field result
    if best_state is not None and cfg.get("select", "best") == "best":
        ema.load_state_dict(best_state)
    cat, fire, dead = evaluate(ema, Xva, labels, heads)
    health = dict(
        best_val=float(best),
        steps=int(hist["step"][-1]) if hist["step"] else 0,
        wall_min=(time.time() - t0) / 60.0,
        peak_mib=(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else 0.0,
        fire=fire, dead=dead, tau=net.learned_tau(),
        params=int(sum(p.numel() for p in net.parameters())),
    )
    return ema, hist, health, cat


# ---------------------------------------------------------------- CLI entry point
if __name__ == "__main__":
    import argparse, yaml, json, torch
    from . import model as M
    from .data import SyntheticData

    parser = argparse.ArgumentParser(description="Train a geophone SNN detector.")
    parser.add_argument("--preset", required=True, help="Name from presets.yaml (tiny, medium, large, ...)")
    parser.add_argument("--window", default="3s", help="Feature-table window directory suffix (default: 3s)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: training/out/<preset>)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    presets = yaml.safe_load(open(os.path.join(here, "presets.yaml")))
    defaults = presets["defaults"]
    if args.preset not in presets["presets"]:
        parser.error(f"unknown preset '{args.preset}'. available: {list(presets['presets'].keys())}")
    cfg = {**defaults, **presets["presets"][args.preset], "window": args.window}

    outdir = args.outdir or os.path.join(here, "out", args.preset)
    os.makedirs(outdir, exist_ok=True)

    print(f"preset: {args.preset}  |  config: {json.dumps(cfg, default=str)}")
    syn = SyntheticData(window=cfg.get("window", "3s"), device=args.device)
    prepared = syn.prepare(cfg)

    ema, hist, health, cat = train(cfg, prepared, device=args.device)

    # save
    torch.save(ema.state_dict(), os.path.join(outdir, "model_ema.pt"))
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg, **health}, f, indent=1, default=str)
    with open(os.path.join(outdir, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"features": prepared["features"], **prepared["scaler"].to_json()}, f, indent=1)
    with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=1)
    print(f"saved to {outdir}  |  val {health['best_val']:.4f}  |  {health['params']} params  |  {health['wall_min']:.1f} min")
