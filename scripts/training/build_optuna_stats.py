"""
Regenerate the FULL statistics pack from the Optuna winner (trial 12) so the presentation
can use this model's numbers + images. Reuses optuna_sweep.py (data, model, train, eval).

Trial-12 config: widths 512-256-128, dropout 0.099, lr 3.83e-4, wd 8.34e-5, T=5,
ema 0.9995, bg_w 3.0, ft_epochs 120, ft_lr 4e-4, ens 5, drop_pctile 10.

Outputs -> model_optuna_stats/ : training curves, synthetic CM/ROC/acc-vs-SNR,
real confusion / per-head ROC / zero-shot-vs-FT, and numbers.json with every slide value.
"""
import os, json, copy, glob, time
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spikingjelly.activation_based import functional, neuron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, accuracy_score,
                             balanced_accuracy_score)

import optuna_sweep as S   # module-level: loads synth corpus + featurizes real, defines model/train/eval

OUTD = os.path.join(S.ROOT, "model_optuna_stats"); os.makedirs(OUTD, exist_ok=True)
def fig(name, figsize=None):
    plt.tight_layout(); plt.savefig(os.path.join(OUTD, name), dpi=130, bbox_inches="tight"); plt.close()

CFG = dict(widths=(512, 256, 128), dropout=0.099, lr=0.00038294663971600364, wd=8.341134823312294e-05,
           T=5, ema_decay=0.9995, bg_w=3.0, ft_epochs=120, ft_lr=4e-4, ens=5, drop_pctile=10)
NUM = {"config": {**CFG, "widths": "-".join(map(str, CFG["widths"]))}}

# ================= 1) pretrain trial-12 config WITH history (deterministic, seed 0) =================
print(">>> pretraining trial-12 config (full schedule, logging history) ...", flush=True)
def pretrain_logged():
    torch.manual_seed(0); np.random.seed(0)
    net = S.build(CFG["widths"], CFG["dropout"], CFG["T"]); ema = copy.deepcopy(net)
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    def lr_at(step):
        if step < S.WARMUP: return CFG["lr"] * (step + 1) / S.WARMUP
        p = (step - S.WARMUP) / max(1, S.PRE_MAX_STEPS - S.WARMUP)
        return 0.5 * CFG["lr"] * (1 + np.cos(np.pi * min(p, 1.0)))
    g = torch.Generator(device=S.DEV); g.manual_seed(0)
    hist = {"step": [], "loss": [], "auroc_raw": [], "auroc_ema": []}
    best, best_ema_sd, best_raw_sd, bad = -1.0, None, None, 0
    net.train(); t0 = time.time()
    for step in range(S.PRE_MAX_STEPS):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        idx = torch.randint(0, S.NTR, (S.BATCH,), generator=g, device=S.DEV)
        functional.reset_net(net); o = net(S.Xtr_t[idx])
        loss = (S.head_loss(o["human"], S.LAB["human"]["tr_lvl"][idx], S.LAB["human"]["tr_soft"][idx], CFG["bg_w"])
                + S.head_loss(o["animal"], S.LAB["animal"]["tr_lvl"][idx], S.LAB["animal"]["tr_soft"][idx], CFG["bg_w"])
                + S.head_loss(o["vehicle"], S.LAB["vehicle"]["tr_lvl"][idx], S.LAB["vehicle"]["tr_soft"][idx], CFG["bg_w"]))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        d = min(CFG["ema_decay"], (step + 1) / (step + 10))
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(d).add_(pn, alpha=1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if (step + 1) % S.EVAL_EVERY == 0 or step == 0:
            ar_r = S.synth_val_auroc(net); ar_e = S.synth_val_auroc(ema)
            mr = float(np.nanmean(list(ar_r.values()))); me = float(np.nanmean(list(ar_e.values())))
            hist["step"].append(step + 1); hist["loss"].append(float(loss))
            hist["auroc_raw"].append(mr); hist["auroc_ema"].append(me)
            if me > best + 1e-4:
                best, bad = me, 0
                best_ema_sd = copy.deepcopy(ema.state_dict()); best_raw_sd = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= S.PATIENCE: break
            print(f"  step {step+1:5d} loss {float(loss):.3f} AUROC ema {me:.4f} | {(time.time()-t0)/60:.1f}m", flush=True)
    return best_ema_sd, best_raw_sd, best, hist

ema_sd, raw_sd, synth_best, hist = pretrain_logged()
torch.save(ema_sd, os.path.join(OUTD, "model_ema.pt")); torch.save(raw_sd, os.path.join(OUTD, "model_raw.pt"))
json.dump(hist, open(os.path.join(OUTD, "history.json"), "w"))
NUM["synth_best_mean_auroc"] = round(synth_best, 4)

# training curves
plt.figure(figsize=(8, 5))
plt.plot(hist["step"], hist["auroc_raw"], label="raw"); plt.plot(hist["step"], hist["auroc_ema"], label="EMA")
plt.xlabel("training step"); plt.ylabel("mean per-head val AUROC"); plt.legend(); plt.title("Validation AUROC: raw vs EMA")
fig("24_train_auroc.png")
plt.figure(figsize=(8, 5)); plt.plot(hist["step"], hist["loss"]); plt.xlabel("training step"); plt.ylabel("training loss")
plt.title("Training loss"); fig("25_train_loss.png")

# ================= 2) synthetic-val plots from the EMA model =================
print(">>> synthetic-val evaluation/plots ...", flush=True)
ema = S.build(CFG["widths"], CFG["dropout"], CFG["T"]); ema.load_state_dict(ema_sd); ema.eval()
# gather per-head logits over val
outs = []
with torch.no_grad():
    for j in range(0, S.NVA, S.BATCH):
        functional.reset_net(ema); o = ema(S.Xva_t[j:j + S.BATCH])
        outs.append({nm: o[nm].cpu() for nm in o})
cat = {nm: torch.cat([f[nm] for f in outs]) for nm in outs[0]}
ar = {}
for nm in ("human", "animal", "vehicle"):
    present = (S.LAB[nm]["va_lvl"] > 0).long().cpu().numpy(); s = cat[nm][:, 0].numpy()
    ar[nm] = round(float(roc_auc_score(present, s)), 4)
NUM["synth_per_head_auroc"] = ar
# 4-class single-source accuracy (argmax over present-prob of the 3 heads vs nothing)
va_lvl = {nm: S.LAB[nm]["va_lvl"].cpu().numpy() for nm in ("human", "animal", "vehicle")}
ph_s = torch.sigmoid(cat["human"][:, 0]).numpy(); pa_s = torch.sigmoid(cat["animal"][:, 0]).numpy(); pv_s = torch.sigmoid(cat["vehicle"][:, 0]).numpy()
# single-source mask: exactly one class present at level==1 (approx 4-class set)
present_any = (va_lvl["human"] > 0).astype(int) + (va_lvl["animal"] > 0).astype(int) + (va_lvl["vehicle"] > 0).astype(int)
single = present_any <= 1
true4 = np.where(va_lvl["human"][single] > 0, 0, np.where(va_lvl["vehicle"][single] > 0, 1, np.where(va_lvl["animal"][single] > 0, 2, 3)))
P = np.stack([ph_s[single], pv_s[single], pa_s[single]], 1)
# nothing if all below 0.5 else argmax
pred4 = np.where(P.max(1) < 0.5, 3, P.argmax(1))
NUM["synth_4class_acc"] = round(float((pred4 == true4).mean()), 4)

# synthetic confusion (human 3-level, vehicle 2-level) + ROC
for nm, k in [("human", 3), ("vehicle", 2)]:
    logit = cat[nm]; lvl = va_lvl[nm]
    if k == 3:
        p1 = torch.sigmoid(logit[:, 0]).numpy(); p2 = torch.sigmoid(logit[:, 1]).numpy()
        pred = (p1 > 0.5).astype(int) + (p2 > 0.5).astype(int); tk = ["none", "single", "multi"]
        cm = confusion_matrix(lvl, pred, labels=[0, 1, 2])
    else:
        p1 = torch.sigmoid(logit[:, 0]).numpy(); pred = (p1 > 0.5).astype(int); lvl2 = (lvl > 0).astype(int)
        tk = ["none", "present"]; cm = confusion_matrix(lvl2, pred, labels=[0, 1])
    plt.figure(figsize=(4.6, 4)); plt.imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm): plt.text(b, a, int(v), ha="center", va="center",
                                                   color="white" if v > cm.max() / 2 else "black")
    plt.xticks(range(len(tk)), tk); plt.yticks(range(len(tk)), tk); plt.xlabel("predicted"); plt.ylabel("true")
    plt.title(f"Synthetic {nm} confusion"); plt.colorbar(); fig(f"14_synth_cm_{nm}.png")
    y = (lvl > 0).astype(int); fpr, tpr, _ = roc_curve(y, p1)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--", lw=.5)
    plt.title(f"Synthetic {nm} ROC (AUC {roc_auc_score(y,p1):.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR"); fig(f"15_synth_roc_{nm}.png")

# acc-vs-SNR (reload common_snr for val rows, in S's shard/concat order)
print(">>> acc-vs-SNR ...", flush=True)
snr_all = pd.concat([pd.read_parquet(s, columns=["common_snr"]) for s in S.shards], ignore_index=True)["common_snr"].to_numpy()
snr_va = snr_all[~S.is_tr]
present_true = (present_any > 0)
score_any = np.maximum(np.maximum(ph_s, pv_s), pa_s)
bins = np.arange(-30, 31, 3); cx, rate = [], []
for i in range(len(bins) - 1):
    m = (snr_va >= bins[i]) & (snr_va < bins[i + 1]) & present_true
    if m.sum() >= 30: cx.append((bins[i] + bins[i + 1]) / 2); rate.append(float((score_any[m] > 0.5).mean()))
plt.figure(figsize=(7, 4)); plt.plot(cx, rate, "o-"); plt.xlabel("per-window SNR (dB)"); plt.ylabel("detection rate")
plt.title("Synthetic detection rate vs SNR"); plt.grid(alpha=.3); fig("19_synth_acc_vs_snr.png")

# ================= 3) real 5-fold fine-tune (drop_10, ft120, ens5) -> plots + numbers =================
print(">>> real 5-fold fine-tune + plots ...", flush=True)
rX, rY, rBLK, rRMS, yh, yv = S.rX, S.rY, S.rBLK, S.rRMS, S.yh, S.yv
present_mask = S.present_mask; K = S.K; CLIPZ = S.CLIPZ
ph_oof = np.zeros(len(rX)); pv_oof = np.zeros(len(rX)); ph_zs = np.zeros(len(rX)); pv_zs = np.zeros(len(rX)); fold_of = np.zeros(len(rX), int)
for k in range(K):
    te = rBLK == k; tr = ~te; fold_of[te] = k
    tr_idx = np.where(tr)[0]
    pres = tr_idx[present_mask[tr_idx]]
    if CFG["drop_pctile"] > 0 and len(pres) > 5:
        thr = np.percentile(rRMS[pres], CFG["drop_pctile"]); tr_idx = np.setdiff1d(tr_idx, pres[rRMS[pres] < thr])
    sc = StandardScaler().fit(rX[tr_idx]); Xzr = np.clip(sc.transform(rX), -CLIPZ, CLIPZ).astype(np.float32)
    base = S.build(CFG["widths"], CFG["dropout"], CFG["T"]); base.load_state_dict(ema_sd)
    lo = S.logits_of(base, Xzr[te]); ph_zs[te] = torch.sigmoid(lo["human"][:, 0]).numpy(); pv_zs[te] = torch.sigmoid(lo["vehicle"][:, 0]).numpy()
    ah = np.zeros(te.sum()); av = np.zeros(te.sum())
    for s in range(CFG["ens"]):
        m = copy.deepcopy(base); m = S.ft_train(m, Xzr, tr_idx, CFG["ft_epochs"], CFG["ft_lr"], s)
        lo = S.logits_of(m, Xzr[te]); ah += torch.sigmoid(lo["human"][:, 0]).numpy(); av += torch.sigmoid(lo["vehicle"][:, 0]).numpy()
        del m
    ph_oof[te] = ah / CFG["ens"]; pv_oof[te] = av / CFG["ens"]
    del base; torch.cuda.empty_cache(); print(f"  fold {k} done", flush=True)

TH, TV = S.bt(ph_oof, yh > 0), S.bt(pv_oof, yv > 0)
pred = S.decide(ph_oof, pv_oof, TH, TV)
labs = ["human", "vehicle", "nothing"]
overall = float(accuracy_score(rY, pred)); NUM["real_overall_acc"] = round(overall, 4)
NUM["real_thr"] = {"human": TH, "vehicle": TV}
NUM["real_per_head_rocauc"] = {"human": round(float(roc_auc_score(rY == "human", ph_oof)), 4),
                               "vehicle": round(float(roc_auc_score(rY == "vehicle", pv_oof)), 4)}
hv = np.isin(rY, ["human", "vehicle"])
NUM["real_human_vs_vehicle_auc"] = round(float(roc_auc_score((rY[hv] == "vehicle").astype(int), (pv_oof - ph_oof)[hv])), 4)
cm = confusion_matrix(rY, pred, labels=labs)
NUM["real_confusion"] = cm.tolist()
noth = rY == "nothing"
fa_h = int(((pred == "human") & noth).sum()); fa_v = int(((pred == "vehicle") & noth).sum())
NUM["real_FAR_nothing"] = {"human_pct": round(fa_h / noth.sum() * 100, 2), "vehicle_pct": round(fa_v / noth.sum() * 100, 2),
                           "total_false_alarms": fa_h + fa_v, "n_nothing": int(noth.sum())}
# per-class recall + per-fold (for CV mean +/- std)
rec = {c: round(float((pred[rY == c] == c).mean()), 4) for c in labs}
NUM["real_recall"] = rec
perfold = {}
for c in labs:
    vals = np.array([float((pred[(fold_of == kk) & (rY == c)] == c).mean()) for kk in range(K)]) * 100
    perfold[c] = {"per_fold": [round(float(x), 2) for x in vals], "mean": round(float(vals.mean()), 2),
                  "std_sample": round(float(vals.std(ddof=1)), 2), "pooled_pct": round(rec[c] * 100, 2)}
NUM["real_perclass_cv"] = perfold
# per-fold overall (zero-shot vs fine-tuned)
ft_fold = [float(accuracy_score(rY[fold_of == k], pred[fold_of == k])) for k in range(K)]
zpred = S.decide(ph_zs, pv_zs, S.bt(ph_zs, yh > 0), S.bt(pv_zs, yv > 0))
zs_fold = [float(accuracy_score(rY[fold_of == k], zpred[fold_of == k])) for k in range(K)]
NUM["real_ft_fold_mean"] = round(float(np.mean(ft_fold)), 4)
NUM["real_zeroshot_acc"] = round(float(accuracy_score(rY, zpred)), 4)
NUM["real_zeroshot_fold_mean"] = round(float(np.mean(zs_fold)), 4)

# real confusion plot
plt.figure(figsize=(5, 4.2)); plt.imshow(cm, cmap="Blues")
for (a, b), v in np.ndenumerate(cm): plt.text(b, a, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
plt.xticks(range(3), labs); plt.yticks(range(3), labs); plt.xlabel("predicted"); plt.ylabel("true")
plt.title(f"Real 3-class (counts) acc {overall:.3f}"); plt.colorbar(); fig("01_real_confusion.png")
# real per-head ROC
for nm, p in [("human", ph_oof), ("vehicle", pv_oof)]:
    y = (rY == nm).astype(int); fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--", lw=.5)
    plt.title(f"Real {nm} ROC (AUC {roc_auc_score(y,p):.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR"); fig(f"04_real_roc_{nm}.png")
# zero-shot vs fine-tuned per fold
x = np.arange(K); w = 0.38
plt.figure(figsize=(6.5, 4.1))
plt.bar(x - w / 2, zs_fold, w, label=f"zero-shot (mean {np.mean(zs_fold):.3f})", color="#4C78A8")
plt.bar(x + w / 2, ft_fold, w, label=f"fine-tuned (mean {np.mean(ft_fold):.3f})", color="#F58518")
plt.xticks(x, [f"fold {k}" for k in range(K)]); plt.ylim(0.8, 1.0); plt.ylabel("real 3-class accuracy")
plt.title("Zero-shot vs fine-tuned (per fold)"); plt.legend(); fig("10_real_zeroshot_vs_ft.png")

# ================= 4) scratch-on-real baseline (same arch, no synthetic pretrain) =================
print(">>> scratch-on-real baseline (trial-12 arch) ...", flush=True)
sc_pred = np.empty(len(rX), dtype=object)
for k in range(K):
    te = rBLK == k; tr = ~te; tr_idx = np.where(tr)[0]
    sc = StandardScaler().fit(rX[tr_idx]); Xzr = np.clip(sc.transform(rX), -CLIPZ, CLIPZ).astype(np.float32)
    ah = np.zeros(te.sum()); av = np.zeros(te.sum())
    for s in range(CFG["ens"]):
        m = S.build(CFG["widths"], CFG["dropout"], CFG["T"])     # random init (no pretrain load)
        m = S.ft_train(m, Xzr, tr_idx, max(CFG["ft_epochs"], 200), CFG["ft_lr"], s)
        lo = S.logits_of(m, Xzr[te]); ah += torch.sigmoid(lo["human"][:, 0]).numpy(); av += torch.sigmoid(lo["vehicle"][:, 0]).numpy()
        del m
    ph_s_, pv_s_ = ah / CFG["ens"], av / CFG["ens"]
    th, tv = S.bt(ph_s_, (yh[te] > 0)), S.bt(pv_s_, (yv[te] > 0))
    sc_pred[te] = S.decide(ph_s_, pv_s_, th, tv)
    torch.cuda.empty_cache(); print(f"  scratch fold {k} done", flush=True)
NUM["scratch_on_real_acc"] = round(float(accuracy_score(rY, sc_pred)), 4)

json.dump(NUM, open(os.path.join(OUTD, "numbers.json"), "w"), indent=1)
print("\n================ NUMBERS ================")
print(json.dumps(NUM, indent=1))
print("\nplots + numbers ->", OUTD)
