"""
Optuna sweep over SNN model/training configurations, steered at the real-data 'nothing'
detection problem. Reuses the EXACT model/train/eval code from geophone_snn_v2_train.py and
plot_0977_stats.py (no new architecture invented), parameterized per trial.

Each trial:
  1. re-pretrains the feature-SNN on the synthetic v2 corpus (reduced step budget for speed),
  2. saves model_ema.pt + model_raw.pt to optuna_runs/trial_XX/,
  3. fine-tunes 5-fold on the real recordings (same protocol as the 0.982 baseline),
  4. scores per-class out-of-fold recall (human / vehicle / nothing).

Objective = macro per-class recall (unweighted mean of the 3 per-class recalls) -> directly
elevates the lagging 'nothing' class without ignoring human/vehicle.

Two knobs specifically target the user's hypotheses:
  * bg_w        : synthetic loss weight on clear-background windows (push the backbone to
                  reject background harder -> fewer real false alarms).
  * drop_pctile : drop the lowest-RMS present-class windows from the fine-tune TRAIN folds
                  (tests "fine-tuning contains garbage data": quiet, mislabeled-present windows).

Results stream to optuna_runs/results.jsonl (one line per trial, flushed immediately) and a
human-readable optuna_runs/results.md is rewritten after every trial.

Env knobs:
  GEO_OPTUNA_TRIALS (default 30)
  GEO_OPTUNA_SMOKE=1  -> 1 trial, tiny pretrain (~400 steps), ens=1 (end-to-end validation)
"""
import os, sys, json, glob, copy, time, gc, sqlite3, warnings
import numpy as np
# SpikingJelly cupy kernels reference removed np.int/np.float/np.bool aliases (numpy>=1.24)
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
import matplotlib; matplotlib.use("Agg")
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import optuna
warnings.filterwarnings("ignore")

ROOT = r"S:/ALL PROJECTS/geophone sensor/finals project/finals project"
SIMGEO = os.path.join(ROOT, "simgeo"); sys.path.insert(0, SIMGEO)
import features as F
FEAT_DIR = r"G:/geophone_synth/features_v2"
SQLITE = os.path.join(ROOT, "feature_analysis_v2.sqlite")
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
OUT = os.path.join(ROOT, "snn_v2_out")
RUNS = os.path.join(ROOT, "optuna_runs"); os.makedirs(RUNS, exist_ok=True)
RESJSONL = os.path.join(RUNS, "results.jsonl")
RESMD = os.path.join(RUNS, "results.md")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE = os.environ.get("GEO_OPTUNA_SMOKE", "0") == "1"
N_TRIALS = 1 if SMOKE else int(os.environ.get("GEO_OPTUNA_TRIALS", "30"))
TIME_BUDGET_MIN = float(os.environ.get("GEO_OPTUNA_MINUTES", "150"))   # wall-clock cap; "up to N_TRIALS"
# FULL pretrain schedule — identical to geophone_snn_v2_train.py (early-stops ~16-22k of the 40k cap)
PRE_MAX_STEPS = 400 if SMOKE else 40000
EVAL_EVERY = 200 if SMOKE else 500
PATIENCE = 2 if SMOKE else 12
BATCH = 4096
WARMUP = 100 if SMOKE else 1000
CLIPZ = 8.0
SCALE_REAL = 25.4                       # real->synthetic amplitude alignment (baseline)
SCENE = 30 * int(F.FS); HOP = 1500; K = 5
WIDTH_CHOICES = ["512-256-128", "384-192-96", "256-256-128", "512-384-256", "640-320-160", "448-224-112"]

# ============================== load synthetic corpus ONCE (GPU-resident) ==============================
print("loading synthetic corpus ...", flush=True)
con = sqlite3.connect(SQLITE)
FEATURES = [r[0] for r in con.execute(
    "SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]
con.close()
assert all(f in F.FEATURE_NAMES for f in FEATURES)
NF = len(FEATURES)
LVL = {"none": 0, "single": 1, "multiple": 2}
META = ["split", "human_level", "human_soft", "vehicle_level", "vehicle_soft", "animal_level", "animal_soft"]
shards = sorted(glob.glob(os.path.join(FEAT_DIR, "features_shard_*.parquet")))
if SMOKE: shards = shards[:1]
t0 = time.time()
df = pd.concat([pd.read_parquet(s, columns=FEATURES + META) for s in shards], ignore_index=True)
print(f"  {len(df):,} windows from {len(shards)} shards in {time.time()-t0:.0f}s", flush=True)
is_tr = (df["split"] == "train").to_numpy()
Xall = df[FEATURES].to_numpy(np.float32)
mu = Xall[is_tr].mean(0); sd = Xall[is_tr].std(0) + 1e-8
Xz = np.clip((Xall - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)

def lvl_arr(col): return df[col].map(LVL).fillna(df[col]).to_numpy().astype(np.int64)
def to_dev(a): return torch.as_tensor(a).to(DEV)
Xtr_t, Xva_t = to_dev(Xz[is_tr]), to_dev(Xz[~is_tr])
LAB = {}
for nm in ("human", "animal", "vehicle"):
    lv = lvl_arr(nm + "_level"); sf = df[nm + "_soft"].to_numpy(np.float32)
    LAB[nm] = {"tr_lvl": to_dev(lv[is_tr]), "tr_soft": to_dev(sf[is_tr]),
               "va_lvl": to_dev(lv[~is_tr]), "va_soft": to_dev(sf[~is_tr])}
NTR, NVA = Xtr_t.shape[0], Xva_t.shape[0]
print(f"  train {NTR:,} | val {NVA:,} | {Xtr_t.element_size()*Xtr_t.nelement()/1e9:.2f} GB on {DEV}", flush=True)

# ============================== featurize real ONCE (cache features + per-window RMS) ==============================
print("featurizing real recordings ...", flush=True)
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]
def feat_with_rms(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE_REAL
    fe, rms = [], []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP):
            fe.append(F.window_features(pre, i0).astype(np.float32))
            w = seg[i0:i0 + F.NW]; rms.append(float(np.sqrt(np.mean(w * w))))
    if not fe: return np.empty((0, F.NFEAT), np.float32), np.empty((0,), np.float32)
    return np.nan_to_num(np.stack(fe)), np.array(rms, np.float32)

rX, rY, rBLK, rSRC, rRMS = [], [], [], [], []
for fn, cls in REALS.items():
    Xr, rmsr = feat_with_rms(os.path.join(REAL_DIR, fn))
    Xr = Xr[:, fidx]; n = len(Xr)
    rX.append(Xr.astype(np.float32)); rY += [cls] * n
    rBLK.append(np.arange(n) * K // max(n, 1)); rSRC += [fn] * n; rRMS.append(rmsr)
    print(f"  {fn}: {n} windows", flush=True)
rX = np.vstack(rX); rY = np.array(rY); rBLK = np.concatenate(rBLK); rSRC = np.array(rSRC); rRMS = np.concatenate(rRMS)
yh = (rY == "human").astype(np.float32); yv = (rY == "vehicle").astype(np.float32)
present_mask = (yh > 0) | (yv > 0)

# ============================== model (verbatim from geophone_snn_v2_train.py, T parameterized) ==============================
class SeqBN(nn.Module):
    def __init__(s, c): super().__init__(); s.bn = nn.BatchNorm1d(c)
    def forward(s, x):
        T_, B_, C_ = x.shape; return s.bn(x.reshape(T_ * B_, C_)).reshape(T_, B_, C_)

class FeatureSNN(nn.Module):
    def __init__(s, nf, widths, dropout, Tsteps):
        super().__init__(); s.T = Tsteps
        s.gate = nn.Parameter(torch.ones(nf)); body = []; d = nf
        for w in widths:
            body += [layer.Linear(d, w), SeqBN(w),
                     neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(alpha=2.0),
                                              detach_reset=True, step_mode="m"),
                     layer.Dropout(dropout)]; d = w
        s.body = nn.Sequential(*body)
        def head(o): return nn.Sequential(layer.Linear(d, o),
                     neuron.LIFNode(v_threshold=float("inf"), surrogate_function=surrogate.ATan(),
                                    step_mode="m", store_v_seq=True, backend="torch"))
        s.human, s.animal, s.vehicle = head(2), head(2), head(1)
    def forward(s, x):
        xs = (x * s.gate).unsqueeze(0).repeat(s.T, 1, 1); h = s.body(xs); o = {}
        for nm, hd in (("human", s.human), ("animal", s.animal), ("vehicle", s.vehicle)):
            hd(h); o[nm] = hd[-1].v_seq.mean(0)
        return o

def build(widths, dropout, Tsteps):
    net = FeatureSNN(NF, widths, dropout, Tsteps).to(DEV)
    functional.set_step_mode(net, "m")
    functional.set_backend(net, "cupy", instance=neuron.ParametricLIFNode)
    return net

def head_loss(logits, lvl, soft, bg_w):
    w = torch.where(soft < 0.2, torch.as_tensor(bg_w, device=soft.device), torch.ones((), device=soft.device))
    ge1 = Fnn.binary_cross_entropy_with_logits(logits[:, 0], soft, weight=w)
    if logits.shape[1] == 2:
        m = lvl >= 1
        if m.any():
            ge2 = Fnn.binary_cross_entropy_with_logits(logits[m, 1], (lvl[m] == 2).float())
            return ge1 + ge2
    return ge1

@torch.no_grad()
def synth_val_auroc(model):
    model.eval(); outs = []
    for j in range(0, NVA, BATCH):
        functional.reset_net(model); o = model(Xva_t[j:j + BATCH])
        outs.append({nm: o[nm].cpu() for nm in o})
    cat = {nm: torch.cat([f[nm] for f in outs]) for nm in outs[0]}
    ar = {}
    for nm in ("human", "animal", "vehicle"):
        present = (LAB[nm]["va_lvl"] > 0).long().cpu().numpy()
        s = cat[nm][:, 0].numpy()
        ar[nm] = float(roc_auc_score(present, s)) if len(np.unique(present)) == 2 else float("nan")
    model.train(); return ar

def pretrain(widths, dropout, Tsteps, lr, wd, ema_decay, bg_w, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = build(widths, dropout, Tsteps)
    ema = copy.deepcopy(net)
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    def lr_at(step):
        if step < WARMUP: return lr * (step + 1) / WARMUP
        p = (step - WARMUP) / max(1, PRE_MAX_STEPS - WARMUP)
        return 0.5 * lr * (1 + np.cos(np.pi * min(p, 1.0)))
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    best, best_ema_sd, best_raw_sd, bad = -1.0, None, None, 0
    net.train(); t0 = time.time()
    for step in range(PRE_MAX_STEPS):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        idx = torch.randint(0, NTR, (BATCH,), generator=g, device=DEV)
        functional.reset_net(net); o = net(Xtr_t[idx])
        loss = (head_loss(o["human"], LAB["human"]["tr_lvl"][idx], LAB["human"]["tr_soft"][idx], bg_w)
                + head_loss(o["animal"], LAB["animal"]["tr_lvl"][idx], LAB["animal"]["tr_soft"][idx], bg_w)
                + head_loss(o["vehicle"], LAB["vehicle"]["tr_lvl"][idx], LAB["vehicle"]["tr_soft"][idx], bg_w))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        d = min(ema_decay, (step + 1) / (step + 10))
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(d).add_(pn, alpha=1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            ar = synth_val_auroc(ema); m = float(np.nanmean(list(ar.values())))
            if m > best + 1e-4:
                best, bad = m, 0
                best_ema_sd = copy.deepcopy(ema.state_dict()); best_raw_sd = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= PATIENCE: break
    if best_ema_sd is None:                       # fallback (no eval improved)
        best_ema_sd = copy.deepcopy(ema.state_dict()); best_raw_sd = copy.deepcopy(net.state_dict())
    pre_min = (time.time() - t0) / 60
    last_ar = synth_val_auroc(ema)
    del net, ema, opt; gc.collect(); torch.cuda.empty_cache()
    return best_ema_sd, best_raw_sd, float(best), last_ar, pre_min

# ============================== fine-tune + OOF eval (verbatim protocol from plot_0977_stats.py) ==============================
def logits_of(model, Xz_, heads=("human", "vehicle")):
    model.eval(); acc = {h: [] for h in heads}
    with torch.no_grad():
        for j in range(0, len(Xz_), 8192):
            functional.reset_net(model); o = model(torch.as_tensor(Xz_[j:j + 8192]).to(DEV))
            for h in heads: acc[h].append(o[h].cpu())
    return {h: torch.cat(acc[h]) for h in heads}

def ft_train(model, Xz_, tr_idx, epochs, lr, seed):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xz_[tr_idx]).to(DEV)
    ht = torch.as_tensor(yh[tr_idx]).to(DEV); vt = torch.as_tensor(yv[tr_idx]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); model.train(); n = len(Xt)
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + np.cos(np.pi * ep / epochs))
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, 256):
            idx = perm[j:j + 256]; functional.reset_net(model); o = model(Xt[idx])
            loss = Fnn.binary_cross_entropy_with_logits(o["human"][:, 0], ht[idx]) \
                 + Fnn.binary_cross_entropy_with_logits(o["vehicle"][:, 0], vt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def bt(p, t):
    grid = np.linspace(0.05, 0.95, 19)
    return float(grid[np.argmax([balanced_accuracy_score(t, p > x) for x in grid])])
def decide(ph, pv, th, tv):
    mh, mv = ph - th, pv - tv
    return np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))

def eval_real(base_sd, widths, dropout, Tsteps, ens, ft_epochs, ft_lr, drop_pctile):
    ph_oof = np.zeros(len(rX)); pv_oof = np.zeros(len(rX)); fold_of = np.zeros(len(rX), int)
    for k in range(K):
        te = rBLK == k; tr = ~te; fold_of[te] = k
        # ---- garbage-hygiene: drop lowest-RMS present windows from the TRAIN fold only ----
        tr_idx = np.where(tr)[0]
        if drop_pctile > 0:
            pres = tr_idx[present_mask[tr_idx]]
            if len(pres) > 5:
                thr = np.percentile(rRMS[pres], drop_pctile)
                drop = pres[rRMS[pres] < thr]
                tr_idx = np.setdiff1d(tr_idx, drop, assume_unique=False)
        sc = StandardScaler().fit(rX[tr_idx]); Xzr = np.clip(sc.transform(rX), -CLIPZ, CLIPZ).astype(np.float32)
        base = build(widths, dropout, Tsteps); base.load_state_dict(base_sd)
        ah = np.zeros(te.sum()); av = np.zeros(te.sum())
        for s in range(ens):
            m = copy.deepcopy(base); m = ft_train(m, Xzr, tr_idx, ft_epochs, ft_lr, s)
            lo = logits_of(m, Xzr[te]); ah += torch.sigmoid(lo["human"][:, 0]).numpy(); av += torch.sigmoid(lo["vehicle"][:, 0]).numpy()
            del m
        ph_oof[te] = ah / ens; pv_oof[te] = av / ens
        del base; gc.collect(); torch.cuda.empty_cache()
    TH, TV = bt(ph_oof, yh > 0), bt(pv_oof, yv > 0)
    pred = decide(ph_oof, pv_oof, TH, TV)
    classes = ["human", "vehicle", "nothing"]
    rec = {c: float((pred[rY == c] == c).mean()) for c in classes}
    macro = float(np.mean([rec[c] for c in classes]))
    overall = float(accuracy_score(rY, pred))
    far = {"human_head": float((ph_oof[rY == "nothing"] > TH).mean()),
           "vehicle_head": float((pv_oof[rY == "nothing"] > TV).mean())}
    perfold = {c: [float((pred[(fold_of == k) & (rY == c)] == c).mean()) for k in range(K)] for c in classes}
    return dict(macro_recall=macro, overall_acc=overall, recall=rec, far=far,
                thr=dict(human=TH, vehicle=TV), perfold=perfold)

# ============================== Optuna objective ==============================
def write_md(study):
    rows = []
    for tr in study.trials:
        if tr.value is None: continue
        u = tr.user_attrs
        rows.append((tr.number, tr.value, u.get("overall_acc"), u.get("rec_human"), u.get("rec_vehicle"),
                     u.get("rec_nothing"), u.get("synth_auroc"), u.get("pre_min"), u.get("params")))
    rows_sorted = sorted(rows, key=lambda r: -r[1])
    with open(RESMD, "w", encoding="utf-8") as f:
        f.write("# Optuna sweep — model configs vs real 'nothing' detection\n\n")
        f.write("Objective = macro per-class recall (human/vehicle/nothing) on real 5-fold OOF. ")
        f.write("Full pretrain schedule (40k-step cap, early-stop patience 12) — same as the 0.982 model.\n\n")
        best = rows_sorted[0] if rows_sorted else None
        if best:
            f.write(f"**Best so far: trial {best[0]} — macro recall {best[1]:.4f}, nothing {best[5]:.4f}, "
                    f"overall {best[2]:.4f}**\n\n")
        f.write("| trial | macro_rec | overall | human | vehicle | nothing | synthAUROC | pre_min | params |\n")
        f.write("|--|--|--|--|--|--|--|--|--|\n")
        for r in rows_sorted:
            f.write(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | "
                    f"{r[6]:.4f} | {r[7]:.1f} | {r[8]} |\n")

def objective(trial):
    widths = tuple(int(x) for x in trial.suggest_categorical("widths", WIDTH_CHOICES).split("-"))
    dropout = trial.suggest_float("dropout", 0.0, 0.35)
    lr = trial.suggest_float("lr", 2.5e-4, 1.1e-3, log=True)
    wd = trial.suggest_float("wd", 1e-5, 3e-3, log=True)
    Tsteps = trial.suggest_categorical("T", [3, 4, 5])
    ema_decay = trial.suggest_categorical("ema_decay", [0.995, 0.999, 0.9995])
    bg_w = trial.suggest_categorical("bg_w", [1.0, 1.5, 2.0, 3.0])
    ft_epochs = trial.suggest_categorical("ft_epochs", [50, 80, 120])
    ft_lr = trial.suggest_categorical("ft_lr", [2e-4, 4e-4, 8e-4])
    ens = 1 if SMOKE else trial.suggest_categorical("ens", [3, 5])
    drop_pctile = trial.suggest_categorical("drop_pctile", [0, 10, 20, 30])

    t0 = time.time()
    ema_sd, raw_sd, synth_best, synth_ar, pre_min = pretrain(widths, dropout, Tsteps, lr, wd, ema_decay, bg_w)
    # save EMA + non-EMA backbone for this trial
    tdir = os.path.join(RUNS, f"trial_{trial.number:02d}"); os.makedirs(tdir, exist_ok=True)
    torch.save(ema_sd, os.path.join(tdir, "model_ema.pt"))
    torch.save(raw_sd, os.path.join(tdir, "model_raw.pt"))
    res = eval_real(ema_sd, widths, dropout, Tsteps, ens, ft_epochs, ft_lr, drop_pctile)
    dt = (time.time() - t0) / 60

    params = dict(widths="-".join(map(str, widths)), dropout=round(dropout, 3), lr=lr, wd=wd, T=Tsteps,
                  ema_decay=ema_decay, bg_w=bg_w, ft_epochs=ft_epochs, ft_lr=ft_lr, ens=ens, drop_pctile=drop_pctile)
    trial.set_user_attr("overall_acc", res["overall_acc"])
    trial.set_user_attr("rec_human", res["recall"]["human"])
    trial.set_user_attr("rec_vehicle", res["recall"]["vehicle"])
    trial.set_user_attr("rec_nothing", res["recall"]["nothing"])
    trial.set_user_attr("synth_auroc", float(np.nanmean(list(synth_ar.values()))))
    trial.set_user_attr("pre_min", pre_min)
    trial.set_user_attr("params", json.dumps(params))
    rec = dict(trial=trial.number, macro_recall=res["macro_recall"], overall_acc=res["overall_acc"],
               recall=res["recall"], far=res["far"], thr=res["thr"], perfold_nothing=res["perfold"]["nothing"],
               synth_val_auroc=synth_ar, synth_best_auroc=synth_best, params=params,
               pretrain_min=round(pre_min, 2), trial_min=round(dt, 2), model_dir=tdir)
    with open(RESJSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n"); f.flush()
    json.dump(rec, open(os.path.join(tdir, "metrics.json"), "w"), indent=1)
    print(f"[trial {trial.number}] macro_rec={res['macro_recall']:.4f} overall={res['overall_acc']:.4f} "
          f"H={res['recall']['human']:.3f} V={res['recall']['vehicle']:.3f} N={res['recall']['nothing']:.3f} "
          f"synthAUROC={float(np.nanmean(list(synth_ar.values()))):.4f} | {dt:.1f}m | {params}", flush=True)
    write_md(trial.study)
    return res["macro_recall"]

if __name__ == "__main__":
    timeout = None if SMOKE else int(TIME_BUDGET_MIN * 60)
    print(f"=== Optuna sweep: up to {N_TRIALS} trials, full {PRE_MAX_STEPS}-step schedule, "
          f"time cap {TIME_BUDGET_MIN:.0f} min, smoke={SMOKE} ===", flush=True)
    study = optuna.create_study(direction="maximize", study_name="geo_snn_nothing",
                                sampler=optuna.samplers.TPESampler(seed=0))
    # anchor: current baseline config (trial 0)
    if not SMOKE:
        study.enqueue_trial(dict(widths="512-256-128", dropout=0.1, lr=5e-4, wd=1e-4, T=4,
                                 ema_decay=0.999, bg_w=1.0, ft_epochs=80, ft_lr=4e-4, ens=5, drop_pctile=0))
    study.optimize(objective, n_trials=N_TRIALS, timeout=timeout, catch=(Exception,))
    print("=== DONE ===", flush=True)
    bt_ = study.best_trial
    print(f"best trial {bt_.number}: macro_recall={bt_.value:.4f}\n  params={bt_.user_attrs.get('params')}", flush=True)
