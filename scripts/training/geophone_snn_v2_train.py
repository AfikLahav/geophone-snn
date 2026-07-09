# %% [markdown]
# # Geophone feature-SNN — v2 training
# Feature-input PLIF SNN (SpikingJelly), independent per-class heads (human/animal ordinal,
# vehicle binary presence; nothing implicit). Trained on synthetic v2, early-stopped on a
# profile-held-out synthetic split, tested once on the real recordings. See TRAINING_PLAN.md.
#
# Set env `GEO_SMOKE=1` for a fast 1-shard / few-hundred-step correctness run.

# %% Cell 1 — config & repro
import os, sys, json, time, glob, sqlite3, warnings, copy
import numpy as np
# SpikingJelly's cupy kernels reference the deprecated np.int/np.float/np.bool aliases
# (removed in numpy >=1.24). Restore them before any cupy kernel runs.
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as Fnn
warnings.filterwarnings("ignore")

SMOKE = os.environ.get("GEO_SMOKE", "0") == "1"
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = r"S:/ALL PROJECTS/geophone sensor/finals project/finals project"
SIMGEO = os.path.join(ROOT, "simgeo"); sys.path.insert(0, SIMGEO)
FEAT_DIR = os.environ.get("GEO_FEAT_DIR", r"G:/geophone_synth/features_v2")   # GEO_FEAT_DIR=features_v3 for v3
SQLITE = os.path.join(ROOT, "feature_analysis_v2.sqlite")
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
OUT = os.environ.get("GEO_OUT") or os.path.join(ROOT, "snn_132_out" if os.environ.get("GEO_FEATURES", "") == "full"
                   else "snn_v2_out")
os.makedirs(OUT, exist_ok=True)
PNG = os.path.join(OUT, "plots"); os.makedirs(PNG, exist_ok=True)

T = 4
BATCH = 4096
EVAL_EVERY = 500
MAX_STEPS = 40_000
PATIENCE = 12                 # eval-intervals without EMA-val AUROC improvement
LR = 5e-4
WARMUP = 1000
EMA_DECAY = 0.999
DROPOUT = 0.1
WD = 1e-4
WIDTHS = [512, 256, 128]
CLIPZ = 8.0
if SMOKE:
    MAX_STEPS, EVAL_EVERY, WARMUP, PATIENCE = 600, 100, 50, 4

# v3.1 supervision flags (GENERATION_PLAN_V4.md §2). All default OFF -> v2 behavior unchanged.
GATED_EVAL = os.environ.get("GEO_GATED_EVAL", "0") == "1"      # gated val AUROC + selection
ORDINAL_MASK = os.environ.get("GEO_ORDINAL_MASK", "0") == "1"  # mask multiplicity loss below tau_lo
BAL_SAMPLER = os.environ.get("GEO_BALANCED_SAMPLER", "0") == "1"  # class-x-zone stratified batches
GATES = None
if GATED_EVAL or ORDINAL_MASK or BAL_SAMPLER:
    GATES = json.load(open(os.path.join(ROOT, "gap_study", "v4_plan", "gates.json")))["classes"]
    print("gates:", {c: (v["tau_lo_db"], v["tau_hi_db"]) for c, v in GATES.items()},
          f"| gated_eval={GATED_EVAL} ordinal_mask={ORDINAL_MASK} balanced={BAL_SAMPLER}")

# feature set: default = 104 (discriminative & is_rep from the analysis sqlite);
# GEO_FEATURES=full -> the full 132-feature bank (features.py FEATURE_NAMES).
import features as Fx
if os.environ.get("GEO_FEATURES", "") == "full":
    FEATURES = list(Fx.FEATURE_NAMES)
else:
    _con = sqlite3.connect(SQLITE)
    FEATURES = [r[0] for r in _con.execute(
        "SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]
    _con.close()
assert all(f in Fx.FEATURE_NAMES for f in FEATURES), "feature name mismatch vs features.py"
NF = len(FEATURES)
print(f"device={DEV} smoke={SMOKE} | {NF} features | T={T} batch={BATCH} max_steps={MAX_STEPS}")

# %% Cell 2 — data load (GPU-resident), scaler, per-head labels
LVL = {"none": 0, "single": 1, "multiple": 2}
META = ["split", "profile_id", "family", "subkind", "coarse", "terrain_vs", "common_snr",
        "scene_id", "t0", "human_level", "human_soft", "human_snr", "vehicle_level", "vehicle_soft",
        "vehicle_snr", "animal_level", "animal_soft", "animal_snr"]
import pandas as pd
shards = sorted(glob.glob(os.path.join(FEAT_DIR, "features_shard_*.parquet")))
if SMOKE: shards = shards[:1]
t0 = time.time()
df = pd.concat([pd.read_parquet(s, columns=FEATURES + META) for s in shards], ignore_index=True)
print(f"loaded {len(df):,} windows from {len(shards)} shard(s) in {(time.time()-t0):.0f}s")

is_tr = (df["split"] == "train").to_numpy()
Xall = df[FEATURES].to_numpy(np.float32)
mu = Xall[is_tr].mean(0); sd = Xall[is_tr].std(0) + 1e-8           # train-only scaler
Xz = np.clip((Xall - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
json.dump({"mean": mu.tolist(), "std": sd.tolist(), "clip": CLIPZ, "features": FEATURES},
          open(os.path.join(OUT, "scaler.json"), "w"))

def lvl_arr(col): return df[col].map(LVL).fillna(df[col]).to_numpy().astype(np.int64)
H_l, A_l = lvl_arr("human_level"), lvl_arr("animal_level")
V_l = lvl_arr("vehicle_level")
H_s = df["human_soft"].to_numpy(np.float32); A_s = df["animal_soft"].to_numpy(np.float32)
V_s = df["vehicle_soft"].to_numpy(np.float32)
H_n = df["human_snr"].to_numpy(np.float32); A_n = df["animal_snr"].to_numpy(np.float32)
V_n = df["vehicle_snr"].to_numpy(np.float32)

def to_dev(a): return torch.as_tensor(a).to(DEV)
Xtr, Xva = to_dev(Xz[is_tr]), to_dev(Xz[~is_tr])
L = {}                                                            # labels per split
for nm, lv, sf, sn in [("human", H_l, H_s, H_n), ("animal", A_l, A_s, A_n), ("vehicle", V_l, V_s, V_n)]:
    L[nm] = {"tr_lvl": to_dev(lv[is_tr]), "tr_soft": to_dev(sf[is_tr]), "tr_snr": to_dev(sn[is_tr]),
             "va_lvl": to_dev(lv[~is_tr]), "va_soft": to_dev(sf[~is_tr]), "va_snr": to_dev(sn[~is_tr])}
fam_va = df["family"].to_numpy()[~is_tr]; snr_va = df["common_snr"].to_numpy()[~is_tr]
coarse_va = df["coarse"].to_numpy()[~is_tr]
scene_va = df["scene_id"].to_numpy()[~is_tr]; t0_va = df["t0"].to_numpy()[~is_tr]
NTR, NVA = Xtr.shape[0], Xva.shape[0]
print(f"train {NTR:,} | val {NVA:,} | GPU tensor {Xtr.element_size()*Xtr.nelement()/1e9:.2f} GB")

# %% Cell 3 — model
from spikingjelly.activation_based import neuron, surrogate, layer, functional

class SeqBN(nn.Module):
    """BatchNorm over channels for multi-step [T,B,C] tensors (folds T into the batch)."""
    def __init__(self, c):
        super().__init__(); self.bn = nn.BatchNorm1d(c)
    def forward(self, x):
        T_, B_, C_ = x.shape
        return self.bn(x.reshape(T_ * B_, C_)).reshape(T_, B_, C_)


class FeatureSNN(nn.Module):
    def __init__(self, nf, widths, dropout):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(nf))                  # interpretability-only scale
        body = []
        d = nf
        for w in widths:
            body += [layer.Linear(d, w), SeqBN(w),                 # BN guards against dead neurons
                     neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(alpha=2.0),
                                              detach_reset=True, step_mode="m"),
                     layer.Dropout(dropout)]
            d = w
        self.body = nn.Sequential(*body)
        def head(out):
            return nn.Sequential(layer.Linear(d, out),
                                 neuron.LIFNode(v_threshold=float("inf"), surrogate_function=surrogate.ATan(),
                                                step_mode="m", store_v_seq=True, backend="torch"))
        self.human, self.animal, self.vehicle = head(2), head(2), head(1)

    def forward(self, x):                                         # x [B,nf]
        xs = (x * self.gate).unsqueeze(0).repeat(T, 1, 1)         # constant-current encode -> [T,B,nf]
        h = self.body(xs)
        out = {}
        for nm, hd in (("human", self.human), ("animal", self.animal), ("vehicle", self.vehicle)):
            hd(h); out[nm] = hd[-1].v_seq.mean(0)                 # mean-membrane over T -> [B,out]
        return out

net = FeatureSNN(NF, WIDTHS, DROPOUT).to(DEV)
functional.set_step_mode(net, "m")
functional.set_backend(net, "cupy", instance=neuron.ParametricLIFNode)   # PLIF on cupy; readout stays torch
assert all(m.backend == "cupy" for m in net.modules() if isinstance(m, neuron.ParametricLIFNode)), "cupy not engaged"
n_params = sum(p.numel() for p in net.parameters())
print(f"params: {n_params:,}")
ema = copy.deepcopy(net)                                          # EMA shadow
for p in ema.parameters(): p.requires_grad_(False)

# firing-rate / dead-neuron monitor via hooks on the PLIF layers
_plif = [m for m in net.modules() if isinstance(m, neuron.ParametricLIFNode)]
_fire = {}
def _mk_hook(i):
    def h(mod, inp, outp): _fire[i] = outp.detach()
    return h
for i, m in enumerate(_plif): m.register_forward_hook(_mk_hook(i))

# %% Cell 4 — train (step-based)
def head_loss(logits, lvl, soft, snr=None, tau_lo=None):
    ge1 = Fnn.binary_cross_entropy_with_logits(logits[:, 0], soft)      # soft presence target
    if logits.shape[1] == 2:                                           # ordinal: conditional >=multiple
        m = lvl >= 1
        if ORDINAL_MASK and snr is not None and tau_lo is not None:
            m = m & (snr >= tau_lo)                                    # no count-the-silence supervision
        if m.any():
            ge2 = Fnn.binary_cross_entropy_with_logits(logits[m, 1], (lvl[m] == 2).float())
            return ge1 + ge2
    return ge1

def lr_at(step):
    if step < WARMUP: return LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return 0.5 * LR * (1 + np.cos(np.pi * min(p, 1.0)))

opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)

@torch.no_grad()
def auroc(score, y):
    from sklearn.metrics import roc_auc_score
    y = y.cpu().numpy(); s = score.cpu().numpy()
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")

_last_ungated = {}                                                # side-channel: ungated AUROCs of the last evaluate()

@torch.no_grad()
def evaluate(model):
    global _last_ungated
    model.eval(); funcs = []
    aurocs, fr, dead = {}, [], []
    for j in range(0, NVA, BATCH):
        functional.reset_net(model)
        xb = Xva[j:j + BATCH]; o = model(xb)
        funcs.append({nm: o[nm].cpu() for nm in o})
    cat = {nm: torch.cat([f[nm] for f in funcs]) for nm in funcs[0]}
    _last_ungated = {}
    for nm in ("human", "animal", "vehicle"):
        present = (L[nm]["va_lvl"] > 0).long()
        if GATED_EVAL:
            # headline positives: present AND detectable (snr >= tau_hi); negatives: absent.
            # present-but-sub-tau_hi windows are dropped from the metric (marginal zone), NOT negatives.
            keep = (L[nm]["va_lvl"] == 0) | (L[nm]["va_snr"] >= GATES[nm]["tau_hi_db"])
            kc = keep.cpu()
            aurocs[nm] = auroc(cat[nm][kc, 0], present[keep])
            _last_ungated[nm] = auroc(cat[nm][:, 0], present)
        else:
            aurocs[nm] = auroc(cat[nm][:, 0], present)
    # firing-rate / dead-neuron on one val batch
    functional.reset_net(model); _ = model(Xva[:min(BATCH, NVA)])
    for i in sorted(_fire):
        o = _fire[i]; fr.append(float(o.mean())); dead.append(float((o.sum((0, 1)) == 0).float().mean()))
    model.train()
    return cat, aurocs, fr, dead

hist = {"step": [], "train_loss": [], "val_auroc_raw": [], "val_auroc_ema": [],
        "val_auroc_ema_ungated": [],
        "per_head_ema": [], "lr": [], "gradnorm": [], "fire": [], "dead": []}
best_ema, best_state_ema, best_state_raw, bad = -1, None, None, 0
g = torch.Generator(device=DEV); g.manual_seed(SEED)

if BAL_SAMPLER:
    # class-x-zone strata: detectable per class (present & snr >= tau_hi), any-marginal, nothing.
    # GEO_STRATA_MIX="wh,wv,wa,wmarg,wnoth" sets per-stratum batch weights (default equal).
    _det = {nm: ((L[nm]["tr_lvl"] > 0) & (L[nm]["tr_snr"] >= GATES[nm]["tau_hi_db"]))
            for nm in ("human", "animal", "vehicle")}
    _anyp = (L["human"]["tr_lvl"] > 0) | (L["animal"]["tr_lvl"] > 0) | (L["vehicle"]["tr_lvl"] > 0)
    _marg = _anyp & ~(_det["human"] | _det["animal"] | _det["vehicle"])
    _mix = [float(x) for x in os.environ.get("GEO_STRATA_MIX", "1,1,1,1,1").split(",")]
    STRATA, SWTS = [], []
    for _wi, (_snm, _sm) in enumerate([("human_det", _det["human"]), ("vehicle_det", _det["vehicle"]),
                                       ("animal_det", _det["animal"]), ("marginal", _marg), ("nothing", ~_anyp)]):
        _ix = torch.where(_sm)[0]
        print(f"  stratum {_snm}: {len(_ix):,} (w={_mix[_wi]:g})")
        if len(_ix) and _mix[_wi] > 0: STRATA.append(_ix); SWTS.append(_mix[_wi])
    _ks = [int(round(BATCH * w / sum(SWTS))) for w in SWTS]
    _ks[-1] += BATCH - sum(_ks)                                # exact batch size

    def draw_batch():
        parts = [s[torch.randint(0, len(s), (k,), generator=g, device=DEV)]
                 for s, k in zip(STRATA, _ks) if k > 0]
        return torch.cat(parts)
else:
    def draw_batch():
        return torch.randint(0, NTR, (BATCH,), generator=g, device=DEV)

def _tau_lo(nm): return GATES[nm]["tau_lo_db"] if GATES else None
net.train(); t0 = time.time()
for step in range(MAX_STEPS):
    for pg in opt.param_groups: pg["lr"] = lr_at(step)
    idx = draw_batch()
    xb = Xtr[idx]
    functional.reset_net(net)
    o = net(xb)
    loss = (head_loss(o["human"], L["human"]["tr_lvl"][idx], L["human"]["tr_soft"][idx],
                      L["human"]["tr_snr"][idx], _tau_lo("human"))
            + head_loss(o["animal"], L["animal"]["tr_lvl"][idx], L["animal"]["tr_soft"][idx],
                        L["animal"]["tr_snr"][idx], _tau_lo("animal"))
            + head_loss(o["vehicle"], L["vehicle"]["tr_lvl"][idx], L["vehicle"]["tr_soft"][idx],
                        L["vehicle"]["tr_snr"][idx], _tau_lo("vehicle")))
    opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    d = min(EMA_DECAY, (step + 1) / (step + 10))                  # EMA with warm-up
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(d).add_(pn, alpha=1 - d)
        for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
    if (step + 1) % EVAL_EVERY == 0 or step == 0:
        _, ar_raw, _, _ = evaluate(net)
        _, ar_ema, fr, dead = evaluate(ema)
        ug_ema = dict(_last_ungated)                              # ungated side-channel (GATED_EVAL only)
        m_raw = np.nanmean(list(ar_raw.values())); m_ema = np.nanmean(list(ar_ema.values()))
        m_ug = float(np.nanmean(list(ug_ema.values()))) if ug_ema else None
        hist["step"].append(step + 1); hist["train_loss"].append(float(loss))
        hist["val_auroc_raw"].append(m_raw); hist["val_auroc_ema"].append(m_ema)
        hist["val_auroc_ema_ungated"].append(m_ug)
        hist["per_head_ema"].append(ar_ema); hist["lr"].append(lr_at(step))
        hist["gradnorm"].append(float(gn)); hist["fire"].append(fr); hist["dead"].append(dead)
        print(f"step {step+1:6d} loss {float(loss):.3f} AUROC raw {m_raw:.4f} ema {m_ema:.4f}"
              + (f" (ungated {m_ug:.4f})" if m_ug is not None else "")
              + f" | fire {[round(x,2) for x in fr]} dead {[round(x,2) for x in dead]} "
              f"| {(time.time()-t0)/60:.1f}m", flush=True)
        if m_ema > best_ema + 1e-4:
            best_ema, bad = m_ema, 0
            best_state_ema = copy.deepcopy(ema.state_dict()); best_state_raw = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE: print(f"early stop @ {step+1} (best EMA AUROC {best_ema:.4f})"); break
torch.save(best_state_ema, os.path.join(OUT, "model_ema.pt"))
torch.save(best_state_raw, os.path.join(OUT, "model_raw.pt"))
json.dump({k: v for k, v in hist.items() if k not in ("per_head_ema",)},
          open(os.path.join(OUT, "history.json"), "w"))
print(f"DONE train in {(time.time()-t0)/60:.1f} min | best EMA mean-AUROC {best_ema:.4f}")

# %% Cell 5 — synthetic-val evaluation + core plots
ema.load_state_dict(best_state_ema); net.load_state_dict(best_state_raw)
cat_ema, ar_ema, fr, dead = evaluate(ema)
cat_raw, ar_raw, _, _ = evaluate(net)
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score, roc_auc_score)

def savefig(name): plt.tight_layout(); plt.savefig(os.path.join(PNG, name), dpi=120); plt.close()

# training curves
plt.figure(figsize=(8, 5))
plt.plot(hist["step"], hist["val_auroc_raw"], label="raw"); plt.plot(hist["step"], hist["val_auroc_ema"], label="EMA")
plt.xlabel("step"); plt.ylabel("mean per-head val AUROC"); plt.legend(); plt.title("Validation AUROC: raw vs EMA")
savefig("01_val_auroc_raw_vs_ema.png")
plt.figure(figsize=(8, 5)); plt.plot(hist["step"], hist["train_loss"]); plt.xlabel("step"); plt.ylabel("train loss")
plt.title("Training loss"); savefig("02_train_loss.png")
plt.figure(figsize=(8, 5)); plt.plot(hist["step"], hist["lr"]); plt.xlabel("step"); plt.ylabel("lr"); plt.title("LR schedule")
savefig("03_lr_schedule.png")
plt.figure(figsize=(8, 5)); plt.plot(hist["step"], hist["gradnorm"]); plt.xlabel("step"); plt.ylabel("grad norm")
plt.title("Gradient norm"); savefig("04_gradnorm.png")
fr_arr = np.array(hist["fire"]); dead_arr = np.array(hist["dead"])
plt.figure(figsize=(8, 5))
for i in range(fr_arr.shape[1]): plt.plot(hist["step"], fr_arr[:, i], label=f"PLIF{i}")
plt.xlabel("step"); plt.ylabel("mean firing rate"); plt.legend(); plt.title("Per-layer firing rate"); savefig("05_firing_rate.png")
plt.figure(figsize=(8, 5))
for i in range(dead_arr.shape[1]): plt.plot(hist["step"], dead_arr[:, i], label=f"PLIF{i}")
plt.xlabel("step"); plt.ylabel("dead-neuron fraction"); plt.legend(); plt.title("Per-layer dead-neuron %"); savefig("06_dead_neurons.png")

# per-head confusion (EMA), ROC, PR, calibration
HEADS = {"human": (cat_ema["human"], L["human"]["va_lvl"], 3),
         "animal": (cat_ema["animal"], L["animal"]["va_lvl"], 3),
         "vehicle": (cat_ema["vehicle"], L["vehicle"]["va_lvl"], 2)}
for nm, (logit, lvl, k) in HEADS.items():
    lvl = lvl.cpu().numpy()
    if k == 3:
        p1 = torch.sigmoid(logit[:, 0]).cpu().numpy(); p2 = torch.sigmoid(logit[:, 1]).cpu().numpy()
        pred = (p1 > 0.5).astype(int) + (p2 > 0.5).astype(int)
    else:
        p1 = torch.sigmoid(logit[:, 0]).cpu().numpy(); pred = (p1 > 0.5).astype(int); lvl = (lvl > 0).astype(int)
    cm = confusion_matrix(lvl, pred)
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm): plt.text(b, a, int(v), ha="center", va="center")
    plt.title(f"{nm} confusion (EMA)"); plt.xlabel("pred"); plt.ylabel("true"); plt.colorbar(); savefig(f"10_cm_{nm}.png")
    y = (lvl > 0).astype(int) if k == 3 else lvl; s = p1
    fpr, tpr, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--", lw=.5)
    plt.title(f"{nm} ROC (AUC {roc_auc_score(y,s):.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR"); savefig(f"11_roc_{nm}.png")
    plt.figure(figsize=(5, 4)); plt.plot(rc, pr); plt.title(f"{nm} PR (AP {average_precision_score(y,s):.3f})")
    plt.xlabel("recall"); plt.ylabel("precision"); savefig(f"12_pr_{nm}.png")
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
    conf = [s[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    acc = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    plt.figure(figsize=(5, 4)); plt.plot([0, 1], [0, 1], "k--", lw=.5); plt.plot(conf, acc, "o-")
    plt.title(f"{nm} reliability"); plt.xlabel("confidence"); plt.ylabel("accuracy"); savefig(f"13_calib_{nm}.png")

# per-terrain-family + accuracy-vs-SNR (presence, OR-of-heads proxy via human as example set)
present_true = ((L["human"]["va_lvl"] > 0) | (L["animal"]["va_lvl"] > 0) | (L["vehicle"]["va_lvl"] > 0)).cpu().numpy()
score_any = torch.maximum(torch.maximum(torch.sigmoid(cat_ema["human"][:, 0]), torch.sigmoid(cat_ema["animal"][:, 0])),
                          torch.sigmoid(cat_ema["vehicle"][:, 0])).cpu().numpy()
fams = sorted(set(fam_va)); fam_auc = []
for f in fams:
    m = fam_va == f
    fam_auc.append(roc_auc_score(present_true[m], score_any[m]) if len(np.unique(present_true[m])) == 2 else np.nan)
plt.figure(figsize=(9, 4)); plt.bar(range(len(fams)), fam_auc); plt.xticks(range(len(fams)), fams, rotation=60, ha="right")
plt.ylabel("any-class AUROC"); plt.title("Per-terrain-family detection (shortcut check)"); savefig("20_per_family.png")

# PCA + correlation + gate weights
from sklearn.decomposition import PCA
sub = np.random.default_rng(0).choice(NVA, min(20000, NVA), replace=False)
pc = PCA(n_components=10).fit(Xz[~is_tr][sub])
proj = pc.transform(Xz[~is_tr][sub])
plt.figure(figsize=(6, 5)); sc = plt.scatter(proj[:, 0], proj[:, 1], c=present_true[sub], s=3, cmap="coolwarm", alpha=.4)
plt.title("PCA of features (val) — colored present/nothing"); plt.xlabel("PC1"); plt.ylabel("PC2"); savefig("30_pca_scatter.png")
plt.figure(figsize=(6, 4)); plt.plot(np.cumsum(pc.explained_variance_ratio_), "o-"); plt.xlabel("components")
plt.ylabel("cum. explained var"); plt.title("PCA scree"); savefig("31_pca_scree.png")
C = np.corrcoef(Xz[~is_tr][sub].T)
plt.figure(figsize=(8, 7)); plt.imshow(C, cmap="RdBu", vmin=-1, vmax=1); plt.colorbar(); plt.title("Feature correlation (104)")
savefig("32_corr_heatmap.png")
gate = net.gate.detach().cpu().numpy(); order = np.argsort(-np.abs(gate))
plt.figure(figsize=(10, 4)); plt.bar(range(NF), gate[order]); plt.title("Learned FeatureGate weights")
plt.xticks([]); plt.ylabel("gate"); savefig("33_feature_gate.png")
json.dump({"per_head_auroc_ema": ar_ema, "per_head_auroc_raw": ar_raw,
           "firing_rate": fr, "dead_frac": dead, "gate_top": [FEATURES[i] for i in order[:20]]},
          open(os.path.join(OUT, "val_metrics.json"), "w"), indent=1)
print("synthetic-val metrics:", json.dumps(ar_ema))

# %% Cell 6 — deployment event-FAR (contiguous nothing stream, OR-of-heads, hysteresis + N-of-M)
def hysteresis(p, entry, exit_):
    out = np.zeros(len(p), bool); on = False
    for i, v in enumerate(p):
        on = v >= entry if not on else v > exit_
        out[i] = on
    return out

# build a temporally-ordered val stream (proxy: order val windows; nothing vs any-present)
order = np.lexsort((t0_va, scene_va))       # contiguous per-scene temporal stream (reviewer C4)
p_any = score_any[order]; truth = present_true[order]
far_curve = []
for entry in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    det = hysteresis(p_any, entry, entry * 0.4)
    # N-of-M persistence (3-of-5)
    M, N = 5, 3
    persist = np.array([det[max(0, i - M + 1):i + 1].sum() >= N for i in range(len(det))])
    tp = (persist & (truth == 1)).sum(); fp = (persist & (truth == 0)).sum()
    recall = tp / max(1, (truth == 1).sum()); far = fp / max(1, (truth == 0).sum())
    far_curve.append((entry, recall, far))
fc = np.array(far_curve)
plt.figure(figsize=(6, 5)); plt.plot(fc[:, 1], fc[:, 2], "o-")
for e, r, f in far_curve: plt.annotate(f"{e}", (r, f))
plt.xlabel("recall (any-class)"); plt.ylabel("window false-alarm rate"); plt.title("Event-FAR vs recall (OR-of-heads, 3-of-5)")
savefig("40_event_far_recall.png")
json.dump([{"entry": e, "recall": r, "far": f} for e, r, f in far_curve],
          open(os.path.join(OUT, "far_curve.json"), "w"), indent=1)
print("FAR curve:", far_curve)

# %% Cell 7 — REAL test (run once; never tunes). Re-featurize CSVs identically, confound-aware.
import features as F
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
SCENE = 30 * int(F.FS)                                  # re-chunk to ~synthetic scene length
def featurize_real(path):
    a = pd.read_csv(path)["amplitude"].to_numpy(np.float32); feats = []
    for c0 in range(0, len(a), SCENE):                  # chunk so impulse/envelope stats match synthetic
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, 1500):
            feats.append(F.window_features(pre, i0).astype(np.float32))
    return np.stack(feats) if feats else np.empty((0, F.NFEAT), np.float32)

real_X, real_y = [], []
fidx = [Fx.FEATURE_NAMES.index(f) for f in FEATURES]
for fn, cls in REALS.items():
    Xr = featurize_real(os.path.join(REAL_DIR, fn))
    Xr = np.clip((Xr[:, fidx] - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
    real_X.append(Xr); real_y += [cls] * len(Xr)
    print(f"  real {fn}: {len(Xr)} windows")
real_X = np.vstack(real_X); real_y = np.array(real_y)
ema.eval()
with torch.no_grad():
    functional.reset_net(ema); ro = ema(to_dev(real_X))
ph = torch.sigmoid(ro["human"][:, 0]).cpu().numpy(); pv = torch.sigmoid(ro["vehicle"][:, 0]).cpu().numpy()
# (a) FAR on the two nothing files; (b) human-vs-vehicle; (c) domain-classifier AUC
noth = real_y == "nothing"
far_real = {"human_head_FA_rate_on_nothing": float((ph[noth] > 0.5).mean()),
            "vehicle_head_FA_rate_on_nothing": float((pv[noth] > 0.5).mean())}
hv = np.isin(real_y, ["human", "vehicle"])
hv_auc = float(roc_auc_score((real_y[hv] == "vehicle").astype(int), (pv - ph)[hv]))
# domain classifier: synthetic-val vs real on the 104 features
from sklearn.linear_model import LogisticRegression
ns = min(len(real_X), 20000)
Xs = Xz[~is_tr][np.random.default_rng(1).choice(NVA, ns, replace=False)]
Xd = np.vstack([Xs, real_X]); yd = np.r_[np.zeros(len(Xs)), np.ones(len(real_X))]
dom_auc = float(roc_auc_score(yd, LogisticRegression(max_iter=200).fit(Xd, yd).predict_proba(Xd)[:, 1]))
real_report = {"FAR_on_nothing": far_real, "human_vs_vehicle_AUC": hv_auc,
               "domain_classifier_AUC": dom_auc,
               "note": "1 session/class -> indicative, confound-aware; see REAL_FEATURE_ANALYSIS.md"}
json.dump(real_report, open(os.path.join(OUT, "real_test.json"), "w"), indent=1)
print("REAL TEST:", json.dumps(real_report, indent=1))
plt.figure(figsize=(5, 4)); plt.hist(ph[noth], bins=30, alpha=.6, label="human-head on nothing")
plt.hist(pv[noth], bins=30, alpha=.6, label="vehicle-head on nothing"); plt.legend()
plt.title(f"Real false-alarm distribution (domain-AUC {dom_auc:.2f})"); plt.xlabel("p(present)"); savefig("50_real_far.png")

# %% Cell 8 — summary card
summary = {"params": int(n_params), "features": NF, "T": T, "best_ema_mean_auroc": float(best_ema),
           "per_head_auroc_ema": ar_ema, "firing_rate": fr, "dead_frac": dead,
           "real_test": real_report}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
plt.figure(figsize=(8, 5)); plt.axis("off")
plt.text(0.02, 0.98, "Geophone feature-SNN v2 — summary\n\n" + json.dumps(summary, indent=1)[:1500],
         va="top", family="monospace", fontsize=8)
savefig("99_summary_card.png")
print("ALL DONE ->", OUT)
