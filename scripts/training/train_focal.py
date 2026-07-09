"""Retrain variant: focal loss + low-SNR oversampling (emphasize hard/faint present windows),
using the EXISTING corpus (no regeneration). Same architecture/schedule as the main run; saves
model_focal_ema.pt and compares real human-recall vs the baseline run. Run: python train_focal.py
"""
import os, sys, json, time, glob, sqlite3, copy
import numpy as np
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
torch.manual_seed(0); np.random.seed(0); DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIR = r"G:/geophone_synth/features_v2"; OUT = os.path.join(HERE, "snn_v2_out")
REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
T, BATCH, EVAL_EVERY, MAX_STEPS, PATIENCE = 4, 4096, 500, 40000, 12
LR, WARMUP, EMA_DECAY, DROPOUT, WD, CLIPZ = 5e-4, 1000, 0.999, 0.1, 1e-4, 8.0
GAMMA = 2.0; FAINT_DB = 12.0; W_HUMAN, W_OTHER = 5.0, 2.0; SCALE_REAL = 25.4
con = sqlite3.connect(os.path.join(HERE, "feature_analysis_v2.sqlite"))
FEATURES = [r[0] for r in con.execute("SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]; con.close()
NF = len(FEATURES); fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]
LVL = {"none": 0, "single": 1, "multiple": 2}

# data
cols = FEATURES + ["split", "human_level", "human_snr", "human_soft", "vehicle_level", "vehicle_snr",
                   "vehicle_soft", "animal_level", "animal_snr", "animal_soft"]
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))], ignore_index=True)
tr = (df["split"] == "train").to_numpy()
Xall = df[FEATURES].to_numpy(np.float32); mu = Xall[tr].mean(0); sd = Xall[tr].std(0) + 1e-8
Xz = np.clip((Xall - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
json.dump({"mean": mu.tolist(), "std": sd.tolist(), "clip": CLIPZ, "features": FEATURES}, open(os.path.join(OUT, "scaler_focal.json"), "w"))
def lvc(c): return df[c].map(LVL).fillna(df[c]).to_numpy().astype(int)
Hl, Vl, Al = lvc("human_level"), lvc("vehicle_level"), lvc("animal_level")
Hs, Vs, As = (df[c].to_numpy(np.float32) for c in ("human_soft", "vehicle_soft", "animal_soft"))
Hn, Vn, An = (df[c].to_numpy(float) for c in ("human_snr", "vehicle_snr", "animal_snr"))
to = lambda a: torch.as_tensor(a).to(DEV)
Xtr, Xva = to(Xz[tr]), to(Xz[~tr])
L = {"human": (to(Hl[tr]), to(Hs[tr]), to(Hl[~tr])), "vehicle": (to(Vl[tr]), to(Vs[tr]), to(Vl[~tr])),
     "animal": (to(Al[tr]), to(As[tr]), to(Al[~tr]))}
NTR = Xtr.shape[0]
# sampling weights: emphasize faint present (human most)
w = np.ones(tr.sum(), np.float32)
w[(Hl[tr] > 0) & (Hn[tr] < FAINT_DB)] *= W_HUMAN
w[(Vl[tr] > 0) & (Vn[tr] < FAINT_DB)] *= W_OTHER
w[(Al[tr] > 0) & (An[tr] < FAINT_DB)] *= W_OTHER
W = to(w)
print(f"{NF} feats | train {NTR:,} | faint-human-present oversampled x{W_HUMAN} ({int(((Hl[tr]>0)&(Hn[tr]<FAINT_DB)).sum()):,} windows)")

class SeqBN(nn.Module):
    def __init__(s, c): super().__init__(); s.bn = nn.BatchNorm1d(c)
    def forward(s, x): T_, B_, C_ = x.shape; return s.bn(x.reshape(T_ * B_, C_)).reshape(T_, B_, C_)
class FeatureSNN(nn.Module):
    def __init__(s, nf, wdt=(512, 256, 128), dp=DROPOUT):
        super().__init__(); s.gate = nn.Parameter(torch.ones(nf)); b = []; d = nf
        for ww in wdt:
            b += [layer.Linear(d, ww), SeqBN(ww), neuron.ParametricLIFNode(init_tau=2.0,
                  surrogate_function=surrogate.ATan(2.0), detach_reset=True, step_mode="m"), layer.Dropout(dp)]; d = ww
        s.body = nn.Sequential(*b)
        def h(o): return nn.Sequential(layer.Linear(d, o), neuron.LIFNode(v_threshold=float("inf"),
                  surrogate_function=surrogate.ATan(), step_mode="m", store_v_seq=True, backend="torch"))
        s.human, s.animal, s.vehicle = h(2), h(2), h(1)
    def forward(s, x):
        xs = (x * s.gate).unsqueeze(0).repeat(T, 1, 1); hh = s.body(xs); o = {}
        for nm, hd in (("human", s.human), ("animal", s.animal), ("vehicle", s.vehicle)): hd(hh); o[nm] = hd[-1].v_seq[-1]
        return o
net = FeatureSNN(NF).to(DEV); functional.set_step_mode(net, "m")
functional.set_backend(net, "cupy", instance=neuron.ParametricLIFNode)
ema = copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]

def focal(logit, target, gamma=GAMMA):
    p = torch.sigmoid(logit); wt = (target - p).abs().pow(gamma)
    return (wt * Fnn.binary_cross_entropy_with_logits(logit, target, reduction="none")).mean()
def head_loss(lg, lvl, soft):
    l = focal(lg[:, 0], soft)
    if lg.shape[1] == 2:
        msk = lvl >= 1
        if msk.any(): l = l + focal(lg[msk, 1], (lvl[msk] == 2).float())
    return l
def lr_at(s): return LR * (s + 1) / WARMUP if s < WARMUP else 0.5 * LR * (1 + np.cos(np.pi * min((s - WARMUP) / (MAX_STEPS - WARMUP), 1)))
from sklearn.metrics import roc_auc_score
@torch.no_grad()
def val_auroc(model):
    model.eval(); ph = {"human": [], "vehicle": [], "animal": []}
    for j in range(0, Xva.shape[0], 8192):
        functional.reset_net(model); o = model(Xva[j:j + 8192])
        for nm in ph: ph[nm].append(torch.sigmoid(o[nm][:, 0]).cpu())
    model.train(); a = []
    for nm in ph:
        s = torch.cat(ph[nm]).numpy(); y = (L[nm][2] > 0).cpu().numpy()
        a.append(roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan)
    return float(np.nanmean(a))

opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD); g = torch.Generator(device=DEV); g.manual_seed(0)
best, bad, best_state = -1, 0, None; net.train(); t0 = time.time()
for step in range(MAX_STEPS):
    for pg in opt.param_groups: pg["lr"] = lr_at(step)
    idx = torch.multinomial(W, BATCH, replacement=True, generator=g)
    functional.reset_net(net); o = net(Xtr[idx])
    loss = sum(head_loss(o[nm], L[nm][0][idx], L[nm][1][idx]) for nm in ("human", "vehicle", "animal"))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    d = min(EMA_DECAY, (step + 1) / (step + 10))
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(d).add_(pn, alpha=1 - d)
        for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
    if (step + 1) % EVAL_EVERY == 0:
        au = val_auroc(ema)
        print(f"step {step+1:6d} loss {float(loss):.3f} ema-AUROC {au:.4f} {(time.time()-t0)/60:.1f}m", flush=True)
        if au > best + 1e-4: best, bad, best_state = au, 0, copy.deepcopy(ema.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE: print(f"early stop @ {step+1} best {best:.4f}"); break
torch.save(best_state, os.path.join(OUT, "model_focal_ema.pt")); ema.load_state_dict(best_state)
print(f"DONE {(time.time()-t0)/60:.1f}m best ema-AUROC {best:.4f}")

# ---- compare on REAL (floor-scaled), per-head syn-cal thresholds ----
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score
@torch.no_grad()
def probs(model, X):
    o = {"human": [], "vehicle": []}
    for j in range(0, len(X), 8192):
        functional.reset_net(model); r = model(to(X[j:j + 8192]))
        for nm in o: o[nm].append(torch.sigmoid(r[nm][:, 0]).cpu())
    return {k: np.array(torch.cat(v)) for k, v in o.items()}
def bestthr(p, y): grid = np.linspace(0.05, 0.95, 19); return float(grid[np.argmax([balanced_accuracy_score(y, p > t) for t in grid])])
vp = probs(ema, Xz[~tr]); TH = {"human": bestthr(vp["human"], Hl[~tr] > 0), "vehicle": bestthr(vp["vehicle"], Vl[~tr] > 0)}
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
def feat_real(p, sc):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * sc; fe = []
    for c0 in range(0, len(a), 30 * int(F.FS)):
        seg = a[c0:c0 + 30 * int(F.FS)]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, 1500): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)
rX, ry = [], []
for fn, cls in REALS.items():
    Xr = feat_real(os.path.join(REAL_DIR, fn), SCALE_REAL)
    rX.append(np.clip((Xr[:, fidx] - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)); ry += [cls] * len(rX[-1])
rX = np.vstack(rX); ry = np.array(ry); pr = probs(ema, rX)
mh, mv = pr["human"] - TH["human"], pr["vehicle"] - TH["vehicle"]
pred = np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
out = {"thresholds": TH, "real_acc": float(accuracy_score(ry, pred)),
       "recall": {c: float(recall_score(ry == c, pred == c, zero_division=0)) for c in ("human", "vehicle", "nothing")},
       "human_missed": int(((ry == "human") & (pred == "nothing")).sum()),
       "best_ema_val_auroc": float(best)}
json.dump(out, open(os.path.join(OUT, "FOCAL_RESULT.json"), "w"), indent=1)
print("\n=== FOCAL/oversampled model on REAL (x25 scale) ===")
print(f"  acc {out['real_acc']:.3f} | human-rec {out['recall']['human']:.3f} | veh-rec {out['recall']['vehicle']:.3f} | "
      f"noth-rec {out['recall']['nothing']:.3f} | human-missed {out['human_missed']}")
print("  baseline (main run, x25): acc 0.900 | human-rec 0.914 | veh-rec 0.848 | noth-rec 0.916 | human-missed 63")
print("wrote FOCAL_RESULT.json")
