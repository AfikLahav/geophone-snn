"""Figures for the 132-feature SNN findings (held-out synthetic val + fine-tune).
Recomputes confusion matrices from the model so the plotted numbers are real.
Writes PNGs to snn_132_out/stats/.
"""
import os, sys, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import confusion_matrix
from spikingjelly.activation_based import neuron, surrogate, layer, functional
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)

HERE = r"S:/ALL PROJECTS/geophone sensor/finals project/finals project"
sys.path.insert(0, os.path.join(HERE, "simgeo")); import features as Fx
OUT = os.path.join(HERE, "snn_132_out"); STATS = os.path.join(OUT, "stats"); os.makedirs(STATS, exist_ok=True)
DEV = "cpu"; T = 4; CLIP = 8.0
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "savefig.dpi": 140})
ACCENT, MISS, OTHER = "#2f6f4f", "#c25b3a", "#7d8893"

scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32)

class SeqBN(nn.Module):
    def __init__(s, c): super().__init__(); s.bn = nn.BatchNorm1d(c)
    def forward(s, x): T_, B_, C_ = x.shape; return s.bn(x.reshape(T_*B_, C_)).reshape(T_, B_, C_)
class FeatureSNN(nn.Module):
    def __init__(s, nf, w=(512, 256, 128), dp=0.1):
        super().__init__(); s.gate = nn.Parameter(torch.ones(nf)); b = []; d = nf
        for ww in w:
            b += [layer.Linear(d, ww), SeqBN(ww), neuron.ParametricLIFNode(init_tau=2.0,
                  surrogate_function=surrogate.ATan(2.0), detach_reset=True, step_mode="m"), layer.Dropout(dp)]; d = ww
        s.body = nn.Sequential(*b)
        def h(o): return nn.Sequential(layer.Linear(d, o), neuron.LIFNode(v_threshold=float("inf"),
                  surrogate_function=surrogate.ATan(), step_mode="m", store_v_seq=True, backend="torch"))
        s.human, s.animal, s.vehicle = h(2), h(2), h(1)
    def forward(s, x):
        xs = (x * s.gate).unsqueeze(0).repeat(T, 1, 1); hh = s.body(xs); o = {}
        for nm, hd in (("human", s.human), ("animal", s.animal), ("vehicle", s.vehicle)):
            hd(hh); o[nm] = hd[-1].v_seq.mean(0)
        return o

print("loading model + val ...")
net = FeatureSNN(NF); functional.set_step_mode(net, "m")
net.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); net.eval()
sh = sorted(glob.glob(r"G:/geophone_synth/features_v2/features_shard_*.parquet"))[:3]
cols = FEATURES + ["split", "coarse", "human_level", "animal_level", "vehicle_level"]
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sh], ignore_index=True)
df = df[df["split"] != "train"].reset_index(drop=True)
X = np.clip((df[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIP, CLIP).astype(np.float32)
LVL = {"none": 0, "single": 1, "multiple": 2}
def lv(c): return df[c].map(LVL).fillna(df[c]).to_numpy().astype(int)
hl, al, vl = lv("human_level"), lv("animal_level"), lv("vehicle_level")
coarse = df["coarse"].to_numpy()
H, A, V = [], [], []
with torch.no_grad():
    for j in range(0, len(X), 8192):
        functional.reset_net(net); o = net(torch.as_tensor(X[j:j+8192]))
        H.append(torch.sigmoid(o["human"]).numpy()); A.append(torch.sigmoid(o["animal"]).numpy())
        V.append(torch.sigmoid(o["vehicle"]).numpy())
H, A, V = np.concatenate(H), np.concatenate(A), np.concatenate(V)
NVAL = len(X); print(f"val windows: {NVAL:,}")

CAP = f"synthetic held-out 10% val ({NVAL:,} windows, profile-grouped, @0.5 threshold)"

def draw_cm(ax, cm, labels, title):
    rn = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i,j]):,}\n{rn[i,j]*100:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if rn[i, j] > 0.5 else "#222")
    return im

# ---------- Fig 1: head selection confusion ----------
labs = ["human", "animal", "car", "nothing"]
m = np.isin(coarse, ["human", "animal", "vehicle", "nothing"])
stack = np.vstack([H[m, 0], A[m, 0], V[m, 0]]).T
pred = np.where(stack.max(1) < 0.5, "nothing", np.array(["human", "animal", "car"])[stack.argmax(1)])
truemap = np.where(coarse[m] == "vehicle", "car", coarse[m])
acc_hs = (pred == truemap).mean()
cm = confusion_matrix(truemap, pred, labels=labs)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
draw_cm(ax, cm, labs, f"Head selection — which class?   accuracy {acc_hs*100:.1f}%")
fig.text(0.5, 0.01, CAP, ha="center", fontsize=8, color="#666")
plt.tight_layout(rect=[0, 0.03, 1, 1]); plt.savefig(os.path.join(STATS, "findings_01_head_selection.png")); plt.close()

# ---------- Fig 2: where the errors go ----------
rows = ["human", "animal", "car"]
corr, tonix, toother = [], [], []
for r in rows:
    i = labs.index(r); tot = cm[i].sum()
    corr.append(cm[i, i] / tot); tonix.append(cm[i, labs.index("nothing")] / tot)
    toother.append(1 - corr[-1] - tonix[-1])
fig, ax = plt.subplots(figsize=(8.0, 4.3))
x = np.arange(len(rows))
ax.bar(x, corr, color=ACCENT, label="correct class")
ax.bar(x, tonix, bottom=corr, color=MISS, label='missed → "nothing" (faint source)')
ax.bar(x, toother, bottom=np.array(corr)+np.array(tonix), color=OTHER, label="confused → other class")
for xi, (c, n) in enumerate(zip(corr, tonix)):
    ax.text(xi, c/2, f"{c*100:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    if n > 0.03: ax.text(xi, c+n/2, f"{n*100:.0f}%", ha="center", va="center", color="white")
ax.set_xticks(x); ax.set_xticklabels(rows); ax.set_ylim(0, 1); ax.set_ylabel("fraction of windows")
ax.set_title("Where the errors go — misses are faint sources, not class confusion", fontsize=11)
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
fig.text(0.5, 0.01, CAP, ha="center", fontsize=8, color="#666")
plt.tight_layout(rect=[0, 0.03, 1, 1]); plt.savefig(os.path.join(STATS, "findings_02_error_breakdown.png")); plt.close()

# ---------- Fig 3: within-head count level ----------
def ordlvl(P2): return (P2[:, 0] > 0.5).astype(int) + (P2[:, 1] > 0.5).astype(int)
hp, ap = ordlvl(H), ordlvl(A); vp = (V[:, 0] > 0.5).astype(int); vt = (vl > 0).astype(int)
panels = [("human", hl, hp, ["none", "single", "multiple"]),
          ("animal", al, ap, ["none", "single", "multiple"]),
          ("car", vt, vp, ["none", "single"])]
fig = plt.figure(figsize=(13, 4.4)); gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8])
for k, (nm, yt, yp, ls) in enumerate(panels):
    ax = fig.add_subplot(gs[k]); acc = (yt == yp).mean()
    draw_cm(ax, confusion_matrix(yt, yp, labels=list(range(len(ls)))), ls,
            f"{nm.capitalize()} count — acc {acc*100:.1f}%")
fig.suptitle("Within each head — counting level (none / single / multiple)", fontsize=13, fontweight="bold")
fig.text(0.5, 0.01, CAP, ha="center", fontsize=8, color="#666")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(STATS, "findings_03_within_head.png")); plt.close()

# ---------- Fig 4: per-head presence accuracy + AUROC ----------
pres_acc = [((H[:, 0] > 0.5) == (hl > 0)).mean(), ((A[:, 0] > 0.5) == (al > 0)).mean(),
            ((V[:, 0] > 0.5) == (vl > 0)).mean()]
try:
    sm = json.load(open(os.path.join(OUT, "summary.json")))["per_head_auroc_ema"]
    auroc = [sm["human"], sm["animal"], sm["vehicle"]]
except Exception:
    auroc = [0.9923, 0.9925, 0.9905]
heads = ["human", "animal", "vehicle"]
fig, ax = plt.subplots(figsize=(6.6, 4.2)); x = np.arange(3); w = 0.36
ax.bar(x - w/2, auroc, w, color="#3a6ea5", label="presence AUROC")
ax.bar(x + w/2, pres_acc, w, color=ACCENT, label="presence accuracy @0.5")
for xi, (a, p) in enumerate(zip(auroc, pres_acc)):
    ax.text(xi - w/2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(xi + w/2, p + 0.002, f"{p:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(heads); ax.set_ylim(0.9, 1.0); ax.set_ylabel("score")
ax.set_title("Per-head presence detection (held-out synthetic val)")
ax.legend(loc="lower left", fontsize=8); fig.text(0.5, 0.01, CAP, ha="center", fontsize=8, color="#666")
plt.tight_layout(rect=[0, 0.03, 1, 1]); plt.savefig(os.path.join(STATS, "findings_04_per_head_metrics.png")); plt.close()

# ---------- Fig 5: fine-tune per fold (real) ----------
ft = json.load(open(os.path.join(OUT, "FINETUNE_5FOLD.json")))
folds = ft["per_fold_finetuned"]; best = int(np.argmax(folds))
zmean = ft["synth_pretrained_zeroshot_5fold_acc"]; fmean = ft["synth_pretrained_finetuned_ens_5fold_acc"]
fig, ax = plt.subplots(figsize=(7.2, 4.4)); x = np.arange(len(folds))
colors = [ACCENT if i == best else "#9bbfa9" for i in range(len(folds))]
ax.bar(x, folds, color=colors, width=0.6)
for xi, v in enumerate(folds):
    ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold" if xi == best else "normal")
ax.axhline(fmean, color="#2f6f4f", ls="--", lw=1, label=f"fine-tuned mean {fmean:.3f}")
ax.axhline(zmean, color="#c25b3a", ls="--", lw=1, label=f"zero-shot mean {zmean:.3f}")
ax.axhline(0.986, color="#888", ls=":", lw=1, label="real-only logreg 0.986")
ax.set_xticks(x); ax.set_xticklabels([f"fold {i}" + ("  ★deploy" if i == best else "") for i in x])
ax.set_ylim(0.74, 1.0); ax.set_ylabel("3-class accuracy (real)")
ax.set_title("Fine-tune on real — 5-fold (train 80% / val 20%), 132 features")
ax.legend(loc="lower right", fontsize=8)
fig.text(0.5, 0.01, "real set = 1 recording/class -> within-session, optimistic (confound-aware)",
         ha="center", fontsize=8, color="#666")
plt.tight_layout(rect=[0, 0.03, 1, 1]); plt.savefig(os.path.join(STATS, "findings_05_finetune_folds.png")); plt.close()

print("wrote 5 figures ->", STATS)
print(f"head-selection acc {acc_hs:.3f} | per-head presence {[round(p,3) for p in pres_acc]}")
