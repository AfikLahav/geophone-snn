"""Comprehensive statistics/plots for the 0.977 model (synthetic-pretrained SNN, fine-tuned on real,
5-fold). Regenerates out-of-fold real predictions + characterizes the underlying synthetic-pretrained
model + training dynamics. All PNGs -> model_0977_stats/.
"""
import os, sys, json, glob, copy, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.path.join(HERE, "snn_v2_out"); FEAT_DIR = r"G:/geophone_synth/features_v2"
REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
PLOTS = os.path.join(HERE, "model_0977_stats"); os.makedirs(PLOTS, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T, SCENE, HOP, K, SCALE, CLIPZ = 4, 30 * int(F.FS), 1500, 5, 25.4, 8.0
torch.manual_seed(0); np.random.seed(0)
scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32)
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]
def fig(name): plt.tight_layout(); plt.savefig(os.path.join(PLOTS, name), dpi=120); plt.close()

class SeqBN(nn.Module):
    def __init__(s, c): super().__init__(); s.bn = nn.BatchNorm1d(c)
    def forward(s, x): T_, B_, C_ = x.shape; return s.bn(x.reshape(T_ * B_, C_)).reshape(T_, B_, C_)
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
        for nm, hd in (("human", s.human), ("animal", s.animal), ("vehicle", s.vehicle)): hd(hh); o[nm] = hd[-1].v_seq.mean(0)
        return o
def logits(model, Xz, heads=("human", "vehicle", "animal")):
    model.eval(); acc = {h: [] for h in heads}
    with torch.no_grad():
        for j in range(0, len(Xz), 8192):
            functional.reset_net(model); o = model(torch.as_tensor(Xz[j:j + 8192]).to(DEV))
            for h in heads: acc[h].append(o[h].cpu())
    return {h: torch.cat(acc[h]) for h in heads}

# ---------------- REAL: featurize + 5-fold fine-tune -> out-of-fold preds ----------------
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
def feat(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE; fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)
X, y, blk, src = [], [], [], []
for fn, cls in REALS.items():
    Xr = feat(os.path.join(REAL_DIR, fn))[:, fidx]; n = len(Xr); X.append(Xr.astype(np.float32))
    y += [cls] * n; blk.append(np.arange(n) * K // max(n, 1)); src += [fn] * n
X = np.vstack(X); y = np.array(y); blk = np.concatenate(blk); src = np.array(src)
yh = (y == "human").astype(np.float32); yv = (y == "vehicle").astype(np.float32)
def train(model, Xz, tr, epochs, lr, seed):
    torch.manual_seed(seed); Xt = torch.as_tensor(Xz[tr]).to(DEV); ht = torch.as_tensor(yh[tr]).to(DEV); vt = torch.as_tensor(yv[tr]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); model.train(); n = len(Xt)
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + np.cos(np.pi * ep / epochs))
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, 256):
            idx = perm[j:j + 256]; functional.reset_net(model); o = model(Xt[idx])
            loss = Fnn.binary_cross_entropy_with_logits(o["human"][:, 0], ht[idx]) + Fnn.binary_cross_entropy_with_logits(o["vehicle"][:, 0], vt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model
ENS = 5; ph_oof = np.zeros(len(X)); pv_oof = np.zeros(len(X)); ph_zs = np.zeros(len(X)); pv_zs = np.zeros(len(X))
fold_of = np.zeros(len(X), int)
for k in range(K):
    te = blk == k; tr = ~te; fold_of[te] = k
    sc = StandardScaler().fit(X[tr]); Xz = np.clip(sc.transform(X), -CLIPZ, CLIPZ).astype(np.float32)
    base = FeatureSNN(NF).to(DEV); functional.set_step_mode(base, "m")
    base.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV))
    lo = logits(base, Xz[te], ("human", "vehicle")); ph_zs[te] = torch.sigmoid(lo["human"][:, 0]).numpy(); pv_zs[te] = torch.sigmoid(lo["vehicle"][:, 0]).numpy()
    ah = np.zeros(te.sum()); av = np.zeros(te.sum())
    for s in range(ENS):
        m = copy.deepcopy(base); m = train(m, Xz, tr, 80, 4e-4, s)
        lo = logits(m, Xz[te], ("human", "vehicle")); ah += torch.sigmoid(lo["human"][:, 0]).numpy(); av += torch.sigmoid(lo["vehicle"][:, 0]).numpy()
    ph_oof[te] = ah / ENS; pv_oof[te] = av / ENS
    print(f"fold {k} done", flush=True)
# per-head thresholds (balanced-acc on OOF)
def bt(p, t): g = np.linspace(0.05, 0.95, 19); return float(g[np.argmax([balanced_accuracy_score(t, p > x) for x in g])])
TH, TV = bt(ph_oof, yh > 0), bt(pv_oof, yv > 0)
def decide(ph, pv, th, tv): mh, mv = ph - th, pv - tv; return np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
pred = decide(ph_oof, pv_oof, TH, TV); acc = accuracy_score(y, pred)
print(f"OOF real accuracy: {acc:.3f}  (thr h{TH:.2f}/v{TV:.2f})")

labs = ["human", "vehicle", "nothing"]
# 1-2 confusion
cm = confusion_matrix(y, pred, labels=labs)
for norm, nm in [(False, "01_real_confusion.png"), (True, "02_real_confusion_norm.png")]:
    M = cm / cm.sum(1, keepdims=True) if norm else cm
    plt.figure(figsize=(5, 4.3)); plt.imshow(M, cmap="Blues", vmin=0, vmax=(1 if norm else None))
    for (a, b), v in np.ndenumerate(M): plt.text(b, a, f"{v:.2f}" if norm else int(v), ha="center", va="center")
    plt.xticks(range(3), labs); plt.yticks(range(3), labs); plt.xlabel("predicted"); plt.ylabel("true")
    plt.title(f"REAL 3-class ({'normalized' if norm else 'counts'}) acc {acc:.3f}"); plt.colorbar(); fig(nm)
# 3 per-class P/R/F1
P, R, Fs, _ = precision_recall_fscore_support(y, pred, labels=labs, zero_division=0)
x = np.arange(3); w = 0.25
plt.figure(figsize=(7, 4)); plt.bar(x - w, P, w, label="precision"); plt.bar(x, R, w, label="recall"); plt.bar(x + w, Fs, w, label="F1")
plt.xticks(x, labs); plt.ylim(0, 1.05); plt.legend(); plt.title("REAL per-class precision/recall/F1"); fig("03_real_prf.png")
# 4-5 ROC + PR per head
for head, p, yy, nm in [("human", ph_oof, (y == "human").astype(int), "h"), ("vehicle", pv_oof, (y == "vehicle").astype(int), "v")]:
    fpr, tpr, _ = roc_curve(yy, p); plt.figure(figsize=(4.6, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--", lw=.5)
    plt.title(f"REAL {head} ROC (AUC {roc_auc_score(yy,p):.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR"); fig(f"04_real_roc_{head}.png")
    pr, rc, _ = precision_recall_curve(yy, p); plt.figure(figsize=(4.6, 4)); plt.plot(rc, pr)
    plt.title(f"REAL {head} PR (AP {average_precision_score(yy,p):.3f})"); plt.xlabel("recall"); plt.ylabel("precision"); fig(f"05_real_pr_{head}.png")
# 6 calibration
plt.figure(figsize=(5, 4)); plt.plot([0, 1], [0, 1], "k--", lw=.5)
for head, p, yy in [("human", ph_oof, (y == "human").astype(int)), ("vehicle", pv_oof, (y == "vehicle").astype(int))]:
    b = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, b) - 1, 0, 9)
    conf = [p[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]; ac = [yy[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)]
    plt.plot(conf, ac, "o-", label=head)
plt.legend(); plt.xlabel("confidence"); plt.ylabel("empirical accuracy"); plt.title("REAL reliability"); fig("06_real_calibration.png")
# 7 confidence correct vs wrong
win = np.maximum(ph_oof, pv_oof); corr = pred == y
plt.figure(figsize=(6, 4)); plt.hist(win[corr], bins=30, alpha=.6, label="correct", density=True); plt.hist(win[~corr], bins=30, alpha=.6, label="wrong", density=True)
plt.legend(); plt.xlabel("winning-head prob"); plt.title("REAL confidence: correct vs wrong"); fig("07_real_confidence.png")
# 8 per-recording acc
plt.figure(figsize=(6, 4)); rec_acc = [accuracy_score(y[src == fn], pred[src == fn]) for fn in REALS]
plt.bar(range(4), rec_acc); plt.xticks(range(4), list(REALS), rotation=20, ha="right"); plt.ylim(0, 1.05)
for i, a in enumerate(rec_acc): plt.text(i, a + .01, f"{a:.3f}", ha="center")
plt.title("REAL accuracy per recording"); fig("08_real_per_recording.png")
# 9 per-fold acc
plt.figure(figsize=(6, 4)); fa = [accuracy_score(y[fold_of == k], pred[fold_of == k]) for k in range(K)]
plt.bar(range(K), fa); plt.axhline(acc, color="r", ls="--", label=f"mean {acc:.3f}"); plt.ylim(0.9, 1.0)
plt.xlabel("fold"); plt.ylabel("accuracy"); plt.legend(); plt.title("REAL per-fold accuracy"); fig("09_real_per_fold.png")
# 10 zero-shot vs fine-tuned per fold
zpred = decide(ph_zs, pv_zs, bt(ph_zs, yh > 0), bt(pv_zs, yv > 0))
zfa = [accuracy_score(y[fold_of == k], zpred[fold_of == k]) for k in range(K)]
plt.figure(figsize=(6, 4)); plt.plot(range(K), zfa, "o-", label="zero-shot"); plt.plot(range(K), fa, "s-", label="fine-tuned")
plt.xlabel("fold"); plt.ylabel("accuracy"); plt.legend(); plt.title("REAL zero-shot vs fine-tuned (per fold)"); fig("10_real_zeroshot_vs_ft.png")
# 11 prob scatter
plt.figure(figsize=(5.5, 5)); cmap = {"human": "C0", "vehicle": "C1", "nothing": "C7"}
for c in labs: m = y == c; plt.scatter(ph_oof[m], pv_oof[m], s=4, alpha=.3, c=cmap[c], label=c)
plt.axvline(TH, color="C0", ls=":", lw=.7); plt.axhline(TV, color="C1", ls=":", lw=.7)
plt.xlabel("human-present prob"); plt.ylabel("vehicle-present prob"); plt.legend(); plt.title("REAL head probs by true class"); fig("11_real_prob_scatter.png")
# 12 prob distributions
fig12, ax = plt.subplots(1, 2, figsize=(10, 4))
for c in labs: ax[0].hist(ph_oof[y == c], bins=30, alpha=.5, label=c, density=True)
ax[0].set_title("human-head prob"); ax[0].legend(); ax[0].set_xlabel("prob")
for c in labs: ax[1].hist(pv_oof[y == c], bins=30, alpha=.5, label=c, density=True)
ax[1].set_title("vehicle-head prob"); ax[1].legend(); ax[1].set_xlabel("prob"); fig("12_real_prob_dist.png")
# 13 accuracy vs threshold
grid = np.linspace(0.05, 0.95, 37); av = [accuracy_score(y, decide(ph_oof, pv_oof, t, t)) for t in grid]
plt.figure(figsize=(6, 4)); plt.plot(grid, av, "-o", ms=3); plt.axvline(0.5, color="r", ls="--", lw=.7)
plt.xlabel("global threshold"); plt.ylabel("accuracy"); plt.title("REAL accuracy vs threshold"); fig("13_real_acc_vs_threshold.png")
np.savez(os.path.join(PLOTS, "real_oof.npz"), ph=ph_oof, pv=pv_oof, y=y, src=src, fold=fold_of)

# ---------------- SYNTHETIC-pretrained model on synthetic val ----------------
LVL = {"none": 0, "single": 1, "multiple": 2}
dv = pd.concat([pd.read_parquet(s, columns=FEATURES + ["split", "coarse", "family", "human_level", "human_snr",
               "vehicle_level", "animal_level"]) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))[:3]], ignore_index=True)
dv = dv[dv.split == "val"].reset_index(drop=True)
samp = np.random.default_rng(0).choice(len(dv), min(200000, len(dv)), replace=False); dv = dv.iloc[samp].reset_index(drop=True)
Xv = np.clip((dv[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ)
def lv(c): return dv[c].map(LVL).fillna(dv[c]).to_numpy().astype(int)
H, V, A = lv("human_level"), lv("vehicle_level"), lv("animal_level")
em = FeatureSNN(NF).to(DEV); functional.set_step_mode(em, "m"); em.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV))
lg = logits(em, Xv)
# 14 synth confusion per head + 15 ROC + 16 PR + 17 calib
HEADS = {"human": (lg["human"], H, 3), "animal": (lg["animal"], A, 3), "vehicle": (lg["vehicle"], V, 2)}
for nm, (l, lvl, k) in HEADS.items():
    p1 = torch.sigmoid(l[:, 0]).numpy()
    if k == 3:
        p2 = torch.sigmoid(l[:, 1]).numpy(); prd = (p1 > .5).astype(int) + (p2 > .5).astype(int); cm = confusion_matrix(lvl, prd)
        tk = ["none", "single", "multi"]
    else:
        prd = (p1 > .5).astype(int); cm = confusion_matrix((lvl > 0).astype(int), prd); tk = ["none", "present"]
    plt.figure(figsize=(4.4, 4)); plt.imshow(cm, cmap="Greens")
    for (a, b), v in np.ndenumerate(cm): plt.text(b, a, int(v), ha="center", va="center")
    plt.xticks(range(len(tk)), tk); plt.yticks(range(len(tk)), tk); plt.xlabel("pred"); plt.ylabel("true")
    plt.title(f"SYNTH {nm} confusion"); fig(f"14_synth_cm_{nm}.png")
    yy = (lvl > 0).astype(int)
    fpr, tpr, _ = roc_curve(yy, p1); plt.figure(figsize=(4.4, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--", lw=.5)
    plt.title(f"SYNTH {nm} presence ROC (AUC {roc_auc_score(yy,p1):.3f})"); fig(f"15_synth_roc_{nm}.png")
    pr, rc, _ = precision_recall_curve(yy, p1); plt.figure(figsize=(4.4, 4)); plt.plot(rc, pr)
    plt.title(f"SYNTH {nm} presence PR (AP {average_precision_score(yy,p1):.3f})"); fig(f"16_synth_pr_{nm}.png")
# 18 per-family accuracy (presence, any head)
present_true = ((H > 0) | (V > 0) | (A > 0)).astype(int)
score_any = np.maximum.reduce([torch.sigmoid(lg[h][:, 0]).numpy() for h in ("human", "vehicle", "animal")])
fams = sorted(dv.family.unique()); fa2 = [roc_auc_score(present_true[dv.family.to_numpy() == f], score_any[dv.family.to_numpy() == f])
       if len(np.unique(present_true[dv.family.to_numpy() == f])) == 2 else np.nan for f in fams]
plt.figure(figsize=(9, 4)); plt.bar(range(len(fams)), fa2); plt.xticks(range(len(fams)), fams, rotation=60, ha="right")
plt.ylabel("any-present AUROC"); plt.title("SYNTH per-terrain-family detection"); fig("18_synth_per_family.png")
# 19 accuracy vs SNR (human presence)
snr = dv.human_snr.to_numpy(float); bins = np.arange(-30, 35, 3); cen = (bins[:-1] + bins[1:]) / 2
ph_s = torch.sigmoid(lg["human"][:, 0]).numpy(); thr = bt(ph_s, H > 0)
pdc = [((ph_s[(snr >= bins[i]) & (snr < bins[i + 1]) & (H > 0)] > thr)).mean() if ((snr >= bins[i]) & (snr < bins[i + 1]) & (H > 0)).sum() >= 30 else np.nan for i in range(len(bins) - 1)]
plt.figure(figsize=(6, 4)); plt.plot(cen, pdc, "-o", ms=3); plt.axvline(0, color="r", ls="--", lw=.7, label="0 dB label cutoff")
plt.xlabel("human SNR (dB)"); plt.ylabel("detection rate"); plt.legend(); plt.title("SYNTH human detection vs SNR"); fig("19_synth_acc_vs_snr.png")
# 21 feature gate
gate = em.gate.detach().cpu().numpy(); order = np.argsort(-np.abs(gate))
plt.figure(figsize=(11, 4)); plt.bar(range(NF), gate[order]); plt.xticks([]); plt.ylabel("gate weight"); plt.title("Learned FeatureGate (sorted)"); fig("21_feature_gate.png")
# 22 PCA + 23 correlation
from sklearn.decomposition import PCA
sub = np.random.default_rng(0).choice(len(Xv), min(20000, len(Xv)), replace=False)
pc = PCA(2).fit_transform(Xv[sub])
plt.figure(figsize=(6, 5)); plt.scatter(pc[:, 0], pc[:, 1], c=present_true[sub], s=4, cmap="coolwarm", alpha=.4)
plt.title("PCA of features (val), present vs nothing"); plt.xlabel("PC1"); plt.ylabel("PC2"); fig("22_pca.png")
C = np.corrcoef(Xv[sub].T); plt.figure(figsize=(8, 7)); plt.imshow(C, cmap="RdBu", vmin=-1, vmax=1); plt.colorbar(); plt.title("Feature correlation (104)"); fig("23_feature_correlation.png")

# ---------------- training dynamics (history.json) ----------------
try:
    h = json.load(open(os.path.join(OUT, "history.json")))
    plt.figure(figsize=(7, 4)); plt.plot(h["step"], h["val_auroc_raw"], label="raw"); plt.plot(h["step"], h["val_auroc_ema"], label="EMA")
    plt.xlabel("step"); plt.ylabel("val AUROC"); plt.legend(); plt.title("Training: val AUROC raw vs EMA"); fig("24_train_auroc.png")
    plt.figure(figsize=(7, 4)); plt.plot(h["step"], h["train_loss"]); plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss"); fig("25_train_loss.png")
    fr = np.array(h["fire"]); de = np.array(h["dead"])
    plt.figure(figsize=(7, 4)); [plt.plot(h["step"], fr[:, i], label=f"PLIF{i}") for i in range(fr.shape[1])]; plt.legend(); plt.xlabel("step"); plt.ylabel("firing rate"); plt.title("Per-layer firing rate"); fig("26_firing_rate.png")
    plt.figure(figsize=(7, 4)); [plt.plot(h["step"], de[:, i], label=f"PLIF{i}") for i in range(de.shape[1])]; plt.legend(); plt.xlabel("step"); plt.ylabel("dead fraction"); plt.title("Per-layer dead-neuron %"); fig("27_dead_neurons.png")
    plt.figure(figsize=(7, 4)); plt.plot(h["step"], h["lr"]); plt.xlabel("step"); plt.ylabel("lr"); plt.title("LR schedule"); fig("28_lr.png")
    plt.figure(figsize=(7, 4)); plt.plot(h["step"], h["gradnorm"]); plt.xlabel("step"); plt.ylabel("grad norm"); plt.title("Gradient norm"); fig("29_gradnorm.png")
except Exception as e: print("history plots skipped:", e)

# ---------------- summary ----------------
plt.figure(figsize=(7, 4)); stages = ["zero-shot\nraw", "+threshold\ncal", "+scale\nalign", "fine-tuned\n(0.977)", "scratch\nbaseline"]
vals = [0.805, 0.878, 0.900, acc, 0.966]
plt.bar(range(5), vals, color=["C7", "C7", "C7", "C2", "C3"]); plt.xticks(range(5), stages); plt.ylim(0.7, 1.0)
for i, v in enumerate(vals): plt.text(i, v + .005, f"{v:.3f}", ha="center")
plt.ylabel("real 3-class accuracy"); plt.title("Accuracy progression"); fig("30_accuracy_progression.png")

print(f"\nDONE -> {PLOTS}")
print(f"plots: {len([f for f in os.listdir(PLOTS) if f.endswith('.png')])} PNGs")
print("\nCANNOT plot (no ground truth in the real set):")
print("  - single/multiple (ordinal) on REAL  - no count labels")
print("  - animal class on REAL               - absent from the real recordings")
print("  - per-window SNR / detection-vs-SNR on REAL - no clean/noise split for real")
print("  - event-level FAR / hysteresis on a deployment stream - 4 CSVs aren't a long 99%-nothing stream")
print("  - proper cross-session generalization - only 1 recording per class")
