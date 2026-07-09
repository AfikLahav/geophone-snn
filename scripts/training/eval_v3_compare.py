"""v3 model comparison — the SYNTHETIC side (the real side is done by finetune_5fold.py /
snn_real_baseline.py). Also TRAINS + SAVES the real-only model C (the baseline script doesn't persist it).

  A = pre-trained                 : snn_v3_out/model_ema.pt
  B = pre-trained + fine-tuned    : snn_v3_out/model_finetuned.pt   (best-fold ensemble)
  C = real-only (from scratch)    : trained here -> snn_v3_out/model_real_only.pt

Evals on the v3 synthetic VAL split (split=='val' in the parquet):
  - A vs B : per-head presence AUROC (human/vehicle/animal) + single-class 4-way accuracy
  - C      : restricted to human/vehicle/nothing (animal head untrained on real) — AUROC + 3-way acc
Consolidates the real-side numbers from FINETUNE_5FOLD.json / SNN_REAL_BASELINE.json into one report.
Run: GEO_OUT=snn_v3_out GEO_FEAT_DIR=features_v3 python eval_v3_compare.py
"""
import os, sys, json, glob, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)          # SpikingJelly cupy shim
import warnings; warnings.filterwarnings("ignore")
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as F
OUT = os.environ.get("GEO_OUT") or os.path.join(HERE, "snn_v3_out")
FEAT_DIR = os.environ.get("GEO_FEAT_DIR", r"G:/geophone_synth/features_v3")
REAL_DIR = os.path.join(HERE, "Goephone-Project", "geophone_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, SCENE, HOP, SCALE = 4, 30 * int(F.FS), 1500, 25.4
torch.manual_seed(0); np.random.seed(0)

scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]


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


def heads_probs(model, Xz):
    model.eval(); ph, pv, pa = [], [], []
    with torch.no_grad():
        for j in range(0, len(Xz), 4096):
            functional.reset_net(model); o = model(torch.as_tensor(Xz[j:j + 4096]).to(DEV))
            ph.append(torch.sigmoid(o["human"][:, 0]).cpu()); pv.append(torch.sigmoid(o["vehicle"][:, 0]).cpu())
            pa.append(torch.sigmoid(o["animal"][:, 0]).cpu())
    return (np.array(torch.cat(ph)), np.array(torch.cat(pv)), np.array(torch.cat(pa)))


def ens_probs(members, Xz):
    ph = pv = pa = 0.0
    for st in members:
        m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m"); m.load_state_dict(st)
        a, b, c = heads_probs(m, Xz); ph = ph + a; pv = pv + b; pa = pa + c
    n = len(members); return ph / n, pv / n, pa / n


def auroc(y, p): return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


# ---------------- synthetic VAL ----------------
shards = sorted(glob.glob(os.path.join(FEAT_DIR, "features_shard_*.parquet")))
COLS = FEATURES + ["split", "coarse", "human_level", "vehicle_level", "animal_level", "common_snr",
                   "human_snr", "vehicle_snr", "animal_snr"]
df = pd.concat([pd.read_parquet(s, columns=COLS) for s in shards], ignore_index=True)
val = df[df["split"] == "val"].reset_index(drop=True)
Xv = np.clip((val[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
coarse = val["coarse"].to_numpy()
yh = (val["human_level"].to_numpy() > 0); yv = (val["vehicle_level"].to_numpy() > 0); ya = (val["animal_level"].to_numpy() > 0)
GATES = json.load(open(os.path.join(HERE, "gap_study", "v4_plan", "gates.json")))["classes"]
# gated keep-mask per head: negatives + detectable positives (present-but-sub-tau_hi dropped, not negatives)
KEEP = {nm: ((~y) | (val[f"{nm}_snr"].to_numpy(float) >= GATES[nm]["tau_hi_db"]))
        for nm, y in (("human", yh), ("vehicle", yv), ("animal", ya))}
print(f"synthetic val: {len(val):,} windows | {NF} feats", flush=True)

single = np.isin(coarse, ["human", "vehicle", "animal", "nothing"])    # 4-way accuracy on single-class windows
true4 = coarse[single]


def synth_report(ph, pv, pa, thr=0.5):
    r = {"AUROC_human": auroc(yh, ph), "AUROC_vehicle": auroc(yv, pv), "AUROC_animal": auroc(ya, pa),
         "AUROC_human_gated": auroc(yh[KEEP["human"]], ph[KEEP["human"]]),
         "AUROC_vehicle_gated": auroc(yv[KEEP["vehicle"]], pv[KEEP["vehicle"]]),
         "AUROC_animal_gated": auroc(ya[KEEP["animal"]], pa[KEEP["animal"]])}
    P = np.vstack([ph[single], pv[single], pa[single]]).T; names = np.array(["human", "vehicle", "animal"])
    pred = np.where(P.max(1) < thr, "nothing", names[P.argmax(1)])
    r["acc4"] = float(accuracy_score(true4, pred)); r["bal_acc4"] = float(balanced_accuracy_score(true4, pred))
    r["recall"] = {c: float((pred[true4 == c] == c).mean()) for c in ["human", "vehicle", "animal", "nothing"] if (true4 == c).any()}
    return r


# A (pretrained) and B (finetuned ensemble)
A = FeatureSNN(NF).to(DEV); functional.set_step_mode(A, "m"); A.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV))
Aph, Apv, Apa = heads_probs(A, Xv); rA = synth_report(Aph, Apv, Apa)
print("A (pretrained) on synthetic:", json.dumps(rA), flush=True)

Bart = torch.load(os.path.join(OUT, "model_finetuned.pt"), map_location=DEV)
Bph, Bpv, Bpa = ens_probs(Bart["members"], Xv); rB = synth_report(Bph, Bpv, Bpa)
print("B (finetuned)  on synthetic:", json.dumps(rB), flush=True)

# ---------------- train + save C (real-only) ----------------
REALS = {"car.csv": "vehicle", "human.csv": "human", "car_nothing.csv": "nothing", "human_nothing.csv": "nothing"}
def feat_real(p):
    a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE; fe = []
    for c0 in range(0, len(a), SCENE):
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW: continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP): fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)
RX, RY = [], []
for fn, cls in REALS.items():
    Xr = feat_real(os.path.join(REAL_DIR, fn))[:, fidx]; RX.append(Xr.astype(np.float32)); RY += [cls] * len(Xr)
RX = np.vstack(RX); RY = np.array(RY)
rsc = StandardScaler().fit(RX); RXz = np.clip(rsc.transform(RX), -CLIPZ, CLIPZ).astype(np.float32)
ryh = (RY == "human").astype(np.float32); ryv = (RY == "vehicle").astype(np.float32)

def train_realonly(seed, epochs=200, lr=5e-4):
    torch.manual_seed(seed); m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m")
    Xt = torch.as_tensor(RXz).to(DEV); ht = torch.as_tensor(ryh).to(DEV); vt = torch.as_tensor(ryv).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4); m.train(); n = len(Xt)
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + np.cos(np.pi * ep / epochs))
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, 256):
            idx = perm[j:j + 256]; functional.reset_net(m); o = m(Xt[idx])
            loss = (Fnn.binary_cross_entropy_with_logits(o["human"][:, 0], ht[idx])
                    + Fnn.binary_cross_entropy_with_logits(o["vehicle"][:, 0], vt[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
    return m

ENS = 5
Cmembers = [{k: v.detach().cpu() for k, v in train_realonly(s).state_dict().items()} for s in range(ENS)]
torch.save({"members": Cmembers, "arch": "FeatureSNN", "nf": NF, "widths": [512, 256, 128], "T": T,
            "ens": ENS, "scaler": {"mean": rsc.mean_.tolist(), "std": rsc.scale_.tolist(), "clip": float(CLIPZ)},
            "note": "real-only (scratch), classes human/vehicle/nothing, NO animal head trained"},
           os.path.join(OUT, "model_real_only.pt"))
print(f"C (real-only) trained+saved: {len(RX)} real windows, ens{ENS} -> model_real_only.pt", flush=True)

# C on synthetic, restricted to human/vehicle/nothing (drop animal + mixed; per-domain synthetic scaler)
hvn = np.isin(coarse, ["human", "vehicle", "nothing"])
Cph, Cpv, _ = ens_probs(Cmembers, Xv[hvn])
truth_hvn = coarse[hvn]
P = np.vstack([Cph, Cpv]).T; pred = np.where(P.max(1) < 0.5, "nothing", np.array(["human", "vehicle"])[P.argmax(1)])
rC = {"AUROC_human": auroc(yh[hvn], Cph), "AUROC_vehicle": auroc(yv[hvn], Cpv),
      "acc3_hvn": float(accuracy_score(truth_hvn, pred)), "bal_acc3_hvn": float(balanced_accuracy_score(truth_hvn, pred)),
      "recall": {c: float((pred[truth_hvn == c] == c).mean()) for c in ["human", "vehicle", "nothing"] if (truth_hvn == c).any()},
      "note": "synthetic restricted to human/vehicle/nothing (real-only model has no animal head)"}
print("C (real-only)  on synthetic h/v/n:", json.dumps(rC), flush=True)

# ---------------- consolidate real-side (written by the other scripts) ----------------
def load(j):
    p = os.path.join(OUT, j); return json.load(open(p)) if os.path.exists(p) else None
report = {"synthetic_val": {"A_pretrained": rA, "B_finetuned": rB, "C_real_only_hvn": rC, "n_val_windows": int(len(val))},
          "real_side": {"finetune_5fold": load("FINETUNE_5FOLD.json"), "snn_real_baseline": load("SNN_REAL_BASELINE.json")},
          "models_saved": ["model_ema.pt (A)", "model_finetuned.pt (B, best fold)", "model_real_only.pt (C)"]}
json.dump(report, open(os.path.join(OUT, "EVAL_V3_COMPARE.json"), "w"), indent=1)
print("\n==================== v3 MODEL COMPARISON ====================")
print(f"SYNTHETIC val ({len(val):,} windows) — per-head presence AUROC:")
print(f"  A pretrained : human {rA['AUROC_human']:.3f}  vehicle {rA['AUROC_vehicle']:.3f}  animal {rA['AUROC_animal']:.3f}  | 4-way acc {rA['acc4']:.3f}")
print(f"  B finetuned  : human {rB['AUROC_human']:.3f}  vehicle {rB['AUROC_vehicle']:.3f}  animal {rB['AUROC_animal']:.3f}  | 4-way acc {rB['acc4']:.3f}")
print(f"  C real-only  : human {rC['AUROC_human']:.3f}  vehicle {rC['AUROC_vehicle']:.3f}  (no animal)        | 3-way acc(h/v/n) {rC['acc3_hvn']:.3f}")
ft = load("FINETUNE_5FOLD.json"); bl = load("SNN_REAL_BASELINE.json")
if ft: print(f"REAL — A zero-shot {ft['synth_pretrained_zeroshot_5fold_acc']:.3f} | B finetuned {ft['synth_pretrained_finetuned_ens_5fold_acc']:.3f} +/- {ft['finetuned_std']:.3f}")
if bl: print(f"REAL — C scratch {bl['SNN_from_scratch_on_real_5fold']:.3f} +/- {bl['SNN_scratch_std']:.3f} | B finetuned {bl['synth_pretrained_finetuned_5fold']:.3f} +/- {bl['finetune_std']:.3f}")
print("wrote EVAL_V3_COMPARE.json")
