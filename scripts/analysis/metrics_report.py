"""Explicit accuracy table for the trained synthetic SNN (per-head accuracy / balanced-acc /
macro-F1 on the synthetic VAL split), plus a combined comparison with the real-data baseline.
Loads the saved EMA checkpoint (torch backend for inference — no cupy needed). Writes
ACCURACY_SUMMARY.json + prints the comparison table.
"""
import os, sys, json, glob, sqlite3
import numpy as np, pandas as pd, torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "simgeo"))
import features as Fx
OUT = os.path.join(HERE, "snn_v2_out"); FEAT_DIR = os.path.join(_GEO_ROOT, "features_v2")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T = 4
sc = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = sc["features"]; NF = len(FEATURES)
mu = np.array(sc["mean"], np.float32); sd = np.array(sc["std"], np.float32); CLIPZ = sc["clip"]


class SeqBN(nn.Module):
    def __init__(self, c): super().__init__(); self.bn = nn.BatchNorm1d(c)
    def forward(self, x): T_, B_, C_ = x.shape; return self.bn(x.reshape(T_ * B_, C_)).reshape(T_, B_, C_)


class FeatureSNN(nn.Module):
    def __init__(self, nf, widths=(512, 256, 128), dropout=0.1):
        super().__init__(); self.gate = nn.Parameter(torch.ones(nf)); body = []; d = nf
        for w in widths:
            body += [layer.Linear(d, w), SeqBN(w),
                     neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(2.0),
                                              detach_reset=True, step_mode="m"), layer.Dropout(dropout)]; d = w
        self.body = nn.Sequential(*body)
        def head(o): return nn.Sequential(layer.Linear(d, o), neuron.LIFNode(v_threshold=float("inf"),
                     surrogate_function=surrogate.ATan(), step_mode="m", store_v_seq=True, backend="torch"))
        self.human, self.animal, self.vehicle = head(2), head(2), head(1)
    def forward(self, x):
        xs = (x * self.gate).unsqueeze(0).repeat(T, 1, 1); h = self.body(xs); o = {}
        for nm, hd in (("human", self.human), ("animal", self.animal), ("vehicle", self.vehicle)):
            hd(h); o[nm] = hd[-1].v_seq.mean(0)
        return o


net = FeatureSNN(NF).to(DEV); functional.set_step_mode(net, "m")     # torch backend everywhere (inference)
net.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); net.eval()

LVL = {"none": 0, "single": 1, "multiple": 2}
cols = FEATURES + ["split", "human_level", "vehicle_level", "animal_level"]
df = pd.concat([pd.read_parquet(s, columns=cols) for s in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))],
               ignore_index=True)
df = df[df.split == "val"].reset_index(drop=True)
X = np.clip((df[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ)
def lv(c): return df[c].map(LVL).fillna(df[c]).to_numpy().astype(int)
H, A, V = lv("human_level"), lv("animal_level"), lv("vehicle_level")

logits = {"human": [], "animal": [], "vehicle": []}
with torch.no_grad():
    for j in range(0, len(X), 8192):
        functional.reset_net(net); o = net(torch.as_tensor(X[j:j + 8192]).to(DEV))
        for k in logits: logits[k].append(o[k].cpu())
logits = {k: torch.cat(v) for k, v in logits.items()}

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
def ordinal_pred(lg): p = torch.sigmoid(lg); return ((p[:, 0] > .5).int() + (p[:, 1] > .5).int()).numpy()
rep = {}
for nm, lg, lvl in (("human", logits["human"], H), ("animal", logits["animal"], A)):
    pred = ordinal_pred(lg)
    rep[nm] = {"level_accuracy": float(accuracy_score(lvl, pred)),
               "level_balanced_acc": float(balanced_accuracy_score(lvl, pred)),
               "level_macro_f1": float(f1_score(lvl, pred, average="macro")),
               "presence_accuracy": float(accuracy_score((lvl > 0).astype(int), (pred > 0).astype(int))),
               "presence_auroc": float(roc_auc_score((lvl > 0).astype(int), torch.sigmoid(lg[:, 0]).numpy()))}
pv = torch.sigmoid(logits["vehicle"][:, 0]).numpy(); yv = (V > 0).astype(int)
rep["vehicle"] = {"presence_accuracy": float(accuracy_score(yv, (pv > .5).astype(int))),
                  "presence_balanced_acc": float(balanced_accuracy_score(yv, (pv > .5).astype(int))),
                  "presence_auroc": float(roc_auc_score(yv, pv))}

base = json.load(open(os.path.join(OUT, "real_baseline.json"))) if os.path.exists(os.path.join(OUT, "real_baseline.json")) else {}
real = json.load(open(os.path.join(OUT, "real_test.json"))) if os.path.exists(os.path.join(OUT, "real_test.json")) else {}
summary = {"synthetic_val_per_head": rep, "synthetic_real_test": real,
           "real_only_baseline_5fold": {k: base.get(k) for k in
               ("3class_logreg", "3class_hgb", "3class_logreg_robustfeat", "3class_hgb_robustfeat", "hv_hgb")}}
json.dump(summary, open(os.path.join(OUT, "ACCURACY_SUMMARY.json"), "w"), indent=1)

print("=== SYNTHETIC-TRAINED SNN — synthetic val (held-out profiles) ===")
for nm in ("human", "animal", "vehicle"):
    r = rep[nm]; print(f"  {nm:8s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
print("\n=== SYNTHETIC-TRAINED SNN — run-once REAL test ===")
print(f"  human-vs-vehicle AUC {real.get('human_vs_vehicle_AUC'):.3f} | FA-on-nothing "
      f"h={real['FAR_on_nothing']['human_head_FA_rate_on_nothing']:.3f} "
      f"v={real['FAR_on_nothing']['vehicle_head_FA_rate_on_nothing']:.3f} | domain-AUC {real.get('domain_classifier_AUC'):.2f}")
print("\n=== REAL-ONLY BASELINE (5-fold, session-confounded) ===")
print(f"  3-class HGB acc {base['3class_hgb']['acc_mean']:.3f}  (robust-feat-only {base['3class_hgb_robustfeat']['acc_mean']:.3f})")
print(f"  human-vs-vehicle HGB acc {base['hv_hgb']['acc_mean']:.3f}")
print("\nwrote", os.path.join(OUT, "ACCURACY_SUMMARY.json"))
