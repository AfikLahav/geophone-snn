"""Re-score the v2 production model under the v3.1 detectability-gated metric — the
apples-to-apples baseline for the v3.1 acceptance test (GENERATION_PLAN_V4.md §2).

Loads snn_v2_out/model_ema.pt + features_v2 val split, computes per-head presence AUROC
both UNGATED (the original 0.9909 definition) and GATED (positives = present & snr >=
tau_hi from gates.json; present-but-sub-tau_hi dropped, NOT negatives). Expected: gating
barely moves v2 (its present windows are ~100% above floor). Writes snn_v2_out/GATED_RESCORE.json.

Run:  python dataset_validation/gated_rescore.py     (env GEO_V2_OUT / GEO_V2_FEAT to override)
"""
import os, sys, json, glob
import numpy as np, pandas as pd
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)          # SpikingJelly cupy shim
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import roc_auc_score

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.environ.get("GEO_V2_OUT") or os.path.join(ROOT, "snn_v2_out")
FEAT_DIR = os.environ.get("GEO_V2_FEAT", os.path.join(_GEO_ROOT, "features_v2"))
GATES = json.load(open(os.path.join(HERE, "gates.json")))["classes"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = 4

scj = json.load(open(os.path.join(OUT, "scaler.json"))); FEATURES = scj["features"]; NF = len(FEATURES)
mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]


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


COLS = FEATURES + ["split", "human_level", "vehicle_level", "animal_level",
                   "human_snr", "vehicle_snr", "animal_snr"]
shards = sorted(glob.glob(os.path.join(FEAT_DIR, "features_shard_*.parquet")))
df = pd.concat([pd.read_parquet(s, columns=COLS) for s in shards], ignore_index=True)
val = df[df["split"] == "val"].reset_index(drop=True); del df
Xv = np.clip((val[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
print(f"v2 val: {len(val):,} windows | {NF} feats | model {os.path.join(OUT, 'model_ema.pt')}")

m = FeatureSNN(NF).to(DEV); functional.set_step_mode(m, "m")
m.load_state_dict(torch.load(os.path.join(OUT, "model_ema.pt"), map_location=DEV)); m.eval()
probs = {}
with torch.no_grad():
    chunks = {nm: [] for nm in ("human", "vehicle", "animal")}
    for j in range(0, len(Xv), 4096):
        functional.reset_net(m); o = m(torch.as_tensor(Xv[j:j + 4096]).to(DEV))
        for nm in chunks: chunks[nm].append(torch.sigmoid(o[nm][:, 0]).cpu())
    for nm in chunks: probs[nm] = torch.cat(chunks[nm]).numpy()

report = {"note": "v2 model_ema re-scored under the v3.1 matched-filter gate (tau_hi); "
                  "ungated = original presence definition", "gates": GATES, "per_head": {}}
for nm in ("human", "vehicle", "animal"):
    lvl = val[f"{nm}_level"].to_numpy(); snr = val[f"{nm}_snr"].to_numpy(float)
    y = (pd.Series(lvl).map({"none": 0, "single": 1, "multiple": 2}).fillna(pd.Series(lvl)).to_numpy() > 0) \
        if lvl.dtype == object else (lvl > 0)
    keep = (~y) | (snr >= GATES[nm]["tau_hi_db"])
    r = {"AUROC_ungated": float(roc_auc_score(y, probs[nm])),
         "AUROC_gated": float(roc_auc_score(y[keep], probs[nm][keep])),
         "n_val": int(len(y)), "n_kept": int(keep.sum()),
         "pos_ungated": int(y.sum()), "pos_gated": int(y[keep].sum())}
    report["per_head"][nm] = r
    print(f"{nm:8s} ungated {r['AUROC_ungated']:.4f} | gated {r['AUROC_gated']:.4f} "
          f"| positives {r['pos_ungated']:,} -> {r['pos_gated']:,} (kept {r['n_kept']:,}/{r['n_val']:,})")
report["mean_gated"] = float(np.mean([report["per_head"][n]["AUROC_gated"] for n in report["per_head"]]))
report["mean_ungated"] = float(np.mean([report["per_head"][n]["AUROC_ungated"] for n in report["per_head"]]))
json.dump(report, open(os.path.join(OUT, "GATED_RESCORE.json"), "w"), indent=1)
print(f"mean: ungated {report['mean_ungated']:.4f} | gated {report['mean_gated']:.4f}")
print("wrote", os.path.join(OUT, "GATED_RESCORE.json"))
