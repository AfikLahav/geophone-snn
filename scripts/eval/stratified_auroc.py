"""SNR-stratified AUROC: per class, AUROC of (positives in an SNR bin) vs (all negatives).
Resolves whether a gated-AUROC gap between two models is model quality or population
composition (v2's positives are almost all >+10 dB; v3's cluster near the gate).

Runs both: v2 model on features_v2 val, and the GEO_OUT model on GEO_FEAT_DIR val.
Writes <GEO_OUT>/STRATIFIED_AUROC.json.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T = 4
BINS = [(-15, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 40)]


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


def run(out_dir, feat_dir, tag):
    scj = json.load(open(os.path.join(out_dir, "scaler.json"))); FEATURES = scj["features"]
    mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
    cols = FEATURES + ["split", "human_level", "vehicle_level", "animal_level",
                       "human_snr", "vehicle_snr", "animal_snr"]
    df = pd.concat([pd.read_parquet(s, columns=cols)
                    for s in sorted(glob.glob(os.path.join(feat_dir, "features_shard_*.parquet")))], ignore_index=True)
    val = df[df["split"] == "val"].reset_index(drop=True); del df
    Xv = np.clip((val[FEATURES].to_numpy(np.float32) - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
    m = FeatureSNN(len(FEATURES)).to(DEV); functional.set_step_mode(m, "m")
    m.load_state_dict(torch.load(os.path.join(out_dir, "model_ema.pt"), map_location=DEV)); m.eval()
    probs = {}
    with torch.no_grad():
        ch = {nm: [] for nm in ("human", "vehicle", "animal")}
        for j in range(0, len(Xv), 4096):
            functional.reset_net(m); o = m(torch.as_tensor(Xv[j:j + 4096]).to(DEV))
            for nm in ch: ch[nm].append(torch.sigmoid(o[nm][:, 0]).cpu())
        for nm in ch: probs[nm] = torch.cat(ch[nm]).numpy()
    res = {}
    for nm in ("human", "vehicle", "animal"):
        lvl = val[f"{nm}_level"].to_numpy()
        y = (pd.Series(lvl).map({"none": 0, "single": 1, "multiple": 2}).fillna(pd.Series(lvl)).to_numpy() > 0) \
            if lvl.dtype == object else (lvl > 0)
        snr = val[f"{nm}_snr"].to_numpy(float)
        neg = ~y; pneg = probs[nm][neg]
        rows = {}
        for lo, hi in BINS:
            mpos = y & (snr >= lo) & (snr < hi); n = int(mpos.sum())
            if n < 100: rows[f"[{lo},{hi})"] = {"n_pos": n, "auroc": None}; continue
            yy = np.r_[np.ones(n), np.zeros(len(pneg))]
            pp = np.r_[probs[nm][mpos], pneg]
            rows[f"[{lo},{hi})"] = {"n_pos": n, "auroc": round(float(roc_auc_score(yy, pp)), 4)}
        res[nm] = rows
        print(f"{tag} {nm:8s} " + " | ".join(f"{b}:{v['auroc']}({v['n_pos']})" for b, v in rows.items()))
    return res


report = {"note": "AUROC per positive-SNR bin vs all negatives; None where n_pos<100",
          "v2": run(os.path.join(ROOT, "snn_v2_out"), r"G:/geophone_synth/features_v2", "v2  "),
          "v31": run(os.environ.get("GEO_OUT", os.path.join(ROOT, "snn_v3_1_out")),
                     os.environ.get("GEO_FEAT_DIR", r"G:/geophone_synth/features_v3"), "v3.1")}
outp = os.path.join(os.environ.get("GEO_OUT", os.path.join(ROOT, "snn_v3_1_out")), "STRATIFIED_AUROC.json")
json.dump(report, open(outp, "w"), indent=1)
print("wrote", outp)
