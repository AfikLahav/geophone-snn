"""Shared inference core for the GeoSense backend.

Loads the synthetic-pretrained 132-feature SNN + scaler and turns a raw 1 kHz
waveform into per-window detections, in the exact shape the web UI consumes.

Caller scale convention (real -> synthetic amplitude alignment):
  - simulated scenes : waveform already in synthetic mV   -> scale = 1.0
  - geophone (real)  : waveform in volts                  -> scale = 25.4
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # vendored copy of simgeo/features.py

T = 4
HOP = 1500                       # 1.5 s, matches training NW=3000 / HOP=1500
HEADS = ("human", "animal", "vehicle")
UI_CLASS = {"human": "human", "vehicle": "car", "animal": "animal"}   # vehicle -> "car" in UI


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


class Detector:
    """Loads one model + scaler; featurizes and classifies waveforms."""
    def __init__(self, model_dir=None):
        model_dir = model_dir or os.path.join(HERE, "model")
        scj = json.load(open(os.path.join(model_dir, "scaler.json")))
        self.feats = scj["features"]
        self.mu = np.array(scj["mean"], np.float32)
        self.sd = np.array(scj["std"], np.float32)
        self.clip = float(scj.get("clip", 8.0))
        self.fidx = [F.FEATURE_NAMES.index(f) for f in self.feats]
        tp = os.path.join(model_dir, "thresholds.json")
        self.thr = json.load(open(tp)) if os.path.exists(tp) else {h: 0.5 for h in HEADS}
        self.net = FeatureSNN(len(self.feats)); functional.set_step_mode(self.net, "m")
        self.net.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
        self.net.eval()

    def featurize(self, wav):
        """wav: 1-D float32 already scaled to synthetic mV. -> [N, nfeat] z-scored."""
        wav = np.asarray(wav, np.float32)
        if len(wav) < F.NW:
            return np.empty((0, len(self.feats)), np.float32)
        pre = F.scene_precompute(wav)
        rows = [F.window_features(pre, i0).astype(np.float32)
                for i0 in range(0, len(wav) - F.NW + 1, HOP)]
        X = np.nan_to_num(np.stack(rows))
        return np.clip((X[:, self.fidx] - self.mu) / self.sd, -self.clip, self.clip).astype(np.float32)

    @torch.no_grad()
    def classify(self, X):
        """[N, nfeat] -> per-window dicts {probs, detected, levels}.
        human/animal heads are ordinal (2 logits): logit0 = present, logit1 = >= multiple.
        vehicle is binary (1 logit) -> always 'single' when present."""
        out = []
        for j in range(0, len(X), 4096):
            functional.reset_net(self.net)
            o = self.net(torch.as_tensor(X[j:j + 4096]))
            p = {h: torch.sigmoid(o[h][:, 0]).numpy() for h in HEADS}                      # presence
            pm = {h: (torch.sigmoid(o[h][:, 1]).numpy() if o[h].shape[1] > 1 else None)
                  for h in HEADS}                                                           # >= multiple
            for i in range(len(p["human"])):
                probs = {UI_CLASS[h]: float(p[h][i]) for h in HEADS}
                probs["nothing"] = float(max(0.0, 1.0 - max(probs.values())))
                detected = [UI_CLASS[h] for h in HEADS if p[h][i] >= self.thr.get(h, 0.5)]
                levels = {UI_CLASS[h]: ("multiple" if (pm[h] is not None and pm[h][i] >= 0.5) else "single")
                          for h in HEADS}
                out.append({"probs": probs, "detected": detected, "levels": levels})
        return out

    def run_waveform(self, wav, scale=1.0):
        """Convenience: raw waveform -> per-window detections (caller sets scale)."""
        return self.classify(self.featurize(np.asarray(wav, np.float32) * scale))


# ---- self-test: geophone path (real CSV -> x25.4 -> 132 model) ----------------
if __name__ == "__main__":
    import pandas as pd
    det = Detector()
    print(f"loaded model: {len(det.feats)} features, thresholds {det.thr}")
    reals = {"human.csv": "human", "car.csv": "car",
             "human_nothing.csv": "nothing", "car_nothing.csv": "nothing"}
    rdir = os.path.join(HERE, "data")
    print(f"{'file':>18} {'true':>8} {'win':>4} {'P(human)':>9} {'P(car)':>8} {'P(animal)':>10} {'mostly':>8}")
    for fn, cls in reals.items():
        a = pd.read_csv(os.path.join(rdir, fn))["amplitude"].to_numpy(np.float32)[:30000 * 6]
        res = det.run_waveform(a, scale=25.4)
        if not res:
            print(f"{fn:>18}  no windows"); continue
        mh = np.mean([r["probs"]["human"] for r in res]); mc = np.mean([r["probs"]["car"] for r in res])
        ma = np.mean([r["probs"]["animal"] for r in res])
        from collections import Counter
        dom = Counter(d for r in res for d in (r["detected"] or ["nothing"])).most_common(1)[0][0]
        print(f"{fn:>18} {cls:>8} {len(res):>4} {mh:>9.3f} {mc:>8.3f} {ma:>10.3f} {dom:>8}")
    print("OK - inference core runs end-to-end.")
