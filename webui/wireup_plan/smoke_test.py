"""M0 smoke test — prove the inference path end-to-end, no new infrastructure.

Loads the SNN checkpoint + feature scaler, featurizes recorded real geophone
CSVs through simgeo/features.py, runs per-window inference, and prints per-head
probabilities so we can confirm the whole chain (waveform -> 104 features ->
SNN -> per-head presence) works before building any backend.

When the fine-tuned model is ready, point this at it:
    python webui/wireup_plan/smoke_test.py --ckpt path/to/model_finetuned.pt \
        --scaler path/to/scaler_real.json

Run from anywhere; paths resolve to the project root.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo"))
import features as F  # noqa: E402

OUT = os.path.join(ROOT, "snn_v2_out")
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
DEV = "cpu"
T = 4
SCENE = 30 * int(F.FS)   # 30 s
HOP = 1500               # 1.5 s, matches training NW=3000/HOP=1500

# true class per recorded file (real set has no animal)
REALS = {
    "human.csv": "human", "car.csv": "vehicle",
    "human_nothing.csv": "nothing", "car_nothing.csv": "nothing",
}

# --- model: copied verbatim from rescale_test.py so the state_dict matches ----
class SeqBN(nn.Module):
    def __init__(s, c):
        super().__init__(); s.bn = nn.BatchNorm1d(c)
    def forward(s, x):
        T_, B_, C_ = x.shape
        return s.bn(x.reshape(T_ * B_, C_)).reshape(T_, B_, C_)


class FeatureSNN(nn.Module):
    def __init__(s, nf, w=(512, 256, 128), dp=0.1):
        super().__init__(); s.gate = nn.Parameter(torch.ones(nf)); b = []; d = nf
        for ww in w:
            b += [layer.Linear(d, ww), SeqBN(ww),
                  neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(2.0),
                                           detach_reset=True, step_mode="m"),
                  layer.Dropout(dp)]
            d = ww
        s.body = nn.Sequential(*b)

        def h(o):
            return nn.Sequential(layer.Linear(d, o),
                                 neuron.LIFNode(v_threshold=float("inf"),
                                                surrogate_function=surrogate.ATan(),
                                                step_mode="m", store_v_seq=True, backend="torch"))
        s.human, s.animal, s.vehicle = h(2), h(2), h(1)

    def forward(s, x):
        xs = (x * s.gate).unsqueeze(0).repeat(T, 1, 1)
        hh = s.body(xs)
        for _, hd in (("human", s.human), ("animal", s.animal), ("vehicle", s.vehicle)):
            hd(hh)


@torch.no_grad()
def infer(net, X):
    out = {"human": [], "animal": [], "vehicle": []}
    for j in range(0, len(X), 4096):
        functional.reset_net(net)
        net(torch.as_tensor(X[j:j + 4096]).to(DEV))
        for nm in out:
            out[nm].append(torch.sigmoid(getattr(net, nm)[-1].v_seq[-1][:, 0]).cpu())
    return {k: np.array(torch.cat(v)) for k, v in out.items()}


def featurize(path, scale, max_seg):
    a = pd.read_csv(path)["amplitude"].to_numpy(np.float32) * scale
    fe = []
    for si, c0 in enumerate(range(0, len(a), SCENE)):
        if max_seg and si >= max_seg:
            break
        seg = a[c0:c0 + SCENE]
        if len(seg) < F.NW:
            continue
        pre = F.scene_precompute(seg)
        for i0 in range(0, len(seg) - F.NW + 1, HOP):
            fe.append(F.window_features(pre, i0).astype(np.float32))
    return np.nan_to_num(np.stack(fe)) if fe else np.empty((0, F.NFEAT), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "model_ema.pt"))
    ap.add_argument("--scaler", default=os.path.join(OUT, "scaler.json"))
    ap.add_argument("--scale", type=float, default=25.4, help="real volts -> synthetic mV (rescale_test=25.4)")
    ap.add_argument("--max-seg", type=int, default=6, help="30 s segments per file (0=all)")
    ap.add_argument("--csv", default=None, help="single CSV instead of the 4 real files")
    args = ap.parse_args()

    print("=" * 72)
    print(f"M0 smoke test  |  device={DEV}  torch={torch.__version__}")
    print(f"  ckpt   : {args.ckpt}")
    print(f"  scaler : {args.scaler}")
    print(f"  scale  : x{args.scale}   max_seg: {args.max_seg or 'all'}")
    print("=" * 72)

    scj = json.load(open(args.scaler))
    FEATURES = scj["features"]; NF = len(FEATURES)
    mu = np.array(scj["mean"], np.float32); sd = np.array(scj["std"], np.float32); CLIPZ = scj["clip"]
    fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES]  # 104 selected columns
    TH = json.load(open(os.path.join(OUT, "ERROR_ANALYSIS.json"))).get("thresholds", {})
    th_h, th_v = TH.get("human", 0.5), TH.get("vehicle", 0.5)
    print(f"  features={NF}  clip=+/-{CLIPZ}  thresholds human={th_h} vehicle={th_v}")

    net = FeatureSNN(NF)
    functional.set_step_mode(net, "m")
    net.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    net.eval()
    print(f"  model loaded: {sum(p.numel() for p in net.parameters()):,} params\n")

    files = ([(os.path.basename(args.csv), "?", args.csv)] if args.csv
             else [(fn, cls, os.path.join(REAL_DIR, fn)) for fn, cls in REALS.items()])

    hdr = f"{'file':>18} {'true':>8} {'n':>5} {'P(human)':>9} {'P(car)':>8} {'P(animal)':>10} {'predicted':>10}"
    print(hdr); print("-" * len(hdr))
    ok = True
    for fn, cls, path in files:
        if not os.path.exists(path):
            print(f"{fn:>18}  MISSING"); ok = False; continue
        X = featurize(path, args.scale, args.max_seg)
        if len(X) == 0:
            print(f"{fn:>18}  no windows"); ok = False; continue
        Xz = np.clip((X[:, fidx] - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)
        if not np.isfinite(Xz).all():
            print(f"{fn:>18}  non-finite features!"); ok = False; continue
        p = infer(net, Xz)
        mh, mv = p["human"] - th_h, p["vehicle"] - th_v
        pred = np.where((mh < 0) & (mv < 0), "nothing", np.where(mh >= mv, "human", "vehicle"))
        from collections import Counter
        dom = Counter(pred).most_common(1)[0][0]
        print(f"{fn:>18} {cls:>8} {len(X):>5} {p['human'].mean():>9.3f} "
              f"{p['vehicle'].mean():>8.3f} {p['animal'].mean():>10.3f} {dom:>10}")

    print("\n" + ("PASS - inference path works end-to-end." if ok else "FAIL - see above."))
    print("(model_ema.pt is the synthetic-pretrained 0.859 zero-shot model; numbers are a\n"
          " sanity check of the PIPELINE, not a validated accuracy claim.)")


if __name__ == "__main__":
    main()
