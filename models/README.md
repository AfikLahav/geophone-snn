# Published models

The repository includes three replicates of each SNN size at 1 kHz and three at 200 Hz. Each directory contains `model_ema.pt`, `config.json`, and `scaler.json`. The configuration records the model architecture; the scaler records the feature order and normalization used during training.

| Model | Parameters | 1 kHz directories | 200 Hz directories |
|---|---:|---|---|
| SNN-S | 4,770 | `U2_lowrank16_rep1..3` | `Z1_snn_S_rep1..3` |
| SNN-M | 54,098 | `BEST_54k_256x128_rep1..3` | `Z2_snn_M_rep1..3` |
| SNN-L | 206,419 | `S6_512x256x128_rep1..3` | `Z3_snn_L_rep1..3` |

Here, `rep1..3` denotes three separate directories. The 200 Hz models were trained with the corresponding bandwidth restriction. Match the feature extraction and sampling-rate treatment to the chosen model.

## Load the weights

Run from the repository root:

```python
import json
from pathlib import Path
import torch
from training.model import build

folder = Path("models/U2_lowrank16_rep1")
cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
net = build(cfg, 77)
net.load_state_dict(torch.load(folder / "model_ema.pt", map_location="cpu"))
net.eval()
```

Before inference, select the features in the order stored in `scaler.json`. Subtract its `mean`, divide by its `std`, and clip to the stored `clip` value in both directions. Do not fit a new scaler on the test recordings. Reset the spiking state between independent batches with `spikingjelly.activation_based.functional.reset_net(net)`.

The weights alone do not specify a detection threshold. The field evaluation code selects thresholds from synthetic background windows and applies any temporal post-processing separately; see [evaluation](../evaluation/README.md).
