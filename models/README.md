# models/ — trained models & experiment results

One folder per training experiment. Each contains model weights (`*.pt`) and the
evaluation results (`*.json`) that produced the reported numbers.

| Folder | What |
|---|---|
| `snn_v4_out/` | v4 experiments (E0–E5, ET terrain-matching) — the current production line |
| `snn_v3_1_out/`, `snn_v3_out/` | v3 / v3.1 experiments |
| `snn_v2_out/` | v2 — the paper's working model |
| `snn_132_out/` | the 132-feature model line |

Typical contents per experiment:
- `model_raw.pt` / `model_ema.pt` / `model_finetuned.pt` — weights (raw, EMA-smoothed, real-fine-tuned).
- `scaler.json`, `thresholds.json` — the feature normalization + decision thresholds needed for inference.
- `*.json` — metrics (accuracy, AUROC, folds, detectability, transfer results).
- `DEPLOY_MANIFEST.json` — deployment artifact list where present.

Weights here are the *outputs* of the training scripts in [`../scripts/training/`](../scripts/training/);
the specs are in [`../docs/`](../docs/). Large corpora/features are external (see `docs/DATA_LOCATIONS.md`).
