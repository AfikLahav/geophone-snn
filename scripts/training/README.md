# scripts/training/ — model training entry points

- `geophone_snn_v2_train.py` — the main SNN pretrain script (env-driven: `GEO_FEAT_DIR`, `GEO_OUT`, …).
- `finetune_5fold.py` — 5-fold synth-pretrained → real fine-tune (writes deploy artifacts).
- `real_baseline.py`, `snn_real_baseline.py` — real-only baselines (classical + same-architecture SNN).
- `eval_v3_compare.py` — the v3 A/B/C comparison (also trains the real-only model).
- `optuna_sweep.py` — hyperparameter search (imported by `build_optuna_stats.py`).
- `train_cdf.py`, `train_focal.py` — loss/label variants (CDF soft-labels, focal + oversampling).

Most read external data via `GEO_DB_ROOT` / `Goephone-Project` — not turnkey from a clean
checkout without that data. Outputs land in [`../../models/`](../../models/).
