# scripts/ — training, evaluation & analysis tooling

Standalone scripts. (The *generation* pipeline lives in `simgeo/`, not here.)

- **`training/`** — model training entry points: `geophone_snn_v2_train` (main
  pretrain), `finetune_5fold`, `real_baseline`, `snn_real_baseline`, `optuna_sweep`
  (HPO), `train_cdf` / `train_focal` (loss variants), `eval_v3_compare`.
- **`eval/`** — synth→real transfer + model-metric evaluation: `eval_synth_to_real`,
  `stratified_auroc`, `exp_folds`, `detectability_curve`, `accuracy_*`, `error_analysis`,
  `terrain_match`, `domain_gap_v4`, … (the CDI *dataset* gates are separate, in `dataset_validation/`).
- **`analysis/`** — one-off diagnostics: dataset-gap, domain-feature audit, chunk
  sensitivity, missed-window analysis.
- **`plotting/`** — figure generation.

> **Note:** most of these hardcode external data paths (`GEO_DB_ROOT`,
> `Goephone-Project`). They're provenance/tools — not turnkey-runnable from a clean
> checkout without that data.
