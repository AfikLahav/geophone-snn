# Field evaluation

These scripts compare model predictions with field recordings and test further training on real data. The repository includes selected weights and aggregate results. Raw recordings, cached features, and the full experiment databases are not included, so those analyses need their original inputs before they can be rerun.

The field-data loaders also need `h5py` for HDF5 recordings and `obspy` for seismic recording formats. Install them when using those loaders:

```bash
python -m pip install h5py obspy
```

## Data locations

Set the following variables to absolute paths as needed. Paths described as siblings are next to the repository.

| Variable | Purpose | Default |
|---|---|---|
| `GEO_SYNTH_ROOT` | Synthetic feature tables | Sibling `geophone_synth` |
| `GEO_REAL_ROOT` | External field datasets | Sibling `geophone_datasets` |
| `GEO_CAMPAIGN_OUT` | Experiment outputs and model directories | Repository `campaign_out` |
| `GEO_REAL_CACHE` | Cached field features | Script-dependent; set explicitly |
| `GEO_CAMPAIGN_DB` | Experiment database used by M3N-VC adaptation | Repository `campaign_out/campaign.sqlite` |
| `M3N_AUDIT_DB` | M3N-VC windows and GPS audit database | Repository `m3n_gps_audit.sqlite` |
| `M3N_OUT` | M3N-VC adaptation outputs | Repository `m3n_out` |

`prepare_real.py` defaults to `campaign/real_cache` inside the repository, while the M3N-VC adaptation script defaults to `real_cache`. Set `GEO_REAL_CACHE` to the same directory for both. M3N-VC adaptation looks for source weights in `GEO_CAMPAIGN_OUT/models/<run_id>/model_ema.pt` and uses the run records from the experiment database.

## Scripts

| Script | Purpose |
|---|---|
| [prepare_real.py](prepare_real.py) | Load field recordings and extract window features |
| [clean_elbit.py](clean_elbit.py) | Apply the Elbit label-cleaning rule |
| [svm_repro.py](svm_repro.py), [classical_field.py](classical_field.py) | Evaluate conventional classifiers and the effect of the data split |
| [snn_cv2.py](snn_cv2.py), [snn_cv.py](snn_cv.py) | Field cross-validation experiments |
| [hyst_select.py](hyst_select.py) | Select hysteresis exit thresholds on synthetic validation data |
| [postprocess.py](postprocess.py) | Apply hysteresis and repeated-window decisions |
| [second_tests.py](second_tests.py) | Evaluate external datasets, including SeisSavanna and M3N-VC |
| [m3n_positive_transfer.py](m3n_positive_transfer.py) | Train and evaluate M3N-VC adaptation with GPS-based distance groups |
| [summarize_m3n_training.py](summarize_m3n_training.py) | Summarize the saved M3N-VC adaptation results |
| [review.py](review.py), [rescore.py](rescore.py) | Review runs and score saved models |
| [configs.py](configs.py) | Define experiment configurations |

## Read the results

[results](../results) contains the report metrics and M3N-VC summaries by scene, terrain, and distance. Keep each result's evaluation conditions with its score: original or cleaned labels, window duration, split, threshold, and temporal post-processing. Synthetic-only transfer and training with real recordings answer different questions and must be reported separately.

M3N-VC vehicle-pass windows support detection-rate measurements. Distant windows are not established negative examples merely because a model does not detect them. Accuracy and precision require valid negative labels as well as positive labels.
