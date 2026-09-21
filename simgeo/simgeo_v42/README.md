# Synthetic dataset generator

This directory contains the v4.3.1 pipeline. The directory name is retained for existing imports. Source forces are propagated through a ground profile, combined with background motion, and passed through a simulated 4.5 Hz geophone and recording chain. The resulting windows are labeled and converted into features.

## Pipeline

Run the commands in the [root README](../../README.md) in this order:

| Script | Output |
|---|---|
| [gfbank_build.py](gfbank_build.py) | Precomputed ground responses in `terrain_models/*.npz` |
| [snr_maps.py](snr_maps.py) | Signal-to-noise ratio versus distance maps and source limits in `GEO_SYNTH_ROOT/config` |
| [generate_dataset_v4.py](generate_dataset_v4.py) | Scenes in `GEO_SYNTH_ROOT/dataset_v431` |
| [label_windows.py](label_windows.py) | Three-second labels in `GEO_SYNTH_ROOT/labels_v431/windows_3s` |
| [extract_features_v4.py](extract_features_v4.py) | 132-feature tables in `GEO_SYNTH_ROOT/features_v431_3s` |

The ground responses are computed with pyprop8. Reusing them lets many source signals and paths share the same propagation calculation. The included library contains 340 ground profiles and 18 modal floor profiles; generated response banks are not stored in Git.

## Main modules

[sources_v2.py](sources_v2.py) defines people, vehicles, animals, and other scene sources. [paths.py](paths.py) controls movement, and [scenes.py](scenes.py) combines the sources with the ground response. [r3_noise.py](r3_noise.py) generates background motion from the included [background statistics](../../datasets/background_model/README.md).

[sensor.py](sensor.py) converts ground velocity in metres per second into the recorded signal in millivolts. Source and noise components remain separate for signal-to-noise calculations. Clipping and optional quantization are applied to their sum when reconstructing the model input. The current v4.3 rendering path applies the geophone response without the older coupling and high-frequency resonance filters.

[label_windows.py](label_windows.py) uses three-second windows with a 1.5-second hop by default. Overlapping windows share samples, so randomly splitting them between training and testing can leak recording information. Preserve the intended scene and terrain splits when generating training tables.

See the [datasheet](../../dataset_validation/DATASHEET_v431.md) for recorded sampling distributions and the settings needed to interpret the v4.3.1 validation records.
