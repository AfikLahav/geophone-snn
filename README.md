# Synthetic training for geophone source detection

This project studies whether models trained on simulated ground vibrations can detect people and vehicles in real recordings. The repository includes the v4.3.1 dataset generator, feature extraction, training code, selected spiking neural network (SNN) weights, and evaluation results. The simulated sensor is a vertical 4.5 Hz geophone.

The published models were trained on synthetic data. Separate experiments evaluate transfer to real recordings and further training with real data. Raw Elbit recordings are not included.

## Contents

| Directory | Contents |
|---|---|
| [simgeo/simgeo_v42](simgeo/simgeo_v42/README.md) | Ground propagation, source signals, sensor response, labels, and features |
| [terrain_models](terrain_models) | 340 ground profiles and 18 modal floor profiles |
| [Background model](datasets/background_model/README.md) | Fitted background statistics used by the generator |
| [dataset_validation](dataset_validation/README.md) | Validation scripts and saved dataset checks |
| [training](training/README.md) | Spiking and conventional models, training, and synthetic validation |
| [evaluation](evaluation/README.md) | Field evaluation, cross-validation, and detection post-processing |
| [models](models/README.md) | 18 checkpoints, each with its configuration and feature scaler |
| [results](results) | Aggregate report metrics and M3N-VC result tables |
| [Feature catalog](docs/FEATURE_CATALOG.html) | Definitions and formulas for all 132 features |

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux or macOS, or `.venv\Scripts\Activate.ps1` in PowerShell. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the appropriate CPU or CUDA build of [PyTorch](https://pytorch.org/get-started/locally/) separately.

## Generate the dataset

Set `GEO_SYNTH_ROOT` to the output directory. For example, in PowerShell:

```powershell
$env:GEO_SYNTH_ROOT = Join-Path $HOME "geophone_synth"
```

In bash, use `export GEO_SYNTH_ROOT=/path/to/geophone_synth`. Without this setting, the scripts use a sibling directory named `geophone_synth`.

Run the stages in order:

```bash
python simgeo/simgeo_v42/gfbank_build.py
python simgeo/simgeo_v42/snr_maps.py
python simgeo/simgeo_v42/generate_dataset_v4.py
python simgeo/simgeo_v42/label_windows.py
python simgeo/simgeo_v42/extract_features_v4.py
```

The generator defaults to 171,700 scenes. Labels use three-second windows with a 1.5-second hop, and feature extraction produces 132 values per window. Propagation banks are written to `terrain_models/`; scene, label, and feature files are written under `GEO_SYNTH_ROOT`. The full workflow needs hundreds of gigabytes of disk space.

The [v4.3.1 datasheet](dataset_validation/DATASHEET_v431.md) describes the saved dataset measurements and differences between the recorded generation settings and current defaults. These commands run the included pipeline; they have not been verified to recreate the original dataset byte for byte.

## Train a model

```bash
python -m training.train --preset tiny
python -m training.evaluate --checkpoint training/out/tiny/model_ema.pt
```

Both commands default to CUDA. Add `--device cpu` for CPU execution. Training reads `features_v431_3s` under `GEO_SYNTH_ROOT`.

| Model | Feature and hidden-layer widths | Parameters |
|---|---|---:|
| SNN-S (`tiny`) | 77 → 16 → 64 → 32 | 4,770 |
| SNN-M (`medium`) | 77 → 256 → 128 | 54,098 |
| SNN-L (`large`) | 77 → 512 → 256 → 128 | 206,419 |

The final models use features 0–76. Other classification tasks may benefit from the remaining features. The [training notes](training/README.md) describe the conventional twins and person-only preset; the [model notes](models/README.md) explain how to load the published weights.

## Check the repository

```bash
python -B scripts/verify_release.py
```

This checks Python and JSON syntax, required files, terrain counts, configured paths, and checkpoint metadata. It does not run generation or verify model predictions.

Original project code is covered by the [PolyForm Noncommercial License 1.0.0](LICENSE). External software and data retain their own terms; see [third-party sources](THIRD_PARTY.md).
