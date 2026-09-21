# Training

The training package reads the synthetic feature tables and trains models to detect people and vehicles independently. Spiking models use SpikingJelly neurons; conventional models use the same layer widths with non-spiking units.

## Run

Set `GEO_SYNTH_ROOT` to the directory containing `features_v431_3s`, then run from the repository root:

```bash
python -m training.train --preset tiny
```

Use `--preset medium` or `--preset large` for the larger models. The conventional presets are `tiny_ann`, `medium_ann`, and `large_ann`. `tiny_person_only` has only a person output, but still sees vehicle scenes as negative examples.

The shared settings in [presets.yaml](presets.yaml) use 77 features, batches of 4,096, and 20,000 optimization steps. Spiking presets use four internal time steps. Training fits the feature scaler on the training split and evaluates on the validation split.

Each run saves `model_ema.pt`, `scaler.json`, `config.json`, and `history.json` under `training/out/<preset>`. The weights use an exponential moving average of the training weights. Set `--outdir` to keep separate runs. The default device is CUDA; use `--device cpu` if needed.

## Synthetic validation

```bash
python -m training.evaluate --checkpoint training/out/tiny/model_ema.pt
```

The evaluator reads the configuration beside the checkpoint and reports synthetic validation scores and neuron firing statistics. Its `--data` option expects compatible training and validation feature tables: this command refits the scaler on their training split. To reproduce a saved model's normalization, supply the original training data. For inference on new recordings, use the saved scaler described in [models](../models/README.md).

Field evaluation and synthetic-plus-real training are described in [evaluation](../evaluation/README.md). The published checkpoint configurations record the settings of those individual runs; the presets are starting configurations for new runs.
