"""Single source of truth for data locations — v4.3.1.
Override with GEO_SYNTH_ROOT env var for alternate machines."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))

SYNTH_ROOT = os.environ.get("GEO_SYNTH_ROOT", os.path.join(ROOT, "..", "..", "geophone_synth"))

CFG_DIR = os.path.join(SYNTH_ROOT, "config")
DATASET_DIR = os.path.join(SYNTH_ROOT, "dataset_v431")
LABELS_DIR = os.path.join(SYNTH_ROOT, "labels_v431")
FEATURES_DIR = os.path.join(SYNTH_ROOT, "features_v431")
