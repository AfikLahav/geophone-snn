"""Verify the public release tree without running generation or training."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
errors = []


def require(path: str) -> Path:
    p = ROOT / path
    if not p.exists():
        errors.append(f"missing: {path}")
    return p


required = [
    "LICENSE", "README.md", "THIRD_PARTY.md", "requirements.txt",
    "datasets/background_model/r3_fits.npz", "terrain_models/library_v3.json",
    "terrain_models/splits_v3.json", "docs/FEATURE_CATALOG.html",
    "simgeo/simgeo_v42/generate_dataset.py", "simgeo/simgeo_v42/generate_dataset_v4.py",
    "simgeo/simgeo_v42/label.py", "simgeo/simgeo_v42/label_windows.py",
    "simgeo/simgeo_v42/extract_features_v4.py", "training/train.py",
    "training/evaluate.py", "results/report_metrics.json",
]
for item in required:
    require(item)

for p in ROOT.rglob("*.py"):
    try:
        ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
    except Exception as exc:
        errors.append(f"Python parse error: {p.relative_to(ROOT)}: {exc}")

for p in ROOT.rglob("*.json"):
    try:
        json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        errors.append(f"JSON parse error: {p.relative_to(ROOT)}: {exc}")

library_path = ROOT / "terrain_models/library_v3.json"
if library_path.exists():
    library = json.loads(library_path.read_text(encoding="utf-8"))
    non_modal = sum(not x.get("modal") for x in library)
    modal = sum(bool(x.get("modal")) for x in library)
    if (len(library), non_modal, modal) != (358, 340, 18):
        errors.append(f"terrain library counts are {len(library)}/{non_modal}/{modal}, expected 358/340/18")

config_paths = ROOT / "simgeo/simgeo_v42/config_paths.py"
if config_paths.exists():
    text = config_paths.read_text(encoding="utf-8")
    for expected in ("dataset_v431", "labels_v431", "features_v431"):
        if expected not in text:
            errors.append(f"config_paths.py does not contain {expected}")
    if "features_v431_3s" in text:
        errors.append("config_paths.py would duplicate the window suffix")

generator = ROOT / "simgeo/simgeo_v42/generate_dataset_v4.py"
if generator.exists() and "else 171700" not in generator.read_text(encoding="utf-8"):
    errors.append("the generator default is not 171,700 scenes")

families = [
    ("U2_lowrank16", 4770), ("BEST_54k_256x128", 54098),
    ("S6_512x256x128", 206419), ("Z1_snn_S", 4770),
    ("Z2_snn_M", 54098), ("Z3_snn_L", 206419),
]
for prefix, expected_params in families:
    for rep in (1, 2, 3):
        d = ROOT / "models" / f"{prefix}_rep{rep}"
        for name in ("model_ema.pt", "scaler.json", "config.json"):
            require(str((d / name).relative_to(ROOT)))
        cfg_path = d / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg.get("params") != expected_params:
                errors.append(f"{d.name}/config.json has params={cfg.get('params')}, expected {expected_params}")

for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() == ".pyc":
        errors.append(f"compiled Python file present: {p.relative_to(ROOT)}")
    if p.is_file() and p.stat().st_size >= 90 * 1024 * 1024:
        errors.append(f"file may exceed the normal GitHub limit: {p.relative_to(ROOT)}")

drive_pattern = re.compile(r"[A-Z]:[\\/][A-Za-z0-9_$][A-Za-z0-9_$ .-]*[\\/]")
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in {".py", ".md", ".txt", ".yaml", ".yml", ".csv"}:
        continue
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        continue
    if drive_pattern.search(text):
        errors.append(f"machine-specific drive path: {p.relative_to(ROOT)}")

if errors:
    print("RELEASE CHECK FAILED")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

python_count = sum(1 for _ in ROOT.rglob("*.py"))
json_count = sum(1 for _ in ROOT.rglob("*.json"))
model_count = sum(1 for p in (ROOT / "models").glob("*/model_ema.pt"))
print(f"RELEASE CHECK PASSED: {python_count} Python files, {json_count} JSON files, {model_count} checkpoints")
