# GeoSense — run on the laptop

Self-contained: the `webui/` folder holds the front-end **and** the Python backend
(model + feature code + sample data are vendored under `server/`). The laptop only
needs **Python 3.11+** and internet for the first install. No Node required.

## Run

Double-click **`start.bat`** (Windows). First run creates a `.venv`, installs the
deps (CPU-only PyTorch), then launches and opens `http://localhost:8000/`.

Manual equivalent:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Using it
Three signal sources (toggle, top-right). **Simulated is the default**; Dataset and
Geophone are complementary.
- **Simulated** *(default)* — the synthetic 2.5D demo (runs entirely in the browser).
  Shows the scene's ground truth alongside the prediction.
- **Dataset** — live-generates **fresh, never-trained** v3 scenes on the backend (seeds
  ≥9,000,000, disjoint from the training corpus), runs them through the real model, and
  shows **prediction vs ground truth**. Fully self-contained: the generator + signal
  banks + noise atlas are vendored under `server/` (`simgeo_v3/`, `simgeo_banks/`,
  `datasets/noise_atlas/`). The button auto-disables if those aren't present.
- **Geophone** — connects to the live sensor and runs detection in real time. A real
  signal has no known truth, so the ground-truth chip is hidden and the field view is a
  **blind sensor** (no terrain, generic range rings).
  - Auto-detects the CH340 device on any COM port (VID/PID, then description, then a
    content probe). Plug it in before launching.
  - On a quiet floor it reads **"All clear"** — tap / walk near the sensor to trigger a
    detection. If no device is found, it automatically **replays a recorded clip**
    (`server/data/*.csv`, vendored) so detections still show.

## Model
Detection uses the **synthetic-pretrained 132-feature SNN** (`server/model/`), CPU only.
Geophone signals are amplitude-aligned to the model's domain (×25.4) before features.

## Copying to the laptop
Copy the whole `webui/` folder (~230 MB; most of it is the vendored Dataset signal
banks under `server/simgeo_banks/`). You can **exclude** `node_modules/`,
`package*.json`, and `.venv/` (dev/build-only; `start.bat` recreates `.venv`). If you
don't need Dataset mode, you may also drop `server/simgeo_v3/`, `server/simgeo_banks/`,
and `server/datasets/` (~165 MB) — Simulated + Geophone still work and the Dataset
button auto-disables.
