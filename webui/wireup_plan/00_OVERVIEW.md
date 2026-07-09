# Geophone → SNN → Web UI — Wire-up Plan (Overview)

**Status: PLAN ONLY. Nothing is implemented.** Date: 2026-06-16.

Goal: replace the web UI's in-browser simulated backend (`webui/src/sim.js`) with a real
pipeline — live geophone (and, optionally, synthetic) signal → 104-feature extraction →
fine-tuned SNN → per-window detection streamed to the existing renderers.

## Documents in this folder

| Doc | Scope |
|---|---|
| [10_model_inference.md](10_model_inference.md) | Load & run the fine-tuned SNN; head decode |
| [20_feature_pipeline.md](20_feature_pipeline.md) | Waveform → 104 z-scored features |
| [30_geophone_live.md](30_geophone_live.md) | Live serial acquisition → rolling 3 s windows |
| [40_backend_integration.md](40_backend_integration.md) | FastAPI + WebSocket service; UI data contract |

## What exists vs. what must be built

| Component | Exists? | Action |
|---|---|---|
| `FeatureSNN` architecture (`geophone_snn_v2_train.py`) | ✅ | reuse |
| Synthetic‑pretrained weights `snn_v2_out/model_ema.pt` | ✅ | = **0.859 zero‑shot**, not the fine‑tuned model |
| **Fine‑tuned checkpoint** | ❌ | **must produce** (see D1) |
| Feature extractor `simgeo/features.py` | ✅ | reuse (`scene_precompute` + `window_features`) |
| Feature scaler `snn_v2_out/scaler.json` (104 feats) | ✅ | reuse (synthetic‑fit — see D3) |
| **Live rolling‑buffer reader** | ❌ | **must build** (all current scripts are one‑shot) |
| **Backend service (FastAPI + WS)** | ❌ | **must build** |
| Web UI renderers | ✅ | reuse; add a thin WS client |

## Critical decisions (lock these before coding)

- **D1 — Produce the fine‑tuned artifact.** The headline **0.977 does not exist on disk**:
  `finetune_5fold.py` computes it *inside* a 5‑fold CV loop over 25 throwaway models and never
  `torch.save`s. Recipe (per 10_): re‑run `finetune()` on **all** real data (no CV split),
  fit **one** deployment `StandardScaler`, save `model_finetuned.pt` + `scaler_real.json` +
  a thresholds sidecar. (Ensemble‑of‑5 alt documented if we want the exact number.)
- **D2 — Animal head is unvalidated on real data** (no real animal recordings; FT supervises
  only human+vehicle). Recommend **grey/disable the Animal class in geophone mode**.
- **D3 — Amplitude calibration is THE accuracy risk.** Live samples are **volts** (`mV/1000`);
  the model expects synthetic‑mV scale via **×25.4 applied at the window boundary** — but 25.4 is
  unvalidated (`rescale_test.py` probes `{1.0, 25.4, 90.0}`) and the scaler was fit on synthetic
  data (real RMS is class‑dependently ~1–4 % of synthetic). A calibration step must be settled
  before any live number is shown.
- **D4 — Heads = 3 independent sigmoids `{human, animal, vehicle}`; there is no `nothing` head.**
  Map `vehicle→car`; `nothing = all heads below threshold` (UI already shows independent bars —
  **do not softmax**). Per‑head thresholds from `ERROR_ANALYSIS.json` (human 0.15 / vehicle 0.05 /
  animal 0.10) or recalibrate for the shipped decode.
- **D5 — Decode = last membrane step** `sigmoid(head[-1].v_seq[-1][:,0])` (matches the saved
  thresholds), not training's `v_seq.mean(0)`. Pick one and calibrate thresholds to it.
- **D6 — Two checkpoints likely:** synthetic `model_ema.pt` for **`sim` mode**, fine‑tuned +
  real‑fitted scaler for **`geo` mode** (synthetic↔real are perfectly separable, domain AUC = 1.0).
- **D7 — Inference on torch/CPU backend** (cupy is training‑only). A few ms/window — comfortably
  real‑time at the 1.5–3 s cadence; no GPU on the demo box.

## Recommended build order (geophone‑first)

- **M0 — Inference smoke test (pure Python, no infra).** Load `model_ema.pt` + `scaler.json`,
  featurize one recorded CSV via `features.py`, print per‑head probs. Proves the whole path.
- **M1 — Produce the fine‑tuned checkpoint (D1).** → `model_finetuned.pt` + `scaler_real.json` +
  thresholds.
- **M2 — Reusable inference module.** `featurize_segment(waveform)→[N,104]` + `load_model()` +
  `predict()` → a `window` dict matching the UI contract.
- **M3 — Backend + CSV‑replay → UI end‑to‑end (PRIMARY GOAL).** FastAPI + WebSocket `/stream`;
  replay `Goephone-Project/geophone_data/*.csv` through M2; UI **Geophone** branch lights up live.
- **M4 — Live serial.** New `GeophoneStream` (deque maxlen=3000, daemon thread, emit every 1500
  samples) replaces replay; CH340 @ 460800, volts → ×25.4.
- **M5 — (optional) Synthetic on the backend.** simgeo is ~50–100 ms/window (≥15× headroom), so
  `sim` mode could also be backend‑served; or keep the JS `sim.js` for geometry and only wire
  geophone to the backend (recommended hybrid).

## UI data contract (one WebSocket `window` message per hop)

Mapped to what `histograms.js` / `detection.js` / `signal.js` / `scene.js` already consume:
`{ win, probs{human, car, animal, nothing}, detected[], signal{rms, db, pct}, snr,
   scene{…geometry…} (sim mode only) }`. Full schema in 40_.

## Publication guardrails (rigor)

- **Do not present 0.977 as field accuracy** in the demo. It is session‑confounded (1 session/
  class; confound‑clean ≈ 0.937) and the synthetic→real domain gap is total (domain AUC = 1.0).
  The live demo shows *behavior*, not a validated accuracy claim.
- Amplitude calibration (×25.4) and the Animal class on real data are both **unvalidated**.
- The current quiet floor is quantization‑limited → live geophone may read mostly "nothing";
  demo with a real footstep/tap near the sensor, or use CSV replay.

## Consolidated open questions

1. Fine‑tune on all‑real (single model) vs persist the 5‑member ensemble for the exact 0.977? (D1)
2. Animal class in geo mode — drop, or show as synthetic‑only/unvalidated? (D2)
3. Which amplitude calibration scale, and can we measure it for *this* sensor/floor? (D3)
4. Confirm last‑step vs mean decode and recalibrate thresholds. (D5)
5. One checkpoint or two (sim vs geo)? (D6)
6. Window on dropped‑sample count vs wall‑clock time (serial jitter)? (30_)
7. Cold‑start IIR transients + per‑window `scene_precompute` context shift — validate feature
   parity vs corpus windows. (20_)
