# 40 — Backend Integration

How the trained SNN geophone model + live geophone connect to the existing web UI.

**Scope of this file:** the backend *service* and its contract with the browser. The model
internals, the feature pipeline, and the live-reader internals are covered (or will be) by the
sibling plans `10_model_inference.md`, `20_feature_pipeline.md`, `30_geophone_live.md`. Where
those files don't exist yet, the relevant facts are restated here so this plan stands alone.

> Status: PLAN ONLY. No code is written or modified by this document.

---

## 0. Ground truth (what the backend must import)

All paths relative to project root:
`S:\ALL PROJECTS\geophone sensor\finals project\finals project`

| Component | Location | Key facts |
|---|---|---|
| Model class `FeatureSNN` | `geophone_snn_v2_train.py` (def ~L97–137) | SpikingJelly PLIF, `step_mode="m"`, T=4 |
| Weights | `snn_v2_out/model_ema.pt` | EMA checkpoint (load via `load_state_dict`) |
| Scaler / feature order | `snn_v2_out/scaler.json` | `{mean[104], std[104], clip:8.0, features:[...104...]}` |
| Feature extractor | `simgeo/features.py` | `FS=1000.0`, `NW=3000`, `scene_precompute(x)`, `window_features(pre,i0)` → 132 feats |
| Synthetic generator | `simgeo/` (`scenes.py`, `sources_v2.py`, `sensor.py`) | importable library; ~50–100 ms / 3 s window |
| Live reader | **does not exist yet** | build from `record15.py` serial code (CH340, 460800 baud) |

**Critical mismatches the backend must reconcile (details in §3):**

1. **Model heads ≠ UI classes.** Model outputs three *independent* sigmoid heads
   `{human, animal, vehicle}`. The UI consumes `{human, car, animal, nothing}`.
   → `vehicle → car`; `nothing` is *derived* (no head exists for it).
2. **Window cadence.** Model window = **3.0 s** with **1.5 s hop** (50% overlap). UI's `WINDOW_MS`
   constant is **2000 ms** (`main.js` L12). The real stream should push **every 1.5 s** (one per hop)
   and the UI's local timer becomes cosmetic in live mode (see §4).
3. **`probs` are not a simplex.** UI code treats `probs.{human,car,animal,nothing}` as independent
   0–1 bars (`histograms.js` thresholds each at 0.5). This matches the model's independent-head
   design — good. We must NOT softmax across classes.
4. **No geometry in live mode.** Real geophone gives no x/y/range/bearing. The UI already has a
   "blind sensor" view for `mode==='geo'` (`scene.js` L121–128 draws a sweeping bearing line and
   skips the source). So the geophone message omits geometry; simulated message includes it.

---

## 1. Backend choice & shape

### Recommendation: **FastAPI + uvicorn**, single Python process.

The model, feature pipeline, and simgeo generator are all Python (torch + SpikingJelly + scipy).
The backend must `import` them directly — there is no value in a non-Python service that then
shells out to Python. So the only real question is *which* Python web framework.

| Option | Verdict | Why |
|---|---|---|
| **FastAPI** | ✅ chosen | First-class native WebSocket support, async event loop, trivial CORS middleware, Pydantic schema validation (lets us *define and enforce* the JSON contract in §2), auto OpenAPI docs for debugging, runs under uvicorn with one command. |
| Flask + flask-sock | viable | Works, but WebSockets are bolt-on, no async-native streaming, no built-in schema validation. More glue. |
| Plain `websockets` lib | viable, minimal | No HTTP routes for health/config, no CORS helper, you hand-roll everything. Fine for a pure socket but we also want a couple of REST endpoints (`/health`, `/config`, mode switch). |
| Django Channels | ❌ | Far too heavy for a single-model demo service. |
| Node/Go service | ❌ | Would need a second process + IPC to reach the Python model. Pure overhead. |

### Process shape

```
┌────────────────────────── FastAPI process (uvicorn) ──────────────────────────┐
│                                                                                │
│  startup:  load model_ema.pt + scaler.json  ──►  model (eval, set_step_mode m) │
│            warm simgeo Bank cache (optional)                                    │
│                                                                                │
│  ┌─ Source layer ───────────────┐     ┌─ Inference core ──────────────────┐    │
│  │ SimSource   (simgeo on-demand)│ ──► │ scene_precompute + window_features │    │
│  │ GeoSource   (live serial RB)  │ ──► │   → 104 feats → z-score+clip       │    │
│  └──────────────────────────────┘     │   → reset_net → model → sigmoid    │    │
│                                        │   → hysteresis / N-of-M            │    │
│                                        └──────────────┬────────────────────┘    │
│                                                       │ per-window result        │
│  WS /stream  ◄────────────────────────────────────────┘                         │
│  REST /health /config                                                          │
└────────────────────────────────────────────────────────────────────────────────┘
                          │ ws://localhost:8000/stream
                          ▼
                   Browser (webui), src/main.js + a thin WS client
```

**Concurrency model.** One async producer task per active stream. The CPU-heavy work
(feature extraction ~per-window, model forward, and — in sim mode — simgeo generation) runs in a
**thread pool / `run_in_executor`** so it never blocks the event loop. Torch inference releases the
GIL during the heavy ops; SpikingJelly with `step_mode="m"` over T=4 is cheap (~ms). The pacing is
done by an `asyncio.sleep` to the next 1.5 s boundary, not by busy-waiting.

**Single-client assumption.** This is a demo/console for one operator. We design for one (or a few)
concurrent WS clients. The live geophone is a single hardware resource: the `GeoSource` reader is a
**singleton ring buffer**; multiple sockets in `geophone` mode subscribe to the *same* buffer rather
than each opening the serial port (which would fail — the port is exclusive).

---

## 2. Streaming contract

### Transport: **WebSocket** at `GET /stream` (upgrade).

Why WebSocket over SSE:
- We want **client → server** control too (switch mode, pause/resume, request "new scene"),
  matching the UI's existing buttons (`sourceToggle`, `playBtn`, `newBtn` in `index.html`).
  SSE is server→client only and would need a second channel for control.
- Binary upgrade path later (raw waveform frames) is available if we ever push the scope trace.

(If a client-control channel is deemed unnecessary, SSE at `/stream` is a drop-in fallback —
the message schema below is transport-agnostic JSON.)

### Cadence

One message per model window. Push **every ~1.5 s** in both modes (matches the 1.5 s hop). This sits
inside the UI's advertised "1.5–3 s" read cadence (`main.js` L12 comment). The server is the clock;
the browser stops driving `sampleWindow()` from its own timer when a socket is connected (see §4).

### Client → Server messages (control)

```jsonc
{ "type": "set_mode", "mode": "simulated" }      // or "geophone"
{ "type": "set_mode", "mode": "geophone" }
{ "type": "new_scene" }                            // simulated only; force a fresh scene
{ "type": "pause" }                                // stop emitting windows
{ "type": "resume" }
{ "type": "set_speed", "speed": 1.0 }              // simulated only; time-compression factor
```

### Server → Client message — the **window** message (one per ~1.5 s)

This is the load-bearing schema. It is deliberately shaped to drop into what the UI already consumes
(`Sim.sampleWindow()` return + `Sim.getState()` return). Field names mirror the JS.

```jsonc
{
  "type": "window",
  "mode": "simulated",          // "simulated" | "geophone"  → drives UI mode + status bar
  "win": 137,                   // monotonic window counter  → main.js updateReadouts (#stWin)

  // ── per-class probabilities (UI: histograms.js, detection.js) ────────────────
  // INDEPENDENT 0..1 values (NOT a simplex). Keys EXACTLY match the UI's ORDER array.
  "probs": {
    "human":   0.91,            // model "human" head, sigmoid(logit[:,0])
    "car":     0.04,            // model "vehicle" head  (vehicle → car, see §3)
    "animal":  0.12,            // model "animal" head
    "nothing": 0.09             // DERIVED = 1 - max(human,car,animal)  (see §3)
  },

  // ── which classes are "on" this window (UI: detection.js lamps + scene.pulse) ─
  // Already-resolved server-side using hysteresis / N-of-M, NOT a naive p>=0.5,
  // sorted by confidence desc. UI consumes verbatim (sim used probs>=0.5; we can
  // keep that fallback, but server-side hysteresis is more honest — see §3).
  "detected": ["human"],

  // ── signal strength (UI: scene-overlay meter via signal.getStrength) ─────────
  // RMS energy of the window → the log_rms feature, mapped to 0..1 + percent.
  "signal": {
    "rms":   0.043,             // linear RMS of the 3 s window (sensor units)
    "db":    -1.2,              // 20*log10(rms / ref)
    "pct":   62                 // 0..100, what #sigMeterFill / #sigStrength show
  },

  // ── SNR for the status bar (UI: #stSnr, shown only if hasSource && snr>-15) ──
  "snr": 18.0,                  // dB; for geophone use a best-effort estimate or null

  // ── scene geometry — SIMULATED MODE ONLY (UI: scene.js render path) ─────────
  // Omit / null entirely in geophone mode → UI 'geo' branch draws the blind view.
  "scene": {
    "hasSource": true,
    "kind": "human",            // null when quiet
    "x": -12.4, "y": 31.0, "r": 33.4,     // metres; sensor at origin
    "intensity": 0.88,          // detection envelope 0..1  (UI: source highlight, signal amp)
    "viewExtent": 67.2,         // metres; sets scene zoom (scene.js uses state.viewExtent)
    "bearing": 2.41,            // rad; blind-view sweep target (unused in sim path)
    "scene": {                  // matches sim's st.scene object
      "style": "pass-by",       // "pass-by" | "oblique" | "quiet"
      "Rdet": 60                // detection range → range rings in scene.js
    }
  },

  // ── terrain (UI: TERRAIN_NAME lookup, #sceneMeta) ───────────────────────────
  // Present in simulated mode. In geophone mode it's unknown → null/"unknown".
  "terrain": "soil",            // "soil" | "gravel" | "asphalt"  (sim vocab; see §3 mapping)
  "sceneId": 42
}
```

#### Other server → client messages

```jsonc
{ "type": "hello",   "mode": "simulated", "windowSec": 3.0, "hopSec": 1.5,
                     "classes": ["human","car","animal","nothing"] }    // on connect
{ "type": "status",  "running": true, "geophoneConnected": false }       // mode/health changes
{ "type": "error",   "code": "NO_CH340", "message": "geophone not found" } // serial absent etc.
```

`NO_CH340` is the exact failure string from `record15.py`'s `find()`. When the user toggles to
Geophone and no device is present, the server replies with this error; the UI should surface it
(e.g. flip the status bar to a fault state) instead of silently showing a dead stream.

### REST endpoints (non-streaming)

```
GET  /health     → {status:"ok", model:"model_ema.pt", classes:[...], device:"cpu"}
GET  /config     → {windowSec:3.0, hopSec:1.5, fs:1000, threshold:0.5, classes:[...]}
```

---

## 3. Model head → UI class mapping

The single most important translation layer. Source of truth: `geophone_snn_v2_train.py`
(forward returns `{"human":[B,2], "animal":[B,2], "vehicle":[B,1]}`) and `scaler.json`.

### 3.1 Class mapping

| UI class (`ORDER` in histograms.js) | Source | Rule |
|---|---|---|
| `human`   | model head `human`   | `p_human  = sigmoid(out["human"][:,0])` |
| `car`     | model head `vehicle` | `p_car     = sigmoid(out["vehicle"][:,0])`  **← vehicle renamed to car** |
| `animal`  | model head `animal`  | `p_animal = sigmoid(out["animal"][:,0])` |
| `nothing` | **derived**          | `p_nothing = 1 - max(p_human, p_car, p_animal)` (clamped 0..1) |

Notes:
- The model has **no `nothing` head** — "nothing implicit" is stated in the training file's own
  docstring. `nothing` is purely a UI convenience row. The derivation above mirrors the sim
  (`sim.js` sets `nothing ≈ 1 - env`). It is the OR-of-heads complement, consistent with the
  model's `score_any = max(sigmoid heads)` logic in training.
- The `[:,1]` ordinal columns of the human/animal heads (count/level) are **not surfaced** to the
  current UI (it has no "how many" display). Carry them in an optional `levels` field if a future
  UI wants them; do not drop the capability — just don't render it yet.

### 3.2 Detection (the `detected[]` array)

The UI uses `detected[]` two ways: lamps/headline in `detection.js`, and `scene.pulse()` ring in
`scene.js`. The sim computes it as `probs[c] >= 0.5`. Two options for the real backend:

- **Minimum:** replicate the sim — `detected = [c for c in (human,car,animal) if probs[c] >= 0.5]`,
  sorted desc. Zero behavioural surprise for the UI.
- **Recommended (more honest, and what the model design intends):** apply **per-class hysteresis +
  N-of-M persistence** server-side (entry/exit thresholds + "3-of-5 windows" as used in the training
  evaluation, `geophone_snn_v2_train.py`). This dramatically cuts single-window false alarms — which
  matters because this is being prepared for publication and event-FAR is a reviewer target. The UI
  still just reads `detected[]`; the smarter logic is invisible to it. **Expose the thresholds in
  `/config`** so they're documented, not hidden magic numbers.

Decision flagged as an **open question** (§6) — depends on whether the article reports
single-window or persisted detection.

### 3.3 Terrain mapping (simulated mode only)

simgeo terrain families are richer than the UI's 3. Collapse to the UI vocab for display:
- `{asphalt, concrete, paving}` → **`asphalt`**
- `{gravel, dirt_road, kurkar, rock, ...}` → **`gravel`**
- `{soft_soil, clay, loess, wet_soil, sand, ...}` → **`soil`**

The UI only needs the 3-way label for `TERRAIN_NAME` + the ground texture (`scene.js` loads
`assets/textures/{soil,gravel,asphalt}.jpg`). Geophone mode has unknown terrain → `null`.

---

## 4. How `webui/src/sim.js` is replaced / augmented

### Recommendation: **Hybrid (option a) — keep `sim.js`, add a thin WS client. Backend owns
geophone; backend MAY also own simulated, but the JS sim stays as the offline fallback.**

Rationale comes straight from the feasibility numbers gathered from the code:

- **simgeo on-demand generation costs ~50–100 ms per 3 s window** (from the generator audit:
  GF-bank convolution + sensor render dominate; banks are LRU-cached so terrain reuse is cheap).
  Against a **1.5 s hop budget** that is a comfortable **~15–30×** headroom — so streaming
  *backend-generated* simulated windows **is feasible**, contradicting any "offline only" worry as
  long as we generate window-by-window (not whole 171k-scene corpora) and keep the Bank cache warm.
- BUT the existing `sim.js` is already a *perfectly good, zero-latency, dependency-free* demo. It
  also produces the smooth per-frame geometry the renderers animate between windows (the 60 fps
  `frame()` loop in `main.js` calls `sim.getState()` every frame, not every window). Replacing that
  with network round-trips would make the field-view animation stutter.

So the clean split:

| Mode | Geometry / per-frame animation | Per-window probabilities |
|---|---|---|
| **Simulated** (default) | `sim.js` in-browser (unchanged) — drives the 60 fps scene | **EITHER** keep `sim.sampleWindow()` (pure demo) **OR**, if connected, let the backend run the *real model* over *backend-simgeo* audio and stream true model probs while JS still animates geometry |
| **Geophone** | no geometry — UI blind view (already exists) | backend: live serial → features → model → stream |

**Concretely, three tiers, ship in this order:**

1. **Tier 1 (geophone only).** `sim.js` untouched for simulated. Add `src/stream.js`: a WS client
   that, when `mode==='geo'` *and* a socket is open, feeds incoming `window` messages into
   `histo.onWindow()`, `detect.onWindow()`, the signal meter, and the status bar — i.e. it calls the
   exact same UI methods `main.js` already calls. The 60 fps loop keeps running the geo blind-view
   render. This gets **geophone → model → UI end-to-end** with the least code.

2. **Tier 2 (real model on simulated).** Optionally let the backend stream true model probabilities
   for simulated mode too, generating audio via simgeo and running the real SNN — while the browser
   `sim.js` continues to own the *visual* scene geometry. Probabilities then reflect the actual
   trained model on synthetic signals (much more honest for a demo/paper than the fake sigmoids in
   `sim.js`). The browser sim's `probs` become a fallback used only when no socket is connected.

3. **Tier 3 (consolidate, optional).** If we want a single source of truth, move scene geometry
   generation into the backend too and stream it in the `scene` block (schema already supports it).
   Only worth it if we need server-recorded, reproducible scenes for the paper. Higher risk
   (network jitter vs. animation), so deferred.

**Wiring touch-points in the existing UI (no behaviour removed):**
- `index.html` `sourceToggle` already exists — its handler (`main.js` L48–53) calls
  `sim.setMode()`. Augment to also send `{type:"set_mode"}` over the socket and choose source
  (local sim vs. stream) per tier.
- `playBtn` / `newBtn` / `speed` handlers → also emit `pause`/`resume`/`new_scene`/`set_speed`.
- `main.js` window block (L91–99): when a socket is the active source, **skip** the local
  `winAccum >= WINDOW_MS` call to `sim.sampleWindow()` and instead let `stream.js` invoke
  `histo/detect/scene.pulse/updateReadouts` on message arrival. The `acqFill` progress tick should
  count to the *server's* next-window time (sent in `hello.hopSec`), keeping the "next 1.5 s" readout
  honest. **Nothing in `sim.js` is deleted** — it remains the simulated path and offline fallback,
  satisfying the "never remove functionality" rule.

---

## 5. CORS, run/deploy, dependencies, milestones

### 5.1 CORS

The webui is static files served by Vite/whatever dev server (e.g. `http://localhost:5173`) or
opened from `file://`; the API is `http://localhost:8000`. Cross-origin → enable CORS.

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
)
```

- WebSocket handshakes are **not** subject to CORS the same way, but browsers send an `Origin`
  header on WS upgrade — validate it server-side against the same allow-list for safety.
- If the UI is ever served by the FastAPI process itself (mount `StaticFiles`), CORS becomes moot
  (same origin). That's a nice option for the final demo: one `uvicorn` command serves both.

### 5.2 Run / deploy

```
# from project root, with .venv active (the repo already has a .venv/)
pip install -r webui/wireup_plan/backend-requirements.txt   # (to be authored)
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

# UI (separate terminal), from webui/
npm run dev        # or: python -m http.server, or mount via FastAPI StaticFiles
```

Suggested layout (new files only — nothing existing is touched):
```
backend/
  app.py            # FastAPI app, routes, WS /stream, CORS, startup model load
  inference.py      # load model_ema.pt + scaler.json; window → probs (reuses simgeo.features)
  sources/
    sim_source.py   # wraps simgeo: generate one 3 s window on demand (+ geometry)
    geo_source.py   # NEW live serial ring-buffer reader (from record15.py serial code)
  mapping.py        # head→UI class map, terrain collapse, detected[] hysteresis
  schema.py         # Pydantic models for the §2 messages
```

- The backend imports `simgeo.features` and the simgeo generator **in place** (add project root to
  `sys.path`); it does not copy or fork them. Keeps one source of truth for the pipeline.
- Device: CPU is fine (T=4, 104→512→256→128 MLP-SNN; one window is sub-ms on CPU). GPU optional.
  Note: training used a `cupy` backend for PLIF; for **inference** force the torch backend / CPU to
  avoid a CUDA/cupy dependency on the demo machine (flag in §6).

### 5.3 Dependencies (`backend-requirements.txt`, to be authored)

```
fastapi
uvicorn[standard]        # includes websockets + httptools
pydantic>=2
torch                    # match the version that produced model_ema.pt
spikingjelly             # SNN runtime (PLIF, functional.reset_net, set_step_mode)
numpy
scipy                    # features.py uses scipy.signal (sosfilt, find_peaks)
pywavelets               # WPE16 wavelet-packet features (db4) in features.py
pyserial                 # live geophone (CH340 @ 460800)
```
> Pin versions against whatever is already in the repo `.venv/` so the loaded `state_dict` and the
> SpikingJelly API match. **Mismatched torch/spikingjelly is the #1 likely breakage.** Verify by
> loading `model_ema.pt` once during Tier-0 below.

### 5.4 Milestones (build order — geophone end-to-end first)

- **M0 — Inference smoke test (no web).** Script: load `model_ema.pt` + `scaler.json`, feed one
  real `geophone_*.csv` through `scene_precompute`/`window_features` → 104 feats → z-score+clip →
  `reset_net` → model → sigmoid. Print per-head probs. **Proves the model loads and the pipeline is
  reproducible outside training.** Highest-risk item; do it before any web code. (Largely covered by
  sibling plans `10`/`20`; this milestone just confirms it runs standalone.)

- **M1 — FastAPI skeleton + WS echo.** `/health`, `/config`, `/stream` that pushes a hardcoded
  `window` message every 1.5 s. **Add `stream.js` to the UI; confirm a fake stream lights up the
  histograms/lamps/meter in geophone mode.** Validates the §2 contract against the real UI before any
  hardware/model is involved.

- **M2 — Geophone live source.** Build `geo_source.py`: pyserial reader (port auto-detect via CH340
  VID/PID, `dtr=rts=False`, 460800), 30 s ring buffer, emit a 3 s window every 1.5 s. Wire into the
  inference core. **Geophone → model → UI end-to-end. This is the primary goal.**

- **M3 — Simulated source via real model (Tier 2).** `sim_source.py` generates a 3 s simgeo window
  on demand (warm Bank cache), run through the *same* inference core, stream true model probs +
  geometry. Keep `sim.js` as fallback.

- **M4 — Robustness & polish.** Hysteresis/N-of-M for `detected[]`, `NO_CH340` error surfacing,
  reconnect logic in `stream.js`, pause/resume/new-scene control wiring, optional `StaticFiles`
  single-origin serve, `/config`-driven thresholds.

---

## 6. Open questions & risks

**Open questions (need a decision before/while building):**

1. **Detection logic for `detected[]`:** naive `p≥0.5` (matches sim) vs. server-side hysteresis +
   N-of-M (matches training eval, lower FAR). What does the paper report? (§3.2) — *publication
   relevance: event-FAR is a stated reviewer target, lean toward persisted.*
2. **Real-data amplitude calibration.** The scaler was fit on **synthetic** data; the feature audit
   found real RMS is class-dependently ~1–4% of synthetic, and `rescale_test.py` probes scale factors
   `{1.0, 25.4, 90.0}`. **Which scale (if any) does the live path apply before featurizing?** If none,
   live `log_rms`/energy features sit far outside the z-score range and the model may degrade. This is
   a real accuracy risk for the geophone demo, and a known domain-gap confound — must be settled with
   the model/feature plans, not hand-waved.
3. **SNR & terrain in geophone mode.** Model gives neither. Send `snr:null`, `terrain:null`? The UI
   already guards `#stSnr` on `snr>-15`; confirm it tolerates `null`/`—`.
4. **Ordinal `[:,1]` head outputs (human/animal count/level).** Surface as optional `levels`, or
   ignore for now? (Capability preserved either way — §3.1.)
5. **Window cadence reconciliation.** Confirm pushing every **1.5 s** (the hop) — not every 3 s (the
   window) — is what we want; overlapping windows mean each push shares 1.5 s with the previous.
6. **CUDA/cupy at inference.** Training used a cupy PLIF backend. Confirm inference runs on the torch
   CPU backend so the demo box needs no GPU/cupy.

**Risks:**

- **R1 — torch/spikingjelly version drift** between training env and a fresh backend env → silent
  `state_dict` load failure or wrong neuron dynamics. *Mitigation:* pin to repo `.venv`, M0 smoke
  test gates everything.
- **R2 — Domain gap (synthetic-trained, real-tested).** Live geophone detections may be poor/biased
  regardless of plumbing correctness. *Mitigation:* M0/M2 validate on recorded real CSVs first; flag
  the calibration question (#2) loudly. **Do not present uncalibrated live numbers as model accuracy.**
- **R3 — Serial exclusivity / hot-unplug.** One process owns the port; unplug mid-stream throws.
  *Mitigation:* singleton `GeoSource`, `NO_CH340` error message, `stream.js` reconnect.
- **R4 — Per-frame animation vs. network cadence.** If geometry ever moves server-side (Tier 3),
  network jitter would stutter the 60 fps field view. *Mitigation:* keep geometry client-side
  (`sim.js`) for simulated mode; only stream probabilities.
- **R5 — simgeo realtime budget.** ~50–100 ms/window is fine at a 1.5 s hop, but a *cold* Bank load
  (~10 MB .npz) can spike. *Mitigation:* warm the cache at startup; reuse one terrain per scene.
- **R6 — Blocking the event loop.** Feature extraction + simgeo are synchronous CPU work.
  *Mitigation:* `run_in_executor` / thread pool; never `await`-around them on the main loop.
- **R7 — Contract drift.** If `probs` keys or `ORDER` change, histograms break silently.
  *Mitigation:* Pydantic schema in `schema.py` is the single contract; `/config` advertises
  `classes` so the UI can assert.
```
