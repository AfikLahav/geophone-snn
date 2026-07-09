# GeoSense — UI prototype

First design prototype of the ground-vibration detection demo. It is a **running**
web app driven by a **simulated backend** — no Python, no SNN, no real geophone yet.
The point right now is to lock the look and the interaction.

## What's in it

- **Three themes, live toggle** (top-right): **Scope** (green CRT), **Console**
  (industrial grey, default), **Paper** (cream / ink). Pure CSS variables — no rebuild.
- **2.5D field view** — isometric ground, sensor at the origin, a source (human /
  car / animal) crossing the field with a trail, distance rings, and a detection
  ripple on each hit. The ground is one of **3 terrains** (soft soil / gravel /
  asphalt) chosen per scene, rendered with real CC0 photo textures and subtly
  affecting how far the signal carries. A **Signal strength** meter (RMS of the
  live trace — the `log_rms` feature) sits top-left. A **Geophone** source toggle
  shows the "blind sensor" view.
- **Detection readout** — plain language ("Human detected", "All clear") + status lamps.
- **Class probability** — four live bars (Human / Car / Animal / All-clear) that update
  every window, **plus a rolling probability-over-time ribbon** so you can watch a class
  light up as the source approaches and fade as it leaves.
- **Live signal** — scrolling scope of the synthesised waveform (footstep impulses for
  human/animal, low rumble for a car).
- Play/pause, **New scene**, and a speed slider.

The simulated backend (`src/sim.js`) mirrors the real pipeline's *structure* — detection
range, geometry, SNR→probability, class confusion — but the numbers are fake. Wiring it
to the real `simgeo` generator + SNN comes later.

## Run it

It uses ES modules, so it needs a static server (opening the file directly won't work).

```powershell
# from this folder:
python -m http.server 5173
# then open http://localhost:5173/
```

Any static server works (`npx serve`, VS Code Live Server, etc.).

## File map

```
index.html            layout + top bar + panels
styles/base.css       structure (theme-agnostic)
styles/themes.css     the 3 themes (--c-* canvas vars + DOM vars)
src/sim.js            simulated scene + window stream  ← swap for real backend later
src/scene.js          2.5D isometric renderer
src/signal.js         scope / waveform
src/histograms.js     probability bars + rolling ribbon
src/detection.js      readout + lamps
src/palette.js        reads theme colours into JS for canvas
src/icons.js          hand-drawn (non-emoji) icons
src/util.js           helpers (canvas fit, rng, math)
src/main.js           wiring + loop
```

## Assets

`assets/textures/{soil,gravel,asphalt}.jpg` are 1K diffuse maps from
[Poly Haven](https://polyhaven.com) (CC0). Re-pull with the same API if needed.

## Not wired yet (next, on your go)

- Real scene generation from `simgeo/` (source → GF banks → noise → sensor).
- Real 104-feature extraction + `model_ema.pt` inference per 3 s window.
- Geophone mode = replay a recorded CSV (live serial optional).
