// Simulated backend — stands in for the Python simgeo generator + SNN.
// It plays out a "scene" (a source crossing the ground near a sensor) and,
// every window, emits per-class probabilities that rise as the source
// approaches and fall as it leaves. The numbers are FAKE but the structure
// mirrors the real pipeline: detection-range, geometry, SNR→probability,
// and class confusion (human↔animal footstep overlap, etc.).

import { clamp, rand, randn, sigmoid, weighted } from './util.js';

// detection range (m), nominal speed (m/s) — same spirit as the real R_det.
const R_DET   = { human: 60,  car: 220, animal: 50 };
const SPEED   = { human: 1.4, car: 12,  animal: 2.6 };
const CONFUSE = { human: 'animal', animal: 'human', car: 'human' };
const SOURCES = ['human', 'car', 'animal'];

// 3 simplified terrains. Stiffer ground carries surface waves further (asphalt),
// soft soil attenuates more — modelled here as a dB shift on the carried signal.
const TERRAINS  = ['soil', 'gravel', 'asphalt'];
const TERRAIN_DB = { soil: -2.5, gravel: 0, asphalt: 2.5 };
export const TERRAIN_NAME = { soil: 'soft soil', gravel: 'gravel', asphalt: 'asphalt' };

export class Sim {
  constructor() {
    this.mode = 'sim';        // 'sim' | 'geo'
    this.win = 0;
    this.smoothed = { human: 0, car: 0, animal: 0, nothing: 1 };
    this.newScene();
  }

  setMode(m) { this.mode = m; this.newScene(); }

  newScene() {
    this.t = 0;
    this.sceneId = (this.sceneId || 0) + 1;
    this.terrain = weighted([['soil', 3], ['gravel', 2], ['asphalt', 2]]);
    this.bearing = rand(0, Math.PI * 2);           // for the "blind sensor" geo view
    const empty = Math.random() < 0.15;
    if (empty) {
      this.kind = null;
      this.duration = rand(7, 13);
      this.viewExtent = 60;
      this.scene = { style: 'quiet', d: 0, speed: 0 };
      return;
    }
    const kind = weighted([['human', 3], ['car', 2], ['animal', 1.6]]);
    const Rdet = R_DET[kind];
    const speed = SPEED[kind] * rand(0.8, 1.3);
    const d = rand(3, 0.5 * Rdet);                  // closest approach
    const half = Math.sqrt(Math.max((1.15 * Rdet) ** 2 - d * d, 4));
    const style = weighted([['pass-by', 5], ['oblique', 3]]);
    const angle = style === 'oblique' ? rand(0.2, 0.9) * (Math.random() < 0.5 ? 1 : -1) : 0;

    this.kind = kind;
    this.scene = { style, d, speed, half, angle, Rdet };
    this.duration = (2 * half) / speed;
    this.viewExtent = Rdet * 1.12;
  }

  // continuous state for the per-frame renderers
  update(dt) {
    this.t += dt;
    if (this.t >= this.duration) this.newScene();
  }

  position() {
    const s = this.scene;
    if (!this.kind) return { x: 0, y: 0, r: Infinity };
    const x = -s.half + s.speed * this.t;
    const y = s.d + (s.angle ? x * Math.tan(s.angle) * 0.12 : 0);
    return { x, y, r: Math.hypot(x, y) };
  }

  // detection "envelope" 0..1 from range, plus the dB SNR for display
  detect(r) {
    if (!this.kind || !isFinite(r)) return { env: 0, snr: -20 };
    const frac = clamp(1 - r / this.scene.Rdet, 0, 1);
    const snr = -8 + 42 * frac + TERRAIN_DB[this.terrain];   // terrain shifts carry
    return { env: sigmoid((snr - 6) / 4.5), snr };
  }

  getState() {
    const p = this.position();
    const { env, snr } = this.detect(p.r);
    return {
      mode: this.mode, kind: this.kind, hasSource: !!this.kind,
      x: p.x, y: p.y, r: p.r, snr, intensity: env,
      t: this.t, duration: this.duration, viewExtent: this.viewExtent,
      bearing: this.bearing, scene: this.scene,
      terrain: this.terrain, sceneId: this.sceneId,
    };
  }

  // one window → per-class probabilities (what the model "thinks")
  sampleWindow() {
    const st = this.getState();
    const target = { human: 0, car: 0, animal: 0, nothing: 0 };

    if (this.kind) {
      const env = st.intensity;
      target[this.kind] = clamp(env * rand(0.86, 0.99) + randn() * 0.03, 0, 1);
      const c = CONFUSE[this.kind];
      target[c] = clamp(env * rand(0.05, 0.22) * Math.random(), 0, 1);
      for (const cls of SOURCES) if (target[cls] === 0) target[cls] = clamp(rand(0, 0.06), 0, 1);
      target.nothing = clamp((1 - env) * rand(0.7, 1.0) + randn() * 0.03, 0, 1);
    } else {
      target.nothing = clamp(rand(0.85, 0.99) + randn() * 0.02, 0, 1);
      for (const cls of SOURCES) target[cls] = clamp(rand(0, 0.06), 0, 1);
    }

    // light smoothing only — each ~2 s read should look like a fresh measurement
    const a = 0.3;
    const probs = {};
    for (const k of ['human', 'car', 'animal', 'nothing']) {
      probs[k] = clamp(this.smoothed[k] * a + target[k] * (1 - a), 0, 1);
      this.smoothed[k] = probs[k];
    }

    const detected = SOURCES.filter((c) => probs[c] >= 0.5)
      .sort((p, q) => probs[q] - probs[p]);

    // ground truth — the mock KNOWS what it generated (one source per scene, or quiet).
    // It only ever emits a single source, so every present class is 'single'.
    const truth = this.kind
      ? { coarse: this.kind === 'car' ? 'vehicle' : this.kind, active: [this.kind],
          human_level: this.kind === 'human' ? 'single' : 'none',
          car_level: this.kind === 'car' ? 'single' : 'none',
          animal_level: this.kind === 'animal' ? 'single' : 'none',
          common_snr: st.snr }
      : { coarse: 'nothing', active: [],
          human_level: 'none', car_level: 'none', animal_level: 'none', common_snr: -99 };
    const levels = { human: 'single', car: 'single', animal: 'single' };

    this.win += 1;
    return { win: this.win, probs, detected, snr: st.snr, kind: this.kind, r: st.r, truth, levels };
  }
}
