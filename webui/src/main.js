// Orchestrator: builds the UI, wires controls, runs the animation + window loop.
// Three sources:
//   Dataset   — live-generated, never-trained v3 scenes (backend) + ground truth
//   Geophone  — the live sensor (backend)
//   Simulated — in-browser mock (offline fallback)

import { Sim, TERRAIN_NAME } from './sim.js';
import { refreshPalette } from './palette.js';
import { createScene } from './scene.js';
import { createSignal } from './signal.js';
import { createHistograms } from './histograms.js';
import { createDetection } from './detection.js';
import { createStream } from './stream.js';
import { clamp } from './util.js';

const $ = (id) => document.getElementById(id);
const WINDOW_MS = 2000;

// v3 terrain family -> one of the 3 ground textures
const FAMILY_TEX = {
  asphalt: 'asphalt', concrete: 'asphalt', paving: 'asphalt',
  gravel: 'gravel', rock: 'gravel', terra_rossa: 'gravel', cover_basalt: 'gravel',
  chalk_marl: 'gravel', frozen: 'gravel', kurkar: 'gravel',
  sand: 'soil', sabkha: 'soil', loess: 'soil', soft_soil: 'soil',
  dirt_road: 'soil', wet_soil: 'soil', clay: 'soil', snow: 'soil',
};
const R_DET = { human: 60, car: 220, animal: 50 };

refreshPalette();

const sim = new Sim();
const scene = createScene($('scene'));
const signal = createSignal($('signal'), sim);
const histo = createHistograms($('histoBars'), $('ribbon'));
const detect = createDetection({ headline: $('detHeadline'), sub: $('detSub'), lamps: $('lamps'), meta: $('detMeta') });
histo.build();
detect.build();

$('sceneLegend').innerHTML =
  `<span class="leg-item"><span class="leg-swatch" style="background:var(--text)"></span>Source</span>` +
  `<span class="leg-item"><span class="leg-swatch" style="background:var(--accent)"></span>Sensor</span>` +
  `<span class="leg-item"><span class="leg-swatch" style="border:1px solid var(--accent);background:transparent"></span>Detection</span>`;

// ---- state ---------------------------------------------------------------
let running = true;
let speed = 1;
let winAccum = 0;
let last = performance.now();
let fpsEMA = 60;
let streamMode = null;            // null = JS sim, 'geo', 'dataset'
let datasetState = null;          // current live-gen scene (drives the canvas)
let lastWindowTime = performance.now();   // drives the acquisition tick in stream modes

const stream = createStream(onStreamMsg);

// ---- backend stream (geophone + dataset) ---------------------------------
function onStreamMsg(m) {
  if (m.type === 'ready') {
    $('stState').textContent = m.live ? 'LIVE' : (m.mode === 'dataset' ? 'LIVE-GEN' : 'REPLAY');
    return;
  }
  if (m.type === 'buffering') { $('detSub').textContent = 'buffering signal…'; return; }
  if (m.type === 'error') { $('stState').textContent = 'NO BACKEND'; return; }
  if (m.type === 'scene') { if (streamMode === 'dataset') onScene(m); return; }
  if (m.type !== 'window') return;
  if ((m.mode === 'dataset') !== (streamMode === 'dataset')) return;   // ignore stale frames
  if (!running) return;                                                // paused -> freeze stream

  lastWindowTime = performance.now();
  const w = { win: m.win, probs: m.probs, detected: m.detected, levels: m.levels };
  histo.onWindow(w);
  detect.onWindow(w);
  signal.pushSamples(m.trace);
  $('sigMeterFill').style.width = `${m.signal.pct}%`;
  $('sigStrength').textContent = `${m.signal.pct}%`;
  $('stWin').textContent = m.win;

  if (streamMode === 'dataset') {
    detect.setTruth(m.truth);
    $('stSnr').textContent = m.truth ? `${Math.round(m.truth.common_snr)} dB` : '—';
    $('histoMeta').textContent = 'live-gen · 1.5 s';
    if (datasetState) datasetState.winIdx += 1;
    scene.pulse(m.detected, datasetSceneState());
  } else {
    $('stSnr').textContent = '—';
    $('histoMeta').textContent = 'live · 1.5 s';
    scene.pulse(m.detected, geoState());
  }
}

function onScene(m) {
  const meta = m.meta;
  const fam = meta.family;
  const terrain = FAMILY_TEX[fam] || 'soil';
  const kind = meta.coarse === 'vehicle' ? 'car'
    : meta.coarse === 'nothing' ? null
    : meta.coarse === 'mixed' ? primaryKind(meta.counts)
    : meta.coarse;
  const Rdet = R_DET[kind] || 60;
  // honest geometry: source classes always carry a real closest-approach + speed
  // (from paths.sample); only 'nothing' is null. No fabricated distance/sweep.
  const hasGeom = meta.closest_approach_m != null && meta.speed_ms != null;
  const d = hasGeom ? meta.closest_approach_m : Rdet * 0.4;
  const speed = meta.speed_ms != null ? meta.speed_ms : 0;
  const dur = meta.dur_s || 0;
  // metres the source travels each side of its closest approach (real speed × duration)
  const halfSpan = hasGeom ? Math.max((speed * dur) / 2, 1) : 0;
  // size the view to the motion, but cap zoom-out so the detection rings stay readable;
  // the source rides the frame edge during any far-field portion beyond the cap.
  const viewExtent = Math.min(Math.max(Rdet * 1.12, halfSpan * 1.08, d * 1.3), Rdet * 3.5);
  datasetState = { terrain, kind, Rdet, viewExtent, d, speed, dur, halfSpan, hasGeom,
                   style: meta.path_style || 'pass-by', nWin: Math.max(1, m.n_windows), winIdx: 0 };
  const sub = meta.subkind ? `/${meta.subkind}` : '';
  const dist = meta.closest_approach_m != null ? ` · ${meta.closest_approach_m} m` : '';
  const spd = hasGeom && speed ? ` · ${speed.toFixed(1)} m/s` : '';
  $('sceneMeta').textContent = `${fam} · ${meta.coarse}${sub}${dist}${spd}`;
}

function primaryKind(counts) {
  let best = null, bn = 0;
  for (const c of ['human', 'vehicle', 'animal']) if ((counts[c] || 0) > bn) { bn = counts[c]; best = c; }
  return best === 'vehicle' ? 'car' : best;
}

function datasetSceneState() {
  if (!datasetState) return geoState();
  const ds = datasetState;
  // normalized scene time 0..1; closest approach sits at mid-scene (the path's true
  // along-track timing isn't in the corpus, so mid-scene is the honest representative).
  const frac = ds.nWin > 1 ? clamp(ds.winIdx / (ds.nWin - 1), 0, 1) : 0.5;
  // straight-line pass-by from the REAL speed × duration; no geometry -> static at origin.
  const xTrue = ds.hasGeom ? ds.halfSpan * (2 * frac - 1) : 0;
  const xMax = ds.viewExtent * 0.98;                 // keep the icon inside the frame
  const x = clamp(xTrue, -xMax, xMax);
  const y = ds.d, r = Math.hypot(xTrue, y);          // SNR from true range, not the clamped x
  const intensity = ds.kind ? clamp(1 - r / ds.Rdet, 0, 1) : 0;
  return { mode: 'sim', kind: ds.kind, hasSource: !!ds.kind, x, y, r, intensity,
           viewExtent: ds.viewExtent, terrain: ds.terrain,
           scene: { Rdet: ds.Rdet, style: ds.style, d: ds.d } };
}

// blind-sensor state for Geophone mode: no terrain, no source, fixed display range.
function geoState() {
  return { mode: 'geo', kind: null, hasSource: false, x: 0, y: 0, r: Infinity,
           intensity: 0, viewExtent: 120, terrain: null, scene: { Rdet: 120, style: 'blind', d: 0 } };
}

// ---- controls ------------------------------------------------------------
$('themeToggle').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-theme]'); if (!btn) return;
  document.documentElement.setAttribute('data-theme', btn.dataset.theme);
  setActive('themeToggle', btn);
  refreshPalette();
});

$('sourceToggle').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-source]'); if (!btn || btn.disabled) return;
  selectSource(btn.dataset.source);
});

function selectSource(src) {
  const btn = $('sourceToggle').querySelector(`[data-source="${src}"]`);
  if (btn) setActive('sourceToggle', btn);
  stream.stop();
  datasetState = null;
  detect.setTruth(null);
  resetReadouts();                       // clear the previous mode's stale values
  if (src === 'dataset') {
    streamMode = 'dataset'; sim.setMode('geo');
    $('stMode').textContent = 'DATASET';
    stream.start({ source: 'sim', speed });
  } else if (src === 'geo') {
    streamMode = 'geo'; sim.setMode('geo');
    $('stMode').textContent = 'GEOPHONE';
    $('sceneMeta').textContent = 'blind sensor · live';
    stream.start({ source: 'live', speed });
  } else {
    streamMode = null; sim.setMode('sim');
    $('stMode').textContent = 'SIMULATED';
    $('stState').textContent = running ? 'RUNNING' : 'PAUSED';
  }
}

// neutralize the readouts so a previous mode's numbers don't linger until the first
// new window (lamps -> All clear, meters -> 0, SNR/window cleared, histograms emptied).
function resetReadouts() {
  winAccum = 0;
  lastWindowTime = performance.now();
  $('stWin').textContent = '0';
  $('stSnr').textContent = '—';
  $('sigStrength').textContent = '—';
  $('sigMeterFill').style.width = '0%';
  $('acqFill').style.width = '0%';
  detect.onWindow({ detected: [], probs: { human: 0, car: 0, animal: 0, nothing: 1 }, levels: {} });
  histo.reset();
}

$('playBtn').addEventListener('click', () => {
  running = !running;
  if (running) { last = performance.now(); lastWindowTime = performance.now(); }   // avoid a dt/acq jump
  $('playBtn').dataset.state = running ? 'playing' : 'paused';
  $('playBtn').querySelector('.btn-label').textContent = running ? 'Pause' : 'Play';
  $('stState').textContent = running ? 'RUNNING' : 'PAUSED';
});

$('newBtn').addEventListener('click', () => {
  if (streamMode === 'dataset') stream.start({ source: 'sim', speed });
  else if (streamMode === 'geo') stream.start({ source: 'live', speed });
  else sim.newScene();
});

$('speed').addEventListener('input', (e) => {
  speed = parseFloat(e.target.value);
  $('speedVal').textContent = `${speed.toFixed(1)}×`;
});

function setActive(groupId, btn) {
  for (const b of $(groupId).querySelectorAll('.seg-btn')) b.classList.toggle('is-active', b === btn);
}

// ---- resize --------------------------------------------------------------
function resizeAll() { scene.resize(); signal.resize(); histo.resize(); histo.renderRibbon(); }
window.addEventListener('resize', resizeAll);
setTimeout(resizeAll, 250);

// ---- loop ----------------------------------------------------------------
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  fpsEMA = fpsEMA * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

  if (running && !streamMode) {              // JS-mock mode advances locally
    const sdt = dt * speed;
    sim.update(sdt);
    const st = sim.getState();
    signal.push(st, sdt);
    winAccum += sdt * 1000;
    if (winAccum >= WINDOW_MS) {
      winAccum = 0;
      const w = sim.sampleWindow();
      histo.onWindow(w); detect.onWindow(w); detect.setTruth(w.truth);
      scene.pulse(w.detected, st); updateReadouts(st, w);
    }
  }

  const st = streamMode === 'dataset' ? datasetSceneState()
    : streamMode === 'geo' ? geoState()
    : sim.getState();
  scene.render(st, running ? (streamMode ? dt : dt * speed) : 0);
  signal.render();

  if (!streamMode) {
    const str = signal.getStrength();
    $('sigMeterFill').style.width = `${str.pct}%`;
    $('sigStrength').textContent = `${str.pct}%`;
    const acq = Math.min(Math.max(winAccum / WINDOW_MS, 0), 1);
    $('acqFill').style.width = `${acq * 100}%`;
    $('histoMeta').textContent = `next ${Math.max(0, (WINDOW_MS - winAccum) / 1000).toFixed(1)} s`;
  } else if (running) {
    // acquisition tick toward the next backend window (~1.5 s cadence)
    const acq = clamp((now - lastWindowTime) / 1500, 0, 1);
    $('acqFill').style.width = `${acq * 100}%`;
  }
  $('stFps').textContent = Math.round(fpsEMA);
  requestAnimationFrame(frame);
}

function updateReadouts(st, w) {
  $('stWin').textContent = w.win;
  $('stSnr').textContent = st.hasSource && st.snr > -15 ? `${st.snr.toFixed(0)} dB` : '—';
  const terr = TERRAIN_NAME[st.terrain] || st.terrain;
  $('sceneMeta').textContent = st.hasSource ? `${terr} · ${st.scene.style}` : `${terr} · quiet`;
}

// Simulated is the default; Dataset + Geophone are complementary. The /sim probe only
// disables the Dataset button when the live generator isn't present on this host.
function disableDataset(reason) {
  const db = $('sourceToggle').querySelector('[data-source="dataset"]');
  if (db) { db.disabled = true; db.title = reason; }
}
fetch('/sim').then((r) => r.json()).then((d) => {
  if (!(d && d.available)) disableDataset('live generator not available on this host');
}).catch(() => disableDataset('live generator unavailable (no backend)'));
selectSource('sim');

requestAnimationFrame(frame);
