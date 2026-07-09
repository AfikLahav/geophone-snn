// Scrolling oscilloscope of the (synthetic) ground-vibration waveform.
// Shape is flavoured by the current source class so it reads believably:
// footstep impulses for human/animal, low rumble for a car.

import { fitCanvas, randn } from './util.js';
import { pal } from './palette.js';

export function createSignal(canvas, sim) {
  let W = 0, H = 0, ctx = null;
  let buf = new Float32Array(900);
  let head = 0;
  let tt = 0;          // internal signal time
  let footPhase = 0;

  function resize() {
    ({ w: W, h: H, ctx } = fitCanvas(canvas));
    const n = Math.max(300, Math.round(W));
    if (n !== buf.length) { buf = new Float32Array(n); head = 0; }
  }
  resize();

  function sample(st, dt) {
    const floorGain = st.mode === 'geo' ? 1.4 : 1;   // geophone floor a touch noisier
    const floor = randn() * 0.05 * floorGain;
    if (!st.hasSource) return floor;

    const amp = st.intensity;
    let v = floor;
    if (st.kind === 'car') {
      v += amp * (0.55 * Math.sin(tt * 2 * Math.PI * 9)
                + 0.30 * Math.sin(tt * 2 * Math.PI * 13.5)
                + 0.18 * Math.sin(tt * 2 * Math.PI * 30));
    } else {
      // footfall train (human / animal)
      const cadence = st.kind === 'animal' ? 2.6 : 1.8;
      footPhase += dt * cadence;
      const frac = footPhase - Math.floor(footPhase);
      const strike = Math.exp(-frac * 22) * Math.sin(frac * 2 * Math.PI * 28);
      v += amp * (0.85 * strike + 0.15 * Math.sin(tt * 2 * Math.PI * 2.2));
    }
    return v;
  }

  function push(st, dt) {
    tt += dt;
    // advance ~220 visual samples / sec, scaled by speed already in dt
    const n = Math.max(1, Math.round(220 * dt));
    for (let i = 0; i < n; i++) {
      buf[head] = sample(st, dt / n);
      head = (head + 1) % buf.length;
    }
  }

  function render() {
    const c = pal();
    ctx.fillStyle = c.signalBg;
    ctx.fillRect(0, 0, W, H);

    // graticule 10 x 8
    ctx.strokeStyle = c.signalGrid;
    ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      const x = (i / 10) * W;
      ctx.globalAlpha = i === 5 ? 0.9 : 0.4;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let j = 1; j < 8; j++) {
      const y = (j / 8) * H;
      ctx.globalAlpha = j === 4 ? 0.9 : 0.4;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const mid = H / 2, gain = H * 0.34;
    // trace
    ctx.beginPath();
    for (let i = 0; i < buf.length; i++) {
      const idx = (head + i) % buf.length;
      const x = (i / (buf.length - 1)) * W;
      const y = mid - Math.max(-1.6, Math.min(1.6, buf[idx])) * gain;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    if (c.traceGlow && c.traceGlow !== 'none') {
      ctx.shadowColor = c.traceGlow; ctx.shadowBlur = 8;
    }
    ctx.strokeStyle = c.trace;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.shadowBlur = 0;

    // leading edge marker
    ctx.fillStyle = c.trace;
    const lastIdx = (head - 1 + buf.length) % buf.length;
    const ly = mid - Math.max(-1.6, Math.min(1.6, buf[lastIdx])) * gain;
    ctx.beginPath(); ctx.arc(W - 2, ly, 2.6, 0, Math.PI * 2); ctx.fill();
  }

  // Feed real samples (Geophone mode) — normalised per push so the trace stays
  // visible regardless of absolute level.
  function pushSamples(arr) {
    if (!arr || !arr.length) return;
    let m = 1e-6;
    for (const v of arr) m = Math.max(m, Math.abs(v));
    const g = 1.2 / m;
    for (let i = 0; i < arr.length; i++) {
      buf[head] = arr[i] * g;
      head = (head + 1) % buf.length;
    }
  }

  // "Signal strength" = RMS energy of the current trace (the log_rms feature),
  // mapped to a 0..1 level for the meter.
  function getStrength() {
    let acc = 0;
    for (let i = 0; i < buf.length; i++) acc += buf[i] * buf[i];
    const rms = Math.sqrt(acc / buf.length);
    const db = 20 * Math.log10(Math.max(rms, 1e-4) / 0.045);
    const level = Math.min(Math.max((db + 3) / 25, 0), 1);
    return { rms, db, level, pct: Math.round(level * 100) };
  }

  return { resize, push, render, getStrength, pushSamples };
}
