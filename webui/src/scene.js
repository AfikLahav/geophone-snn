// 2.5D isometric field view — austere. Sensor at the origin; the source
// crosses the ground plane. Flat fills, hairline grid, no glow/pulse.
// A single thin ring marks a detection, then fades.

import { fitCanvas, clamp } from './util.js';
import { pal } from './palette.js';
import { drawSource } from './icons.js';

// real CC0 terrain photos (Poly Haven), loaded once and tiled onto the ground
const TEX = {};
for (const name of ['soil', 'gravel', 'asphalt']) {
  const img = new Image();
  img.src = `./assets/textures/${name}.jpg`;
  TEX[name] = { img, pat: null };
}

function hexA(hex, a) {
  const h = hex.replace('#', '');
  const n = h.length === 3 ? h.split('').map((x) => x + x).join('') : h.slice(0, 6);
  const r = parseInt(n.slice(0, 2), 16), g = parseInt(n.slice(2, 4), 16), b = parseInt(n.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export function createScene(canvas) {
  let W = 0, H = 0, ctx = null;
  let trail = [];
  let ripples = [];          // {age, life} — emitted from the source on a hit
  let sweep = 0;

  function resize() { ({ w: W, h: H, ctx } = fitCanvas(canvas)); }
  resize();

  const project = (wx, wy, s, cx, cy) => [cx + (wx - wy) * s * 0.86, cy + (wx + wy) * s * 0.5];

  function ring(R, s, cx, cy) {
    ctx.beginPath();
    for (let i = 0; i <= 48; i++) {
      const a = (i / 48) * Math.PI * 2;
      const [px, py] = project(Math.cos(a) * R, Math.sin(a) * R, s, cx, cy);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
  }

  function pulse(detected, state) {
    if (!detected || detected.length === 0) return;
    ripples.push({ age: 0, life: 1.4, atSource: state.hasSource });
  }

  // tile the terrain photo onto the iso ground plane, fade it into the UI
  function drawGround(c, terrain, cx, cy, ext, s) {
    const t = TEX[terrain] || TEX.soil;
    if (!t.img.complete || !t.img.naturalWidth) return;
    if (!t.pat) t.pat = ctx.createPattern(t.img, 'repeat');
    const SPAN = 280;                                  // px the FULL texture spans on screen (iso)
    const k = SPAN / (t.img.naturalWidth || 1024);     // screen px per texture pixel
    t.pat.setTransform(new DOMMatrix([0.86 * k, 0.5 * k, -0.86 * k, 0.5 * k, cx, cy]));
    ctx.save();
    ring(ext * 1.05, s, cx, cy); ctx.clip();
    ctx.globalAlpha = 0.9; ctx.fillStyle = t.pat; ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1;
    const R = Math.min(W, H);
    const vg = ctx.createRadialGradient(cx, cy, R * 0.10, cx, cy, R * 0.52);
    vg.addColorStop(0, 'rgba(0,0,0,0)');
    vg.addColorStop(0.7, hexA(c.sceneBg, 0.4));
    vg.addColorStop(1, c.sceneBg);
    ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }

  function render(state, dt) {
    const c = pal();
    const geo = state.mode === 'geo';          // blind sensor: no terrain, no source, generic range
    const cx = W / 2, cy = H * 0.52;
    const s = (Math.min(W, H) * 0.40) / Math.max(state.viewExtent, 20);

    ctx.fillStyle = c.sceneBg;
    ctx.fillRect(0, 0, W, H);

    const ext = state.viewExtent;

    // textured terrain ground (real photo, iso-projected) — hidden in geo: the live
    // sensor cannot know the ground material, so a blind view shows neutral ground only.
    if (!geo) drawGround(c, state.terrain, cx, cy, ext, s);

    // faint iso grid over the texture
    const step = niceStep(ext / 5);
    ctx.save();
    ctx.globalAlpha = 0.16; ctx.lineWidth = 1; ctx.strokeStyle = c.grid;
    for (let v = -ext; v <= ext + 1; v += step) {
      let [ax, ay] = project(v, -ext, s, cx, cy);
      let [bx, by] = project(v, ext, s, cx, cy);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      [ax, ay] = project(-ext, v, s, cx, cy);
      [bx, by] = project(ext, v, s, cx, cy);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    ctx.restore();

    // range rings (kept readable over the texture). In geo they're a GENERIC display
    // grid off the view extent, not a class-specific detection radius the sensor can't know.
    const Rdet = geo ? ext : (state.scene?.Rdet || 60);
    ctx.font = '10px var(--font-mono, monospace)';
    for (const frac of [0.25, 0.5, 0.75, 1.0]) {
      ring(Rdet * frac, s, cx, cy);
      ctx.strokeStyle = c.textDim;
      ctx.globalAlpha = frac === 1 ? 0.75 : 0.4;
      ctx.lineWidth = 1;
      ctx.setLineDash(frac === 1 ? [] : [2, 4]);
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      const [lx, ly] = project(0, Rdet * frac, s, cx, cy);
      ctx.fillStyle = c.text;
      ctx.fillText(`${Math.round(Rdet * frac)} m`, lx + 4, ly - 3);
    }

    // detection rings (thin, single, fading)
    ripples = ripples.filter((rp) => rp.age < rp.life);
    for (const rp of ripples) {
      rp.age += dt;
      const t = rp.age / rp.life;
      ring(Math.max((rp.atSource ? Rdet * 0.8 : ext * 0.85) * t, 0.1), s, cx, cy);
      ctx.strokeStyle = c.accent;
      ctx.globalAlpha = clamp(1 - t, 0, 1) * 0.55;
      ctx.lineWidth = 1.4; ctx.stroke(); ctx.globalAlpha = 1;
    }

    if (state.mode === 'geo') {
      drawSensor(cx, cy, c, true);
      sweep = (sweep + dt * 0.5) % (Math.PI * 2);   // slow bearing sweep
      const [ex, ey] = project(Math.cos(sweep) * ext * 0.85, Math.sin(sweep) * ext * 0.85, s, cx, cy);
      ctx.strokeStyle = c.grid; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke(); ctx.globalAlpha = 1;
      return;
    }

    // trail
    if (state.hasSource) { trail.push({ x: state.x, y: state.y }); if (trail.length > 80) trail.shift(); }
    else if (trail.length) trail = [];
    if (trail.length > 1) {
      ctx.lineWidth = 1.4; ctx.lineCap = 'round'; ctx.strokeStyle = c.trail;
      for (let i = 1; i < trail.length; i++) {
        const [ax, ay] = project(trail[i - 1].x, trail[i - 1].y, s, cx, cy);
        const [bx, by] = project(trail[i].x, trail[i].y, s, cx, cy);
        ctx.globalAlpha = (i / trail.length) * 0.5;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    drawSensor(cx, cy, c, false);

    if (state.hasSource) {
      const [px, py] = project(state.x, state.y, s, cx, cy);
      const size = clamp(Math.min(W, H) * 0.032, 9, 24);
      // flat ground shadow
      ctx.fillStyle = 'rgba(0,0,0,0.22)';
      ctx.beginPath(); ctx.ellipse(px, py, size * 0.65, size * 0.3, 0, 0, Math.PI * 2); ctx.fill();
      // source silhouette in the foreground ink; outline accent when over threshold
      const detected = state.intensity >= 0.5;
      drawSource(ctx, state.kind, px, py, size, c.text, detected ? c.accent : c.sceneBg);
    }
  }

  function drawSensor(cx, cy, c, geo) {
    // static crosshair + small square — an instrument marker, no animation
    ctx.strokeStyle = c.grid; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 9, cy); ctx.lineTo(cx + 9, cy);
    ctx.moveTo(cx, cy - 9); ctx.lineTo(cx, cy + 9);
    ctx.stroke();
    ctx.fillStyle = c.sensor;
    ctx.fillRect(cx - 3, cy - 3, 6, 6);
    ctx.fillStyle = c.textDim;
    ctx.font = '600 10px var(--font-mono, monospace)';
    ctx.fillText(geo ? 'GEOPHONE' : 'SENSOR', cx + 9, cy + 13);
  }

  return { resize, render, pulse };
}

function niceStep(x) {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  const n = x / p;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * p;
}
