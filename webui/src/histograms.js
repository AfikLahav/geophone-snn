// Probability section. Four class rows (Human / Car / Animal / All-clear):
// a dim bar that turns to the accent only when it crosses threshold, plus a
// single-hue, discrete-cell ribbon (one cell per ~2 s window) — a measurement
// log, not a smooth heatmap.

import { fitCanvas, clamp } from './util.js';
import { pal } from './palette.js';
import { SVG, NAME } from './icons.js';

const ORDER = ['human', 'car', 'animal', 'nothing'];
const MAXH = 60;              // history length (windows) shown on the chart
// one line per class. B/W-safe: foreground colour at 4 opacities + 4 line styles.
// [class, opacity, canvas-dash]
const LINES = [
  ['human', 1.0, []],
  ['car', 0.82, [5, 3]],
  ['animal', 0.6, [1.5, 2.5]],
  ['nothing', 0.42, []],
];
const LEGNAME = { human: 'Human', car: 'Car', animal: 'Animal', nothing: 'Clear' };
const LEGSTYLE = { human: 'solid', car: 'dashed', animal: 'dotted', nothing: 'solid' };

export function createHistograms(barsEl, ribbonCanvas) {
  const rows = {};
  let W = 0, H = 0, ctx = null;
  const history = [];

  function build() {
    barsEl.innerHTML = '';
    for (const cls of ORDER) {
      const row = document.createElement('div');
      row.className = 'hbar';
      row.innerHTML = `
        <span class="hbar-ico">${SVG[cls]}</span>
        <span class="hbar-name">${NAME[cls]}</span>
        <span class="hbar-track"><span class="hbar-fill"></span></span>
        <span class="hbar-pct">0%</span>`;
      barsEl.appendChild(row);
      rows[cls] = { row, fill: row.querySelector('.hbar-fill'), pct: row.querySelector('.hbar-pct') };
    }
    const legEl = document.getElementById('ribbonLegend');
    if (legEl) legEl.innerHTML = LINES.map(([cls, a]) =>
      `<span class="leg2"><i style="opacity:${a};border-top-style:${LEGSTYLE[cls]}"></i>${LEGNAME[cls]}</span>`).join('');
  }

  function onWindow(w) {
    for (const cls of ORDER) {
      const p = w.probs[cls];
      const r = rows[cls];
      r.fill.style.width = `${(p * 100).toFixed(0)}%`;
      r.pct.textContent = `${Math.round(p * 100)}%`;
      r.row.classList.toggle('is-hit', p >= 0.5);
    }
    history.push({ ...w.probs });
    if (history.length > MAXH) history.shift();
    renderRibbon();
  }

  // clear bars + ribbon history (called on a source switch so modes don't bleed)
  function reset() {
    history.length = 0;
    for (const cls of ORDER) {
      const r = rows[cls];
      if (!r) continue;
      r.fill.style.width = '0%';
      r.pct.textContent = '0%';
      r.row.classList.remove('is-hit');
    }
    renderRibbon();
  }

  function resize() { ({ w: W, h: H, ctx } = fitCanvas(ribbonCanvas)); }
  resize();

  function renderRibbon() {
    if (!ctx) return;
    const c = pal();
    ctx.clearRect(0, 0, W, H);
    const padR = 22, padT = 4, padB = 12;
    const x0 = 2, x1 = W - padR, y0 = padT, y1 = H - padB;
    const pw = x1 - x0, ph = y1 - y0;
    const yOf = (p) => y1 - clamp(p, 0, 1) * ph;

    // 0% / 100% baselines + dashed 50% threshold guide
    ctx.strokeStyle = c.baseline; ctx.lineWidth = 1; ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
    ctx.globalAlpha = 0.45; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(x0, yOf(0.5)); ctx.lineTo(x1, yOf(0.5)); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    // y labels
    ctx.fillStyle = c.textDim; ctx.font = '9px var(--font-mono, monospace)';
    ctx.fillText('100', x1 + 3, y0 + 7);
    ctx.fillText('50', x1 + 3, yOf(0.5) + 3);
    ctx.fillText('0', x1 + 3, y1 + 1);

    // one line per class, newest pinned to the right edge
    const n = history.length;
    if (n >= 2) {
      const step = pw / (MAXH - 1);
      for (const [cls, a, dash] of LINES) {
        ctx.strokeStyle = c.text; ctx.globalAlpha = a;
        ctx.lineWidth = cls === 'human' ? 1.8 : 1.4;
        ctx.lineJoin = 'round'; ctx.setLineDash(dash);
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = x1 - (n - 1 - i) * step;
          const y = yOf(history[i][cls]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
  }

  return { build, onWindow, resize, renderRibbon, reset };
}
