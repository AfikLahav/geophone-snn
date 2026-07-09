// Small shared helpers.

export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const lerp  = (a, b, t) => a + (b - a) * t;
export const rand  = (a = 0, b = 1) => a + Math.random() * (b - a);
export const choice = (arr) => arr[(Math.random() * arr.length) | 0];
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Gaussian noise (Box–Muller).
export function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Weighted pick: items = [[value, weight], ...]
export function weighted(items) {
  const total = items.reduce((s, [, w]) => s + w, 0);
  let r = Math.random() * total;
  for (const [val, w] of items) { if ((r -= w) <= 0) return val; }
  return items[items.length - 1][0];
}

// Resize a canvas to its CSS box at devicePixelRatio; returns {w,h,ctx,dpr}.
export function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, ctx, dpr };
}
