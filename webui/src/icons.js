// Hand-drawn (not emoji) icons. SVG strings for the DOM, and canvas
// drawing for the moving source in the 2.5D scene.

export const SVG = {
  human: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="5" r="2.4"/>
    <path d="M12 7.6V14M12 14l-3.4 6M12 14l3.4 6M7.6 10.6h8.8"/></svg>`,
  car: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4 15v-3l2-4.4h12L20 12v3"/><path d="M3 15h18"/><path d="M6.5 7.6h11"/>
    <circle cx="8" cy="16.6" r="1.7"/><circle cx="16" cy="16.6" r="1.7"/></svg>`,
  animal: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4 8.4l1.7-1.7v2.5"/><path d="M5.7 9.2c3-1.6 7-1.2 9.5.2 1.2.7 2.4.9 3.6.8"/>
    <path d="M6.6 10.6V15M9.6 11.1V15M14 11.1V15M17.6 11V15"/>
    <path d="M18.7 10.3c1.2-.2 2-.9 2.2-2.1"/></svg>`,
  nothing: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12h5l1.6-2.4L11.2 14 12.6 12H21"/></svg>`,
};

export const NAME = { human: 'Human', car: 'Car', animal: 'Animal', nothing: 'All clear' };

// ---- canvas source silhouettes ---------------------------------------
// Draw at ground point (x,y); the figure stands "up" the screen by `s`.
export function drawSource(ctx, kind, x, y, s, color, ink) {
  ctx.save();
  ctx.translate(x, y);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.fillStyle = color;
  ctx.strokeStyle = ink;
  ctx.lineWidth = Math.max(1, s * 0.06);

  if (kind === 'human') {
    const h = s * 1.7;
    // body capsule
    roundCap(ctx, 0, -h * 0.18, s * 0.30, h * 0.62);
    ctx.fill(); ctx.stroke();
    // head
    ctx.beginPath();
    ctx.arc(0, -h * 0.74, s * 0.26, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
  } else if (kind === 'car') {
    const w = s * 1.7, h = s * 0.95;
    // lower body
    rr(ctx, -w / 2, -h * 0.55, w, h * 0.5, s * 0.18); ctx.fill(); ctx.stroke();
    // cabin
    ctx.beginPath();
    ctx.moveTo(-w * 0.28, -h * 0.55);
    ctx.lineTo(-w * 0.16, -h * 1.0);
    ctx.lineTo(w * 0.18, -h * 1.0);
    ctx.lineTo(w * 0.30, -h * 0.55);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    // wheels
    ctx.fillStyle = ink;
    for (const wx of [-w * 0.28, w * 0.28]) {
      ctx.beginPath(); ctx.arc(wx, -h * 0.06, s * 0.18, 0, Math.PI * 2); ctx.fill();
    }
  } else { // animal
    const w = s * 1.6, h = s * 0.9;
    rr(ctx, -w / 2, -h * 0.7, w * 0.82, h * 0.42, s * 0.2); ctx.fill(); ctx.stroke();
    // head
    ctx.beginPath();
    ctx.arc(w * 0.34, -h * 0.78, s * 0.24, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // legs
    ctx.strokeStyle = ink; ctx.lineWidth = Math.max(1.2, s * 0.09);
    for (const lx of [-w * 0.34, -w * 0.08, w * 0.16]) {
      ctx.beginPath(); ctx.moveTo(lx, -h * 0.32); ctx.lineTo(lx, 0); ctx.stroke();
    }
  }
  ctx.restore();
}

function roundCap(ctx, cx, cy, rw, hh) {
  rr(ctx, cx - rw, cy - hh / 2, rw * 2, hh, rw);
}
function rr(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
