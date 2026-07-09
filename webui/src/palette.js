// Bridges CSS theme variables (--c-*) into a JS object the canvas
// renderers can read. Call refreshPalette() whenever the theme changes.

const MAP = {
  accent: '--accent', text: '--text', textDim: '--c-text-dim', barDim: '--bar-dim',
  sceneBg: '--c-scene-bg', ground: '--c-ground', grid: '--c-grid', gridMinor: '--c-grid-minor',
  sensor: '--c-sensor', trail: '--c-trail',
  signalBg: '--c-signal-bg', signalGrid: '--c-signal-grid', trace: '--c-trace',
  baseline: '--c-baseline',
};

let cache = {};

export function refreshPalette() {
  const cs = getComputedStyle(document.documentElement);
  cache = {};
  for (const key in MAP) cache[key] = cs.getPropertyValue(MAP[key]).trim() || '#888';
  return cache;
}

export function pal() { return cache; }
