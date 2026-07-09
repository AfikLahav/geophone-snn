// Plain-language detection readout + status lamps.
// Surfaces the model's count level (single / multiple) per class, the post-process
// "mixed" state (>=2 classes detected), and — in Dataset mode — the ground truth.

import { SVG, NAME } from './icons.js';

const LAMP_CLASSES = ['human', 'car', 'animal'];
const PLURAL = { human: 'humans', car: 'cars', animal: 'animals' };

export function createDetection(els) {
  const { headline, sub, lamps, meta } = els;
  const lampRefs = {};
  const truthBox = document.getElementById('detTruth');
  const truthVal = document.getElementById('detTruthVal');

  function build() {
    lamps.innerHTML = '';
    for (const cls of LAMP_CLASSES) {
      const el = document.createElement('div');
      el.className = 'lamp';
      el.innerHTML = `<span class="lamp-led"></span><span class="lamp-ico">${SVG[cls]}</span>` +
        `<span class="lamp-label">${NAME[cls]}</span><span class="lamp-lvl"></span>`;
      lamps.appendChild(el);
      lampRefs[cls] = { el, lvl: el.querySelector('.lamp-lvl') };
    }
  }

  function onWindow(w) {
    const lv = w.levels || {};
    for (const cls of LAMP_CLASSES) {
      const on = w.detected.includes(cls);
      const multi = on && lv[cls] === 'multiple';
      lampRefs[cls].el.classList.toggle('is-on', on);
      lampRefs[cls].el.classList.toggle('is-multi', multi);
      lampRefs[cls].lvl.textContent = on ? (multi ? '2+' : '1') : '';
    }

    if (w.detected.length === 0) {
      headline.textContent = 'All clear';
      headline.classList.remove('is-hit');
      sub.textContent = 'no source above threshold';
      meta.textContent = 'monitoring';
      return;
    }
    headline.classList.add('is-hit');

    if (w.detected.length >= 2) {                       // mixed = post-process (>=2 classes)
      const parts = w.detected.map((c) => NAME[c] + (lv[c] === 'multiple' ? ' ×2+' : ''));
      headline.textContent = 'Mixed signal';
      sub.textContent = parts.join('  +  ');
      meta.textContent = 'MIXED';
    } else {
      const c = w.detected[0];
      const multi = lv[c] === 'multiple' && c !== 'car';
      headline.textContent = multi ? `Multiple ${PLURAL[c]}` : `${NAME[c]} detected`;
      const conf = Math.round(w.probs[c] * 100);
      sub.textContent = multi ? `confidence ${conf}% · multiple` : `confidence ${conf}%`;
      meta.textContent = 'ALERT';
    }
  }

  // Dataset mode: show the scene's ground truth (null hides the chip)
  function setTruth(truth) {
    if (!truthBox) return;
    if (!truth) { truthBox.hidden = true; return; }
    truthBox.hidden = false;
    const active = truth.active || [];
    if (active.length === 0) {
      truthVal.textContent = 'nothing';
    } else {
      const lvl = { human: truth.human_level, car: truth.car_level, animal: truth.animal_level };
      const parts = active.map((c) => NAME[c] + (lvl[c] === 'multiple' ? ' ×2+' : ''));
      truthVal.textContent = (active.length >= 2 ? 'mixed — ' : '') + parts.join(' + ');
    }
  }

  return { build, onWindow, setTruth };
}
