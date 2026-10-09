// On-device learning from the examiner's marker corrections, and validation of
// app measurements against clinical (ruler / slit-lamp) values.
// Nothing leaves the phone: samples live in localStorage.
import { EYES, dist, mid, toUV, fromUV } from './analysis.js?v=13';

const LEARN_KEY = 'ptosis.learn.v1';
const VALID_KEY = 'ptosis.validation.v1';
export const MIN_SAMPLES = 5;     // corrections are applied only after this many eyes
const WINDOW = 40;                // most recent samples used
export const PARAMS = [['mrd1', 'MRD1'], ['mrd2', 'MRD2'], ['pfh', 'PFH'], ['mcd', 'MCD'], ['lf', 'LF']];

const read = (k, d) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } };
const write = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } };

export const loadSamples = () => read(LEARN_KEY, []);
export const loadValidation = () => read(VALID_KEY, []);
export function resetLearning() { write(LEARN_KEY, []); }
export function resetValidation() { write(VALID_KEY, []); }

const deep = o => JSON.parse(JSON.stringify(o));
const median = a => { const s = a.slice().sort((x, y) => x - y); const n = s.length; return n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2; };

// ---------- learning ----------

// Learned correction per marker: median of recent corrections, shrunk toward
// zero while there are few samples so one odd patient can't skew it.
export function learnedBias(samples = loadSamples()) {
  const out = {};
  for (const k of ['upper', 'lower', 'crease']) {
    const v = samples.map(s => s[k]).filter(x => typeof x === 'number').slice(-WINDOW);
    out[k] = { n: v.length, mm: v.length >= MIN_SAMPLES ? median(v) * v.length / (v.length + MIN_SAMPLES) : 0 };
  }
  const l = samples.map(s => s.limbus).filter(x => typeof x === 'number').slice(-WINDOW);
  out.limbus = { n: l.length, scale: l.length >= MIN_SAMPLES ? 1 + (median(l) - 1) * l.length / (l.length + MIN_SAMPLES) : 1 };
  out.eyes = samples.length;
  return out;
}

// Apply the learned correction to a fresh automatic primary capture. The raw
// automatic markers are kept in cap.rawH so later corrections are measured
// against what the detector produced, not against an earlier correction.
// `measure(cap)` must return { F, mmpp } for the capture.
export function applyLearned(cap, bias, measure) {
  cap.rawH = { OD: deep(cap.eyes.OD.h), OS: deep(cap.eyes.OS.h) };
  cap.learned = null;
  if (!bias || cap.manual) return;
  const used = {};
  if (bias.limbus.scale !== 1) {
    for (const s of EYES) {
      const h = cap.eyes[s].h;
      if (!h.limbN) continue;
      const c = mid(h.limbN, h.limbT);
      for (const k of ['limbN', 'limbT']) h[k] = { x: c.x + (h[k].x - c.x) * bias.limbus.scale, y: c.y + (h[k].y - c.y) * bias.limbus.scale };
      cap.eyes[s].irisRpx = dist(h.limbN, h.limbT) / 2;
    }
    used.limbus = bias.limbus.scale;
  }
  const { F, mmpp } = measure(cap);
  for (const k of ['upper', 'lower', 'crease']) {
    const mm = bias[k].mm;
    if (!mm) continue;
    for (const s of EYES) {
      if (k === 'crease' && !cap.eyes[s].flags.crease) continue; // fallback crease is a guess, not a detection
      const p = toUV(F, cap.eyes[s].h[k]);
      cap.eyes[s].h[k] = fromUV(F, p.u, p.v + mm / mmpp);
    }
    used[k] = mm;
  }
  cap.learned = Object.keys(used).length ? used : null;
}

// One sample per eye: how far the examiner moved each marker from the raw
// automatic position (mm along the vertical axis; + = moved down), and the
// limbus diameter ratio. Replaces earlier samples from the same capture.
export function recordCorrections(cap, measure) {
  if (!cap || cap.kind !== 'primary' || cap.manual || !cap.rawH) return 0;
  const { F, mmpp } = measure(cap);
  const dv = (a, b) => (toUV(F, a).v - toUV(F, b).v) * mmpp;
  const samples = loadSamples().filter(x => x.cap !== cap.time);
  let added = 0;
  for (const s of EYES) {
    const h = cap.eyes[s].h, r = cap.rawH[s], f = cap.eyes[s].flags;
    const smp = { cap: cap.time, t: Date.now(), eye: s, upper: dv(h.upper, r.upper), lower: dv(h.lower, r.lower) };
    const creaseMoved = Math.abs(dv(h.crease, r.crease)) > 0.2;
    if (f.crease || creaseMoved) smp.crease = dv(h.crease, r.crease);
    if (h.limbN && r.limbN) smp.limbus = dist(h.limbN, h.limbT) / dist(r.limbN, r.limbT);
    samples.push(smp); added++;
  }
  write(LEARN_KEY, samples.slice(-500));
  return added;
}

// ---------- validation against clinical values ----------

export function recordValidation(entry) {
  const list = loadValidation().filter(x => x.cap !== entry.cap);
  list.push(entry);
  write(VALID_KEY, list.slice(-1000));
}

// Bland–Altman statistics (app − clinical) per parameter, pooling both eyes.
export function validationStats(list = loadValidation()) {
  const out = {};
  for (const [k] of PARAMS) {
    const pairs = [];
    for (const e of list) for (const s of EYES) {
      const a = e.app?.[s]?.[k], c = e.ref?.[s]?.[k];
      if (typeof a === 'number' && typeof c === 'number' && Number.isFinite(a) && Number.isFinite(c)) pairs.push({ a, c, d: a - c, m: (a + c) / 2 });
    }
    const n = pairs.length;
    if (!n) { out[k] = { n }; continue; }
    const bias = pairs.reduce((t, p) => t + p.d, 0) / n;
    const sd = n > 1 ? Math.sqrt(pairs.reduce((t, p) => t + (p.d - bias) ** 2, 0) / (n - 1)) : 0;
    out[k] = {
      n, bias, sd, lo: bias - 1.96 * sd, hi: bias + 1.96 * sd,
      mae: pairs.reduce((t, p) => t + Math.abs(p.d), 0) / n,
      within1: pairs.filter(p => Math.abs(p.d) <= 1).length / n,
      pairs,
    };
  }
  return out;
}

export function describeLearned(L) {
  if (!L) return '';
  return Object.entries(L).map(([k, v]) => (k === 'limbus'
    ? `limbus ×${v.toFixed(3)}`
    : `${{ upper: 'upper lid', lower: 'lower lid', crease: 'crease' }[k]} ${v > 0 ? '+' : ''}${v.toFixed(2)} mm`)).join(', ');
}

export function validationCSV(list = loadValidation()) {
  const head = ['date', 'patient', 'eye', ...PARAMS.flatMap(([k]) => [`app_${k}`, `clinical_${k}`]), 'learned_correction'];
  const rows = [head.join(',')];
  const q = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  for (const e of list) for (const s of EYES) {
    const vals = PARAMS.flatMap(([k]) => [e.app?.[s]?.[k], e.ref?.[s]?.[k]].map(v => (typeof v === 'number' ? v.toFixed(2) : '')));
    rows.push([new Date(e.t).toISOString(), q(e.id), s, ...vals, q(describeLearned(e.learned))].join(','));
  }
  return rows.join('\n');
}

// Bland–Altman plot as inline SVG.
export function blandAltmanSVG(st, label) {
  if (!st || st.n < 2) return '';
  const W = 320, H = 190, P = { l: 36, r: 10, t: 14, b: 26 };
  const xs = st.pairs.map(p => p.m), ys = st.pairs.map(p => p.d);
  const x0 = Math.min(...xs) - 0.5, x1 = Math.max(...xs) + 0.5;
  const yr = Math.max(1.5, ...ys.map(Math.abs), Math.abs(st.lo), Math.abs(st.hi)) * 1.15;
  const X = x => P.l + (x - x0) / (x1 - x0) * (W - P.l - P.r);
  const Y = y => P.t + (yr - y) / (2 * yr) * (H - P.t - P.b);
  // label above (dy<0) or below (dy>0) its line, at the right or left end
  const hl = (y, cls, txt, dy, left) => `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(y)}" y2="${Y(y)}" class="${cls}"/><text x="${left ? P.l + 4 : W - P.r}" y="${Y(y) + dy}" text-anchor="${left ? 'start' : 'end'}" class="ba-t">${txt}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" class="ba" role="img" aria-label="Bland–Altman plot for ${label}">
    <line x1="${P.l}" x2="${P.l}" y1="${P.t}" y2="${H - P.b}" class="ba-axis"/>
    <line x1="${P.l}" x2="${W - P.r}" y1="${Y(0)}" y2="${Y(0)}" class="ba-axis"/>
    ${hl(st.bias, 'ba-bias', `bias ${st.bias.toFixed(2)}`, -3, true)}
    ${hl(st.hi, 'ba-loa', `+1.96 SD ${st.hi.toFixed(2)}`, -3)}
    ${hl(st.lo, 'ba-loa', `−1.96 SD ${st.lo.toFixed(2)}`, 10)}
    ${st.pairs.map(p => `<circle cx="${X(p.m)}" cy="${Y(p.d)}" r="3.5" class="ba-pt"/>`).join('')}
    <text x="${P.l}" y="${H - 6}" class="ba-t">mean of app &amp; clinical (mm)</text>
    <text x="4" y="${P.t + 8}" class="ba-t">app − clinical</text>
  </svg>`;
}
