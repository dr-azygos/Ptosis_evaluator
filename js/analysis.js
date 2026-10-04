// Geometry, image analysis and clinical measurement logic.
// All image-space points are {x, y} in source pixels.
// "OD" = patient's right eye, "OS" = patient's left eye.

export const LM = {
  OD: {
    outer: 33, inner: 133,
    upper: [33, 246, 161, 160, 159, 158, 157, 173, 133],
    lower: [33, 7, 163, 144, 145, 153, 154, 155, 133],
    browLower: [46, 53, 52, 65, 55],
  },
  OS: {
    outer: 263, inner: 362,
    upper: [263, 466, 388, 387, 386, 385, 384, 398, 362],
    lower: [263, 249, 390, 373, 374, 380, 381, 382, 362],
    browLower: [276, 283, 282, 295, 285],
  },
  IRIS_A: [468, 469, 470, 471, 472],
  IRIS_B: [473, 474, 475, 476, 477],
  NOSE_TIP: 1,
};

export const EYES = ['OD', 'OS'];

// ---------- vector helpers ----------
export const dot = (a, b) => a.x * b.x + a.y * b.y;
export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

// Orthonormal measurement frame: u runs from OD toward OS (horizontal),
// v is perpendicular and points toward the patient's chin. Vertical distances
// are taken along v so a tilted head does not inflate MRD values.
// `nose` anchors "down" to the face, so a rotated or upside-down image (phone
// held sideways, photo with unapplied rotation) cannot flip the signs.
export function makeFrame(pOD, pOS, nose) {
  const dx = pOS.x - pOD.x, dy = pOS.y - pOD.y;
  const len = Math.hypot(dx, dy) || 1;
  let u = { x: dx / len, y: dy / len };
  let v = { x: -u.y, y: u.x };
  const flip = nose
    ? dot({ x: nose.x - (pOD.x + pOS.x) / 2, y: nose.y - (pOD.y + pOS.y) / 2 }, v) < 0
    : v.y < 0;
  if (flip) v = { x: -v.x, y: -v.y };
  // Head tilt relative to the nearest image axis, so a phone held sideways
  // with an upright face still reads as level.
  const a = Math.atan2(dy, dx) * 180 / Math.PI;
  const rollDeg = a - 90 * Math.round(a / 90);
  return { u, v, rollDeg };
}
export const toUV = (F, p) => ({ u: dot(p, F.u), v: dot(p, F.v) });
export const fromUV = (F, u, v) => ({ x: F.u.x * u + F.v.x * v, y: F.u.y * u + F.v.y * v });

// Point on a polyline at horizontal position u (in frame F).
export function polyAt(F, pts, u) {
  const q = pts.map(p => toUV(F, p));
  for (let i = 0; i < q.length - 1; i++) {
    const a = q[i], b = q[i + 1];
    if (a.u === b.u) continue;
    if ((a.u - u) * (b.u - u) <= 0) {
      const t = (u - a.u) / (b.u - a.u);
      return fromUV(F, u, a.v + t * (b.v - a.v));
    }
  }
  return null;
}

// Highest point (smallest v) on a polyline, interpolated finely.
export function polyApex(F, pts) {
  const q = pts.map(p => toUV(F, p));
  let best = q[0];
  for (let i = 0; i < q.length - 1; i++) {
    for (let t = 0; t <= 1; t += 0.1) {
      const c = { u: q[i].u + t * (q[i + 1].u - q[i].u), v: q[i].v + t * (q[i + 1].v - q[i].v) };
      if (c.v < best.v) best = c;
    }
  }
  return fromUV(F, best.u, best.v);
}

// ---------- landmarks → geometry ----------
function px(lms, i, W, H) { return { x: lms[i].x * W, y: lms[i].y * H }; }

function iris(lms, idx, W, H) {
  const c = px(lms, idx[0], W, H);
  let r = 0;
  for (let k = 1; k < 5; k++) r += dist(c, px(lms, idx[k], W, H));
  return { c, r: r / 4 };
}

export function extractGeometry(lms, W, H) {
  const irisA = iris(lms, LM.IRIS_A, W, H), irisB = iris(lms, LM.IRIS_B, W, H);
  const g = { W, H, nose: px(lms, LM.NOSE_TIP, W, H) };
  for (const s of EYES) {
    const L = LM[s];
    const outer = px(lms, L.outer, W, H), inner = px(lms, L.inner, W, H);
    const m = mid(outer, inner);
    // Assign iris by proximity rather than relying on index convention.
    const ir = dist(irisA.c, m) < dist(irisB.c, m) ? irisA : irisB;
    g[s] = {
      iris: ir, outer, inner,
      upper: L.upper.map(i => px(lms, i, W, H)),
      lower: L.lower.map(i => px(lms, i, W, H)),
      brow: L.browLower.map(i => px(lms, i, W, H)),
    };
  }
  return g;
}

// ---------- pixel sampling ----------
export class Sampler {
  constructor() {
    this.c = document.createElement('canvas');
    this.ctx = this.c.getContext('2d', { willReadFrequently: true });
  }
  setSource(src, W, H) { this.src = src; this.W = W; this.H = H; }
  patch(x, y, w, h) {
    x = Math.max(0, Math.floor(x)); y = Math.max(0, Math.floor(y));
    w = Math.min(this.W - x, Math.ceil(w)); h = Math.min(this.H - y, Math.ceil(h));
    if (w < 3 || h < 3) return null;
    this.c.width = w; this.c.height = h;
    this.ctx.drawImage(this.src, x, y, w, h, 0, 0, w, h);
    const d = this.ctx.getImageData(0, 0, w, h).data;
    const n = w * h, gray = new Float32Array(n), minc = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const r = d[j], g = d[j + 1], b = d[j + 2];
      gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      minc[i] = Math.min(r, g, b);
    }
    return { x, y, w, h, gray, minc };
  }
}

function sampleGray(P, p) {
  const ix = Math.round(p.x - P.x), iy = Math.round(p.y - P.y);
  if (ix < 0 || iy < 0 || ix >= P.w || iy >= P.h) return NaN;
  return P.gray[iy * P.w + ix];
}

// Corneal light reflex: brightest near-white blob inside the iris.
export function findReflex(sampler, ir) {
  const r = ir.r, R = r * 0.92;
  const P = sampler.patch(ir.c.x - r, ir.c.y - r, 2 * r + 1, 2 * r + 1);
  if (!P) return null;
  const inside = new Uint8Array(P.w * P.h);
  const vals = [];
  let max = -1, maxI = -1;
  for (let y = 0; y < P.h; y++) {
    for (let x = 0; x < P.w; x++) {
      const dx = P.x + x + 0.5 - ir.c.x, dy = P.y + y + 0.5 - ir.c.y;
      if (dx * dx + dy * dy > R * R) continue;
      const i = y * P.w + x;
      inside[i] = 1;
      const m = P.minc[i];
      vals.push(m);
      if (m > max) { max = m; maxI = i; }
    }
  }
  if (vals.length < 20) return null;
  vals.sort((a, b) => a - b);
  const med = vals[vals.length >> 1];
  if (max < 140 || max - med < 45) return null;
  const thr = max - Math.max(10, (max - med) * 0.3);
  // Flood-fill the blob containing the brightest pixel.
  const seen = new Uint8Array(P.w * P.h);
  const stack = [maxI];
  seen[maxI] = 1;
  let sw = 0, sx = 0, sy = 0, count = 0;
  while (stack.length) {
    const i = stack.pop();
    const x = i % P.w, y = (i / P.w) | 0;
    const wgt = P.minc[i] - thr + 1;
    sw += wgt; sx += wgt * x; sy += wgt * y; count++;
    for (const j of [i - 1, i + 1, i - P.w, i + P.w]) {
      if (j < 0 || j >= seen.length || seen[j] || !inside[j]) continue;
      if (Math.abs((j % P.w) - x) > 1) continue;
      if (P.minc[j] < thr) continue;
      seen[j] = 1; stack.push(j);
    }
  }
  if (count > Math.PI * R * R * 0.1) return null; // diffuse glare, not a reflex
  return { x: P.x + sx / sw + 0.5, y: P.y + sy / sw + 0.5, size: count, contrast: max - med };
}

// Upper lid crease: most prominent dark valley in the vertical intensity
// profile above the lid margin (2–14 mm), averaged over ±1.2 mm horizontally.
export function findCrease(sampler, F, upper, brow, mmpp) {
  const U = toUV(F, upper);
  const vStart = U.v - 2 / mmpp;
  let vEnd = U.v - 14 / mmpp;
  if (brow) vEnd = Math.max(vEnd, toUV(F, brow).v + 1.5 / mmpp);
  const n = Math.round(vStart - vEnd);
  if (n < 3 / mmpp) return null;
  const hw = 1.2 / mmpp;
  const corners = [[-hw, vStart], [hw, vStart], [-hw, vEnd], [hw, vEnd]].map(([du, v]) => fromUV(F, U.u + du, v));
  const xs = corners.map(c => c.x), ys = corners.map(c => c.y);
  const P = sampler.patch(Math.min(...xs) - 1, Math.min(...ys) - 1,
    Math.max(...xs) - Math.min(...xs) + 3, Math.max(...ys) - Math.min(...ys) + 3);
  if (!P) return null;
  const prof = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0, c = 0;
    for (let j = -2; j <= 2; j++) {
      const g = sampleGray(P, fromUV(F, U.u + j * hw / 2, vStart - k));
      if (!Number.isNaN(g)) { s += g; c++; }
    }
    prof[k] = c ? s / c : NaN;
  }
  const rad = Math.max(1, Math.round(0.25 / mmpp));
  const sm = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, k - rad); j <= Math.min(n - 1, k + rad); j++) if (!Number.isNaN(prof[j])) { s += prof[j]; c++; }
    sm[k] = c ? s / c : 255;
  }
  const win = Math.max(2, Math.round(1.5 / mmpp));
  let best = -1, bestProm = 0;
  for (let k = win; k < n - win; k++) {
    let isMin = true;
    for (let j = k - 2; j <= k + 2; j++) if (sm[j] < sm[k]) { isMin = false; break; }
    if (!isMin) continue;
    let lmax = 0, rmax = 0;
    for (let j = k - win; j < k; j++) lmax = Math.max(lmax, sm[j]);
    for (let j = k + 1; j <= k + win; j++) rmax = Math.max(rmax, sm[j]);
    const prom = Math.min(lmax, rmax) - sm[k];
    if (prom > bestProm) { bestProm = prom; best = k; }
  }
  if (best < 0 || bestProm < 4) return null;
  return { pt: fromUV(F, U.u, vStart - best), prominence: bestProm };
}

// ---------- auto-placement of measurement handles ----------
export function autoPrimary(g, sampler, opts = {}) {
  const irisDiamPx = g.OD.iris.r + g.OS.iris.r; // mean diameter
  const mmpp = opts.hvid / irisDiamPx;
  const reflex = {}, found = {};
  for (const s of EYES) {
    const det = sampler ? findReflex(sampler, g[s].iris) : null;
    found[s] = !!det;
    reflex[s] = det ? { x: det.x, y: det.y } : { ...g[s].iris.c };
  }
  const F = makeFrame(reflex.OD, reflex.OS, g.nose);
  const eyes = {};
  for (const s of EYES) {
    const e = g[s], R = toUV(F, reflex[s]);
    const upper = polyAt(F, e.upper, R.u) || polyApex(F, e.upper);
    const lower = polyAt(F, e.lower, R.u) || fromUV(F, R.u, toUV(F, e.lower[4]).v);
    const brow = polyAt(F, e.brow, R.u) || e.brow[2];
    let crease = null;
    if (sampler && !opts.skipCrease) crease = findCrease(sampler, F, upper, brow, mmpp);
    const up = toUV(F, upper);
    eyes[s] = {
      h: {
        reflex: reflex[s], upper, lower, brow,
        crease: crease ? crease.pt : fromUV(F, up.u, up.v - 7 / mmpp),
        med: e.inner, lat: e.outer,
      },
      flags: { reflex: found[s], crease: !!crease },
      irisRpx: e.iris.r,
    };
  }
  return { kind: 'primary', irisDiamPx, eyes, nose: { ...g.nose } };
}

export function autoGaze(g, kind) {
  const F = makeFrame(g.OD.inner, g.OS.inner, g.nose);
  const eyes = {};
  for (const s of EYES) {
    eyes[s] = { h: { upper: polyApex(F, g[s].upper), med: g[s].inner, lat: g[s].outer }, flags: {}, irisRpx: g[s].iris.r };
  }
  return { kind, irisDiamPx: g.OD.iris.r + g.OS.iris.r, eyes, nose: { ...g.nose } };
}

// ---------- measurements ----------
export function primaryScale(cap, settings) {
  if (settings.calib === 'ruler' && cap.ruler) {
    const d = dist(cap.ruler.a, cap.ruler.b);
    if (d > 2) return { mmpp: settings.rulerMm / d, method: `ruler ${settings.rulerMm} mm` };
  }
  return { mmpp: settings.hvid / cap.irisDiamPx, method: `HVID ${settings.hvid} mm` };
}

export function measurePrimary(cap, settings) {
  const { mmpp, method } = primaryScale(cap, settings);
  const F = makeFrame(cap.eyes.OD.h.reflex, cap.eyes.OS.h.reflex, cap.nose);
  const out = { mmpp, method, F, eyes: {} };
  for (const s of EYES) {
    const { h, flags, irisRpx } = cap.eyes[s];
    const v = p => toUV(F, p).v;
    const mrd1 = (v(h.reflex) - v(h.upper)) * mmpp;
    const mrd2 = (v(h.lower) - v(h.reflex)) * mmpp;
    const irisR = irisRpx * mmpp;
    out.eyes[s] = {
      mrd1, mrd2, pfh: mrd1 + mrd2,
      pfw: dist(h.med, h.lat) * mmpp,
      mcd: (v(h.upper) - v(h.crease)) * mmpp,
      brow: (v(h.reflex) - v(h.brow)) * mmpp,
      coverage: Math.max(0, irisR - mrd1), // mm of cornea covered by upper lid
      scleralShowInf: Math.max(0, mrd2 - irisR),
      reflexFound: flags.reflex, creaseAuto: flags.crease,
    };
  }
  out.icd = dist(cap.eyes.OD.h.med, cap.eyes.OS.h.med) * mmpp;
  out.ipd = dist(cap.eyes.OD.h.reflex, cap.eyes.OS.h.reflex) * mmpp;
  return out;
}

// Upper-lid apex height above the intercanthal line of that eye (mm).
export function measureGaze(cap, settings, icdMm) {
  const icdPx = dist(cap.eyes.OD.h.med, cap.eyes.OS.h.med);
  const mmpp = icdMm ? icdMm / icdPx : settings.hvid / cap.irisDiamPx;
  const F = makeFrame(cap.eyes.OD.h.med, cap.eyes.OS.h.med, cap.nose);
  const out = { mmpp, F, eyes: {} };
  for (const s of EYES) {
    const { h } = cap.eyes[s];
    const M = toUV(F, h.med), L = toUV(F, h.lat), U = toUV(F, h.upper);
    const t = (U.u - M.u) / ((L.u - M.u) || 1);
    const vLine = M.v + t * (L.v - M.v);
    out.eyes[s] = { lidHeight: (vLine - U.v) * mmpp };
  }
  return out;
}

// ---------- clinical interpretation ----------
export function ptosisGrade(mrd1, normal) {
  const amt = normal - mrd1;
  if (amt < 1) return { amount: Math.max(0, amt), label: 'No significant ptosis', level: 0 };
  if (amt <= 2) return { amount: amt, label: 'Mild ptosis', level: 1 };
  if (amt <= 3.5) return { amount: amt, label: 'Moderate ptosis', level: 2 };
  return { amount: amt, label: 'Severe ptosis', level: 3 };
}

export function lfGrade(lf) {
  if (lf >= 12) return 'Excellent';
  if (lf >= 8) return 'Good';
  if (lf >= 5) return 'Fair';
  return 'Poor';
}

// Educational pointers only — final decisions rest with the surgeon.
export function suggestions(m, lf, clinical, normal) {
  const out = [];
  if (!m) return out;
  for (const s of EYES) {
    const e = m.eyes[s];
    const g = ptosisGrade(e.mrd1, normal);
    if (g.level === 0) continue;
    const parts = [];
    const l = lf ? lf[s] : null;
    if (l != null) {
      if (l <= 4) parts.push('poor LF → frontalis sling');
      else if (l < 8) parts.push('fair LF → levator resection');
      else if (g.level === 1 && clinical.phenylephrine === 'positive') parts.push('good LF, mild, phenylephrine +ve → Müller muscle–conjunctival resection (MMCR) or levator advancement');
      else parts.push('good LF → levator aponeurosis advancement / repair');
    } else {
      parts.push('measure levator function (capture Down & Up gaze)');
    }
    if (e.mcd > 10) parts.push('high lid crease suggests aponeurotic (involutional) ptosis');
    if (!e.creaseAuto) parts.push('crease not detected automatically — confirm manually (faint/absent crease is common in congenital ptosis)');
    if (e.mrd1 <= 2) parts.push('MRD1 ≤ 2 mm — likely visually significant; consider superior visual field testing');
    out.push(`${s}: ${g.label} (~${g.amount.toFixed(1)} mm). ${parts.join('; ')}.`);
  }
  if (EYES.some(s => m.eyes[s].pfh <= 0 || m.eyes[s].brow <= 0))
    out.unshift('⚠ Implausible values (negative fissure height or brow distance): markers are inverted or misplaced. Re-capture with the face upright, or correct the markers.');
  if (Math.abs(m.eyes.OD.mrd1 - m.eyes.OS.mrd1) >= 1.5)
    out.push('Asymmetry ≥ 1.5 mm — check for Hering\'s dependence (lift the ptotic lid and re-check the fellow eye).');
  if (clinical.jawwink) out.push('Jaw-winking noted — consider Marcus Gunn synkinesis before planning surgery.');
  if (clinical.fatigue) out.push('Fatigability / variability noted — rule out ocular myasthenia (ice-pack test, AChR antibodies).');
  if (clinical.bells === 'poor' || clinical.bells === 'absent') out.push('Poor Bell\'s phenomenon — under-correct and protect the cornea.');
  return out;
}
