// Benchmark: lid margin and limbus detection on synthetic eyes with known truth.
//   node test/bench.mjs [count] [--png dir]
// Reports errors in mm along the eye's vertical axis at the reflex column
// (the MRD1 / MRD2 error), and the limbus radius error, for the face-model
// starting guess versus the image-based detector.
import { renderEye, mulberry32 } from './synth.mjs';
import { analyzeEye } from '../js/eyeseg.js';
import zlib from 'node:zlib';
import fs from 'node:fs';

const N = Number(process.argv[2]) || 200;
const pngDir = process.argv.includes('--png') ? process.argv[process.argv.indexOf('--png') + 1] : null;

const dot = (a, b) => a.x * b.x + a.y * b.y;
function polyAtU(F, poly, u) {
  const q = poly.map(p => ({ u: dot(p, F.u), v: dot(p, F.v) }));
  for (let i = 0; i < q.length - 1; i++) {
    if ((q[i].u - u) * (q[i + 1].u - u) <= 0 && q[i].u !== q[i + 1].u) {
      const t = (u - q[i].u) / (q[i + 1].u - q[i].u);
      return q[i].v + t * (q[i + 1].v - q[i].v);
    }
  }
  return NaN;
}

const rnd = mulberry32(12345);
const pick = a => a[Math.floor(rnd() * a.length)];
const rows = [];
let tDet = 0;
for (let n = 0; n < N; n++) {
  const p = {
    seed: 1000 + n,
    r: 22 + rnd() * 50,
    mrd1: -1 + rnd() * 6.5,
    mrd2: 3 + rnd() * 3.5,
    iris: pick(['brown', 'brown', 'dark', 'dark', 'hazel', 'blue']),
    skin: Math.floor(rnd() * 5),
    blur: 0.4 + rnd() * 1.3,
    noise: 2 + rnd() * 6,
    roll: (rnd() - 0.5) * 10,
    lashes: 0.5 + rnd(),
  };
  const s = renderEye(p);
  const { F, mm, truth, model } = s;
  const tA = performance.now();
  const res = analyzeEye(s.img, { F, center: model.center, r0: model.r0, upperPoly: model.upperPoly, lowerPoly: model.lowerPoly, med: model.med, lat: model.lat, reflex: truth.reflexVisible ? truth.reflex : null });
  tDet += performance.now() - tA;
  const uRef = dot(truth.reflex, F.u);
  const vTrueU = dot(truth.upper, F.v), vTrueL = dot(truth.lower, F.v);
  const base = {
    upper: (polyAtU(F, model.upperPoly, uRef) - vTrueU) / mm,
    lower: (polyAtU(F, model.lowerPoly, uRef) - vTrueL) / mm,
    r: (model.r0 - truth.r) / mm,
    c: Math.hypot(model.center.x - truth.irisC.x, model.center.y - truth.irisC.y) / mm,
  };
  const det = {
    upper: res.upper ? (dot(res.upper.pt, F.v) - vTrueU) / mm : null,
    lower: res.lower ? (dot(res.lower.pt, F.v) - vTrueL) / mm : null,
    r: res.iris ? (res.iris.r - truth.r) / mm : null,
    c: res.iris ? Math.hypot(res.iris.c.x - truth.irisC.x, res.iris.c.y - truth.irisC.y) / mm : null,
  };
  rows.push({ p, base, det });
  const cases = process.argv.includes('--case') ? process.argv[process.argv.indexOf('--case') + 1].split(',').map(Number) : null;
  if (pngDir && (cases ? cases.includes(n) : n < 12)) writeOverlay(`${pngDir}/eye_${n}.png`, s, res);
}
const ms = tDet / N;

function stats(vals) {
  const a = vals.filter(v => v != null && Number.isFinite(v)).map(Math.abs).sort((x, y) => x - y);
  if (!a.length) return 'n/a';
  const mean = a.reduce((s, v) => s + v, 0) / a.length;
  return `mean ${mean.toFixed(2)}  p90 ${a[Math.floor(0.9 * (a.length - 1))].toFixed(2)}  max ${a[a.length - 1].toFixed(2)}  ≤0.5mm ${(100 * a.filter(v => v <= 0.5).length / a.length).toFixed(0)}%`;
}
// final = detector when it produced a result, otherwise the model (what the app would show)
const final = k => rows.map(r => (r.det[k] ?? r.base[k]));
const found = k => rows.filter(r => r.det[k] != null).length;
console.log(`${N} synthetic eyes, detector ${ms.toFixed(0)} ms per eye\n`);
for (const [k, name] of [['upper', 'Upper lid at reflex (MRD1 error, mm)'], ['lower', 'Lower lid at reflex (MRD2 error, mm)'], ['r', 'Limbus radius (mm)'], ['c', 'Iris centre (mm)']]) {
  console.log(name);
  console.log(`  face model only : ${stats(rows.map(r => r.base[k]))}`);
  console.log(`  detector        : ${stats(rows.map(r => r.det[k]))}   (found in ${found(k)}/${N})`);
  console.log(`  app result      : ${stats(final(k))}`);
}
// worst cases, to look at
const worst = rows.map((r, i) => ({ i, e: Math.abs(r.det.upper ?? 99) })).sort((a, b) => b.e - a.e).slice(0, 5);
console.log('\nworst upper-lid cases:', worst.map(w => `#${w.i} err=${w.e.toFixed(2)} mrd1=${rows[w.i].p.mrd1.toFixed(1)} iris=${rows[w.i].p.iris} r=${rows[w.i].p.r.toFixed(0)}`).join('\n  '));
const worstL = rows.map((r, i) => ({ i, e: Math.abs(r.det.lower ?? 99) })).sort((a, b) => b.e - a.e).slice(0, 5);
console.log('worst lower-lid cases:', worstL.map(w => `#${w.i} err=${w.e.toFixed(2)} mrd2=${rows[w.i].p.mrd2.toFixed(1)} iris=${rows[w.i].p.iris}`).join('\n  '));

// ---------- PNG overlay for visual inspection ----------
function writeOverlay(file, s, res) {
  const { w, h } = s.img, px = new Uint8Array(s.img.data);
  const put = (x, y, c) => { x = Math.round(x); y = Math.round(y); if (x < 0 || y < 0 || x >= w || y >= h) return; const i = (y * w + x) * 4; px[i] = c[0]; px[i + 1] = c[1]; px[i + 2] = c[2]; };
  const ring = (c, r, col) => { for (let a = 0; a < 6.283; a += 0.5 / r) put(c.x + r * Math.cos(a), c.y + r * Math.sin(a), col); };
  const cross = (p, col) => { for (let d = -6; d <= 6; d++) { put(p.x + d, p.y, col); put(p.x, p.y + d, col); } };
  for (const q of s.model.upperPoly.concat(s.model.lowerPoly)) cross(q, [255, 160, 0]);
  if (res.upper) { res.upper.curve.forEach(q => put(q.x, q.y, [0, 220, 255])); cross(res.upper.pt, [0, 220, 255]); }
  if (res.lower) { res.lower.curve.forEach(q => put(q.x, q.y, [120, 255, 120])); cross(res.lower.pt, [120, 255, 120]); }
  if (res.iris) ring(res.iris.c, res.iris.r, [255, 0, 255]);
  cross(s.truth.upper, [255, 0, 0]); cross(s.truth.lower, [255, 0, 0]);
  fs.writeFileSync(file, png(w, h, px));
}
function png(w, h, rgba) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * 4 + 1)] = 0; Buffer.from(rgba.buffer, rgba.byteOffset + y * w * 4, w * 4).copy(raw, y * (w * 4 + 1) + 1); }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) { c ^= buf[i]; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); }
  return ~c;
}

if (process.argv.includes('--outliers')) {
  for (const [k, lim] of [['upper', 0.8], ['r', 0.5], ['c', 0.6]]) {
    const o = rows.map((r, i) => ({ i, r })).filter(({ r }) => r.det[k] != null && Math.abs(r.det[k]) > lim);
    console.log(`\n${k} outliers (> ${lim} mm):`, o.map(({ i, r }) => `#${i} det=${r.det[k].toFixed(2)} base=${r.base[k].toFixed(2)} mrd1=${r.p.mrd1.toFixed(1)} iris=${r.p.iris} r=${r.p.r.toFixed(0)} blur=${r.p.blur.toFixed(1)} roll=${r.p.roll.toFixed(1)} skin=${r.p.skin}`).join('\n  '));
  }
  const nf = rows.map((r, i) => ({ i, r })).filter(({ r }) => r.det.upper == null);
  console.log('\nupper not found:', nf.map(({ i, r }) => `#${i} mrd1=${r.p.mrd1.toFixed(1)} iris=${r.p.iris} r=${r.p.r.toFixed(0)} irisErr=${r.det.r?.toFixed(2)}`).join('\n  '));
}
