// Image-based detection of the limbus (corneal edge) and the lid margins.
//
// Each eye is resampled into a small upright patch (the eye-aligned frame,
// iris radius ≈ 32 px), so the detectors below work at a fixed scale whatever
// the camera distance. The face model only supplies the starting guesses.
//
//  - Limbus: Daugman's integro-differential operator. For candidate circles it
//    finds the radius where the mean brightness along the circle jumps most
//    (dark iris → bright sclera), using only the side arcs the lids rarely cover.
//  - Lid margins: a dynamic-programming path across the palpebral fissure that
//    follows the margin edge (lid/lashes → globe for the upper lid, globe → lid
//    for the lower) while staying smooth, then a robust quadratic fit gives the
//    margin height exactly at the reflex column.
//
// Pure functions on RGBA buffers: no DOM, so it runs in the browser and in Node.

const R = 32; // iris radius in patch pixels

const toUV = (F, p) => ({ u: p.x * F.u.x + p.y * F.u.y, v: p.x * F.v.x + p.y * F.v.y });
const fromUV = (F, u, v) => ({ x: F.u.x * u + F.v.x * v, y: F.u.y * u + F.v.y * v });

// ---------- resampling ----------

// img: { data: RGBA, w, h, x0, y0 } (a window of the full image at offset x0, y0)
function sampleRGB(img, x, y, out) {
  let fx = x - img.x0 - 0.5, fy = y - img.y0 - 0.5;
  fx = Math.max(0, Math.min(img.w - 1.001, fx)); fy = Math.max(0, Math.min(img.h - 1.001, fy));
  const x0 = fx | 0, y0 = fy | 0, ax = fx - x0, ay = fy - y0, d = img.data, w = img.w;
  const i00 = (y0 * w + x0) * 4, i10 = i00 + 4, i01 = i00 + w * 4, i11 = i01 + 4;
  for (let c = 0; c < 3; c++) {
    out[c] = (d[i00 + c] * (1 - ax) + d[i10 + c] * ax) * (1 - ay) + (d[i01 + c] * (1 - ax) + d[i11 + c] * ax) * ay;
  }
}

export function rectify(img, F, center, r0) {
  const k = R / r0; // patch px per image px
  const left = Math.round(3.4 * R), right = Math.round(3.4 * R), top = Math.round(2.4 * R), bottom = Math.round(2.1 * R);
  const W = left + right, H = top + bottom;
  const C = toUV(F, center);
  const ss = Math.max(1, Math.min(4, Math.ceil(1 / k))); // supersample when shrinking
  const Y = new Float32Array(W * H), S = new Float32Array(W * H);
  const rgb = [0, 0, 0];
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      let y = 0, s = 0;
      for (let a = 0; a < ss; a++) for (let b = 0; b < ss; b++) {
        const u = C.u + (i + (a + 0.5) / ss - 0.5 - left) / k;
        const v = C.v + (j + (b + 0.5) / ss - 0.5 - top) / k;
        const p = fromUV(F, u, v);
        sampleRGB(img, p.x, p.y, rgb);
        y += 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        s += Math.max(rgb[0], rgb[1], rgb[2]) - Math.min(rgb[0], rgb[1], rgb[2]);
      }
      Y[j * W + i] = y / (ss * ss); S[j * W + i] = s / (ss * ss);
    }
  }
  return {
    W, H, Y, S, k, ox: left, oy: top,
    toImg: (i, j) => fromUV(F, C.u + (i - left) / k, C.v + (j - top) / k),
    toPatch: p => { const q = toUV(F, p); return { i: (q.u - C.u) * k + left, j: (q.v - C.v) * k + top }; },
  };
}

function blur(src, W, H, sigma) {
  const rad = Math.max(1, Math.ceil(sigma * 2.5)), ker = [];
  let sum = 0;
  for (let t = -rad; t <= rad; t++) { const g = Math.exp(-(t * t) / (2 * sigma * sigma)); ker.push(g); sum += g; }
  for (let t = 0; t < ker.length; t++) ker[t] /= sum;
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    let a = 0;
    for (let t = -rad; t <= rad; t++) a += src[j * W + Math.max(0, Math.min(W - 1, i + t))] * ker[t + rad];
    tmp[j * W + i] = a;
  }
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    let a = 0;
    for (let t = -rad; t <= rad; t++) a += tmp[Math.max(0, Math.min(H - 1, j + t)) * W + i] * ker[t + rad];
    out[j * W + i] = a;
  }
  return out;
}

function bil(A, W, H, x, y) {
  if (x < 0 || y < 0 || x > W - 1.001 || y > H - 1.001) return NaN;
  const x0 = x | 0, y0 = y | 0, ax = x - x0, ay = y - y0, i = y0 * W + x0;
  return (A[i] * (1 - ax) + A[i + 1] * ax) * (1 - ay) + (A[i + W] * (1 - ax) + A[i + W + 1] * ax) * ay;
}

// ---------- limbus ----------

// Side arcs (0° = toward the lateral side of the patch, + = downward). The
// lower-lateral sectors are favoured because the upper lid covers the top of
// the cornea far more often than the lower lid covers the bottom.
const ARC_R = [], ARC_L = [];
for (let n = 0; n < 40; n++) {
  const a = (-25 + n * (80 / 39)) * Math.PI / 180;
  ARC_R.push([Math.cos(a), Math.sin(a)]);
  ARC_L.push([-Math.cos(a), Math.sin(a)]);
}

function ringMean(Ys, W, H, ci, cj, r, arc, skip) {
  let s = 0, n = 0;
  for (const [c, sn] of arc) {
    const x = ci + c * r, y = cj + sn * r;
    if (skip && Math.hypot(x - skip.i, y - skip.j) < skip.rad) continue; // the light reflex
    const v = bil(Ys, W, H, x, y);
    if (!Number.isNaN(v)) { s += v; n++; }
  }
  return n > arc.length * 0.6 ? s / n : NaN;
}

function irisScore(Ys, W, H, ci, cj, rMin, rMax, dr) {
  const rs = [];
  for (let r = rMin - 2 * dr; r <= rMax + 2 * dr + 1e-6; r += dr) rs.push(r);
  const mr = rs.map(r => ringMean(Ys, W, H, ci, cj, r, ARC_R));
  const ml = rs.map(r => ringMean(Ys, W, H, ci, cj, r, ARC_L));
  let best = -Infinity, br = 0, bdr = 0, bdl = 0;
  const wide = Math.max(2, Math.round(4 / dr));
  for (let t = 2; t < rs.length - 2; t++) {
    // smoothed radial derivative on each side
    const dR = (mr[t + 1] + mr[t + 2] - mr[t - 1] - mr[t - 2]) / 2;
    const dL = (ml[t + 1] + ml[t + 2] - ml[t - 1] - ml[t - 2]) / 2;
    if (Number.isNaN(dR) || Number.isNaN(dL)) continue;
    // both sides must show the edge; the weaker side counts double
    let sc = dR + dL - Math.abs(dR - dL) * 0.5;
    // large-scale contrast: bright sclera outside, iris inside (a texture ring
    // inside the iris has iris on both sides)
    const tin = Math.max(0, t - wide), tout = Math.min(rs.length - 1, t + wide);
    const cw = (mr[tout] - mr[tin]) + (ml[tout] - ml[tin]);
    if (Number.isFinite(cw)) sc += 0.35 * cw;
    if (sc > best) { best = sc; br = rs[t]; bdr = dR; bdl = dL; }
  }
  return { score: best, r: br, dR: bdr, dL: bdl };
}

export function fitIris(P, pupil) {
  const Ys = blur(P.Y, P.W, P.H, 1.0);
  // Mild preference for the face model's estimate (patch centre, radius R), so
  // a weak spurious ring in a low-contrast or blurred iris cannot win.
  const prior = (di, dj, r) => 14 * Math.abs(r / R - 1) + 10 * Math.hypot(di, dj) / R;
  // The limbus is near-concentric with the pupil and well outside it.
  const rMin = Math.max(0.72 * R, pupil ? pupil.r * 1.6 : 0), rMax = 1.38 * R;
  const c0i = pupil ? pupil.ci : P.ox, c0j = pupil ? pupil.cj : P.oy;
  let best = { score: -Infinity };
  const span = Math.round((pupil ? 0.16 : 0.32) * R);
  for (let dj = -span; dj <= span; dj += 2) {
    for (let di = -span; di <= span; di += 2) {
      const s = irisScore(Ys, P.W, P.H, c0i + di, c0j + dj, rMin, rMax, 1);
      s.score -= prior(c0i + di - P.ox, c0j + dj - P.oy, s.r);
      if (s.score > best.score) best = { ...s, ci: c0i + di, cj: c0j + dj };
    }
  }
  if (!Number.isFinite(best.score)) return null;
  // refine centre and radius on a finer grid
  let fine = best;
  for (let dj = -2; dj <= 2; dj += 0.5) {
    for (let di = -2; di <= 2; di += 0.5) {
      const s = irisScore(Ys, P.W, P.H, best.ci + di, best.cj + dj, best.r - 3, best.r + 3, 0.25);
      s.score -= prior(best.ci + di - P.ox, best.cj + dj - P.oy, s.r);
      if (s.score > fine.score) fine = { ...s, ci: best.ci + di, cj: best.cj + dj };
    }
  }
  // contrast of the edge relative to the local brightness spread
  const conf = Math.min(fine.dR, fine.dL) / 12;
  return { ci: fine.ci, cj: fine.cj, r: fine.r, conf };
}

// Pupil: strongest dark → bright step moving outward from the iris centre,
// sampled on the lower and side arcs (the top may be under the lid).
const ARC_P = [];
for (let n = 0; n < 48; n++) { const a = (-20 + n * (220 / 47)) * Math.PI / 180; ARC_P.push([Math.cos(a), Math.sin(a)]); }
export function fitPupil(P, iris, span = 4, refl = null) {
  const skip = refl ? { i: refl.i, j: refl.j, rad: 0.25 * R } : null;
  const Ys = blur(P.Y, P.W, P.H, 0.8);
  let best = { score: -Infinity };
  const step = span > 6 ? 2 : 1;
  for (let dj = -span; dj <= span; dj += step) for (let di = -span; di <= span; di += step) {
    const ci = iris.ci + di, cj = iris.cj + dj;
    const rs = [], m = [];
    for (let r = 0.12 * iris.r; r <= 0.6 * iris.r; r += 0.5) { rs.push(r); m.push(ringMean(Ys, P.W, P.H, ci, cj, r, ARC_P, skip)); }
    for (let t = 2; t < rs.length - 2; t++) {
      const d = (m[t + 1] + m[t + 2] - m[t - 1] - m[t - 2]) / 2;
      if (d > best.score) best = { score: d, ci, cj, r: rs[t] };
    }
  }
  return best.score > 8 ? best : null;
}

// ---------- lid margins ----------

function quantile(arr, q) {
  if (!arr.length) return 0;
  const s = Float32Array.from(arr).sort();
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

// Robust quadratic fit j = a + b·x + c·x², Tukey-weighted. Returns null if
// there are too few points.
function robustQuad(xs, ys, ws) {
  if (xs.length < 6) return null;
  let w = ws.slice(), coef = null;
  for (let it = 0; it < 6; it++) {
    // weighted normal equations for [1, x, x²]
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], v = [0, 0, 0];
    for (let n = 0; n < xs.length; n++) {
      const b = [1, xs[n], xs[n] * xs[n]];
      for (let r = 0; r < 3; r++) { v[r] += w[n] * b[r] * ys[n]; for (let c = 0; c < 3; c++) M[r][c] += w[n] * b[r] * b[c]; }
    }
    coef = solve3(M, v);
    if (!coef) return null;
    const res = xs.map((x, n) => ys[n] - (coef[0] + coef[1] * x + coef[2] * x * x));
    const scale = Math.max(1.2, 1.4826 * quantile(res.map(Math.abs), 0.5)) * 3;
    w = ws.map((w0, n) => { const t = res[n] / scale; return Math.abs(t) < 1 ? w0 * (1 - t * t) ** 2 : 0; });
  }
  return coef;
}

function solve3(M, v) {
  const a = M.map((row, r) => [...row, v[r]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    if (Math.abs(a[p][c]) < 1e-9) return null;
    [a[c], a[p]] = [a[p], a[c]];
    for (let r = 0; r < 3; r++) {
      if (r === c) continue;
      const f = a[r][c] / a[c][c];
      for (let k = c; k < 4; k++) a[r][k] -= f * a[c][k];
    }
  }
  return [a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]];
}

// prior: function i -> j (the face model's lid line in patch coordinates)
// iris: { ci, cj, r } in patch coordinates (or null); refl: { i, j } or null
export function traceLid(P, which, prior, iStart, iEnd, iRef, iris, refl, pupil, maxJ) {
  const { W, H } = P;
  const Ys = blur(P.Y, W, H, 1.2);
  const upper = which === 'upper';
  const band = Math.round(0.85 * R); // ±5 mm around the model line
  const i0 = Math.max(4, Math.round(iStart)), i1 = Math.min(W - 5, Math.round(iEnd));
  const nI = i1 - i0 + 1;
  if (nI < R) return null;
  const nJ = 2 * band + 1;
  const pj = new Float32Array(nI);
  for (let n = 0; n < nI; n++) pj[n] = prior(i0 + n);

  // edge evidence: brightness step across a 3-px gap, signed by the margin type
  const E = new Float32Array(nI * nJ).fill(NaN);
  const pos = [];
  for (let n = 0; n < nI; n++) {
    const i = i0 + n;
    for (let t = 0; t < nJ; t++) {
      const j = Math.round(pj[n]) - band + t;
      if (j < 4 || j > H - 5) continue;
      if (maxJ && j > maxJ(i)) continue; // the upper margin must stay above the lower one
      const above = (Ys[(j - 1) * W + i] + Ys[(j - 2) * W + i] + Ys[(j - 3) * W + i]) / 3;
      const below = (Ys[j * W + i] + Ys[(j + 1) * W + i] + Ys[(j + 2) * W + i]) / 3;
      let e;
      if (upper) e = below - above; // lashes / lid (dark) above, globe below
      else {
        const inIris = iris && Math.hypot(i - iris.ci, j - iris.cj) < iris.r - 2;
        e = inIris ? below - above : above - below; // iris → lid skin, or sclera → lid
      }
      if (refl && Math.hypot(i - refl.i, j - refl.j) < 0.22 * R) e = 0; // ignore the light reflex
      // Inside the pupil there is no usable lid edge (lashes over a black pupil
      // give almost no contrast), and the pupil rim itself is a strong false edge.
      if (pupil && Math.hypot(i - pupil.ci, j - pupil.cj) < pupil.r + 3) e = 0;
      E[n * nJ + t] = e;
      if (e > 0) pos.push(e);
    }
  }
  const scale = Math.max(6, quantile(pos, 0.9));

  // dynamic programming: strongest smooth path, gently held near the model line
  const LAMBDA = [0, 0.06, 0.24]; // penalty for |Δj| = 0, 1, 2 per column
  const MU = 0.25 / R;           // pull toward the model line, per px
  const S = new Float32Array(nI * nJ), B = new Int8Array(nI * nJ);
  for (let n = 0; n < nI; n++) {
    for (let t = 0; t < nJ; t++) {
      const e = E[n * nJ + t];
      const local = Number.isNaN(e) ? -1 : Math.max(-1, Math.min(1.5, e / scale)) - MU * Math.abs(t - band);
      if (n === 0) { S[t] = local; continue; }
      let bestPrev = -Infinity, bd = 0;
      const shift = Math.round(pj[n]) - Math.round(pj[n - 1]); // keep rows aligned in image space
      for (let d = -2; d <= 2; d++) {
        const tp = t + shift + d;
        if (tp < 0 || tp >= nJ) continue;
        const v = S[(n - 1) * nJ + tp] - LAMBDA[Math.abs(d)];
        if (v > bestPrev) { bestPrev = v; bd = d; }
      }
      S[n * nJ + t] = local + (bestPrev === -Infinity ? -5 : bestPrev);
      B[n * nJ + t] = bd;
    }
  }
  let t = 0;
  for (let q = 1; q < nJ; q++) if (S[(nI - 1) * nJ + q] > S[(nI - 1) * nJ + t]) t = q;
  const path = new Float32Array(nI), strength = new Float32Array(nI);
  for (let n = nI - 1; n >= 0; n--) {
    const e0 = E[n * nJ + t];
    // sub-pixel: parabola through neighbouring edge values; margin lies between rows j-1 and j
    const em = E[n * nJ + Math.max(0, t - 1)], ep = E[n * nJ + Math.min(nJ - 1, t + 1)];
    let off = 0;
    const den = em - 2 * e0 + ep;
    if (Number.isFinite(den) && den < 0) off = Math.max(-0.5, Math.min(0.5, 0.5 * (em - ep) / den));
    path[n] = Math.round(pj[n]) - band + t + off - 0.5;
    strength[n] = Number.isNaN(e0) ? 0 : Math.max(0, e0 / scale);
    if (n > 0) {
      const shift = Math.round(pj[n]) - Math.round(pj[n - 1]);
      t = t + shift + B[n * nJ + t];
      t = Math.max(0, Math.min(nJ - 1, t));
    }
  }

  // robust quadratic over the central fissure, evaluated at the reflex column
  const win = 1.7 * R, xs = [], ys = [], ws = [];
  for (let n = 0; n < nI; n++) {
    const x = i0 + n - iRef;
    if (Math.abs(x) > win || strength[n] < 0.25) continue;
    xs.push(x / R); ys.push(path[n]); ws.push(Math.min(1.5, strength[n]));
  }
  const coef = robustQuad(xs, ys, ws);
  if (!coef) return null;
  const jRef = coef[0];
  // confidence: share of the window with a clear edge, and how well it fits
  let inl = 0, tot = 0, sumS = 0;
  for (let n = 0; n < nI; n++) {
    const x = (i0 + n - iRef) / R;
    if (Math.abs(x) > win / R) continue;
    if (pupil && Math.abs(i0 + n - pupil.ci) < pupil.r + 3 && Math.abs(path[n] - pupil.cj) < pupil.r + 4) continue;
    tot++;
    const fit = coef[0] + coef[1] * x + coef[2] * x * x;
    if (strength[n] >= 0.25 && Math.abs(path[n] - fit) < 2.5) { inl++; sumS += Math.min(1.5, strength[n]); }
  }
  const conf = tot ? (inl / tot) * Math.min(1, sumS / Math.max(1, inl)) : 0;
  const curve = [];
  for (let n = 0; n < nI; n += 3) curve.push({ i: i0 + n, j: path[n] });
  return { jRef, conf, coef, curve };
}

// ---------- per-eye analysis ----------

// opts: { F, center, r0, upperPoly, lowerPoly, med, lat, reflex } in image coords
export function analyzeEye(img, opts) {
  const { F, center, r0 } = opts;
  const P = rectify(img, F, center, r0);
  const refl = opts.reflex ? P.toPatch(opts.reflex) : null;
  // A clear pupil (light irises) anchors the limbus search; with dark irises
  // the pupil edge is too faint to trust, so the iris is fitted on its own.
  const pupil0 = fitPupil(P, { ci: P.ox, cj: P.oy, r: R }, Math.round(0.3 * R), refl);
  const pupilOk0 = pupil0 && pupil0.r >= 0.25 * R && pupil0.r <= 0.55 * R && pupil0.score >= 22;
  const iris = fitIris(P, pupilOk0 ? pupil0 : null);
  const irisOk = iris && iris.conf > 0.35 && Math.abs(iris.r / R - 1) < 0.36;
  const ir = irisOk ? iris : { ci: P.ox, cj: P.oy, r: R };
  const pupil = irisOk ? fitPupil(P, ir, 4, refl) : (pupilOk0 ? pupil0 : null);
  const iRef = refl ? refl.i : ir.ci;

  const lineOf = poly => {
    const pts = poly.map(p => P.toPatch(p)).sort((a, b) => a.i - b.i);
    return i => {
      if (i <= pts[0].i) return pts[0].j;
      for (let n = 0; n < pts.length - 1; n++) {
        if (i <= pts[n + 1].i) { const t = (i - pts[n].i) / ((pts[n + 1].i - pts[n].i) || 1); return pts[n].j + t * (pts[n + 1].j - pts[n].j); }
      }
      return pts[pts.length - 1].j;
    };
  };
  const m = P.toPatch(opts.med), l = P.toPatch(opts.lat);
  const iA = Math.min(m.i, l.i) + 0.35 * R, iB = Math.max(m.i, l.i) - 0.35 * R;
  const res = { patch: P };
  // Lower lid first: its edge over the iris (dark iris → lighter skin) would
  // otherwise read as an upper-lid edge in a narrow or ptotic fissure.
  const trL = traceLid(P, 'lower', lineOf(opts.lowerPoly), iA, iB, iRef, irisOk ? ir : null, refl, pupil);
  let maxJ = null;
  if (trL && trL.conf > 0.3) {
    const pts = trL.curve;
    const gap = 0.6 / 5.85 * R; // keep ≥ 0.6 mm between the margins
    maxJ = i => {
      let best = pts[0];
      for (const q of pts) if (Math.abs(q.i - i) < Math.abs(best.i - i)) best = q;
      return best.j - gap;
    };
    res.lower = { pt: P.toImg(iRef, trL.jRef), conf: trL.conf, curve: trL.curve.map(q => P.toImg(q.i, q.j)) };
  }
  // A visible reflex proves the lid margin is above it.
  const maxU = refl ? (i => Math.min(maxJ ? maxJ(i) : Infinity, Math.abs(i - refl.i) < 0.5 * R ? refl.j - 0.07 * R : Infinity)) : maxJ;
  const trU = traceLid(P, 'upper', lineOf(opts.upperPoly), iA, iB, iRef, irisOk ? ir : null, refl, pupil, maxU);
  if (trU && trU.conf > 0.3) {
    if (maxU) trU.jRef = Math.min(trU.jRef, maxU(iRef));
    res.upper = { pt: P.toImg(iRef, trU.jRef), conf: trU.conf, curve: trU.curve.map(q => P.toImg(q.i, q.j)) };
  }
  if (irisOk) res.iris = { c: P.toImg(iris.ci, iris.cj), r: iris.r / P.k, conf: iris.conf };
  return res;
}
