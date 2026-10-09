// Synthetic eye renderer with known ground truth, for benchmarking the lid and
// limbus detectors. Not a substitute for clinical validation: it checks that
// the geometry and the edge logic are right across lid heights, iris colours,
// skin tones, image scale, blur, noise and head roll.

export function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = rnd => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// smooth value noise
function noise2(seed) {
  const rnd = mulberry32(seed), P = 64, g = new Float32Array(P * P);
  for (let i = 0; i < g.length; i++) g[i] = rnd();
  const at = (x, y) => g[((y % P + P) % P) * P + ((x % P + P) % P)];
  return (x, y) => {
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    return (at(x0, y0) * (1 - sx) + at(x0 + 1, y0) * sx) * (1 - sy) + (at(x0, y0 + 1) * (1 - sx) + at(x0 + 1, y0 + 1) * sx) * sy;
  };
}

const IRIS = {
  brown: [92, 56, 34], dark: [52, 33, 24], hazel: [120, 92, 52], blue: [92, 120, 148],
};
const SKIN = [[205, 160, 132], [176, 126, 96], [150, 104, 78], [118, 80, 60], [92, 62, 48]];

export function renderEye(p) {
  const rnd = mulberry32(p.seed);
  const r = p.r;                  // iris radius, px
  const mm = r / 5.85;            // px per mm (HVID 11.7)
  const W = Math.round(r * 13), H = Math.round(r * 9);
  const cx = W / 2, cy = H / 2;   // iris centre (image)
  const th = (p.roll || 0) * Math.PI / 180, ct = Math.cos(th), st = Math.sin(th);
  // eye-frame coordinates (x lateral→, y down) of an image point, and back
  const toEye = (x, y) => ({ x: (x - cx) * ct + (y - cy) * st, y: -(x - cx) * st + (y - cy) * ct });
  const toImg = (ex, ey) => ({ x: cx + ex * ct - ey * st, y: cy + ex * st + ey * ct });

  const refl = { x: (rnd() - 0.5) * 0.16 * r, y: (rnd() - 0.5) * 0.16 * r };
  const xm = -2.35 * r, xl = 2.6 * r;                 // canthi
  const ym = 0.12 * r, yl = -0.22 * r;
  const base = x => ym + (yl - ym) * (x - xm) / (xl - xm);
  const bump = (x, pw, skew) => {
    const t = clamp((x - xm) / (xl - xm), 0, 1);
    const ts = t + skew * t * (1 - t);                // shift the apex
    return Math.pow(Math.sin(Math.PI * clamp(ts, 0, 1)), pw);
  };
  const pwU = 0.9 + rnd() * 0.5, skU = (rnd() - 0.6) * 0.5, pwL = 1.1 + rnd() * 0.4, skL = (rnd() - 0.5) * 0.4;
  const aU = (base(refl.x) - (refl.y - p.mrd1 * mm)) / bump(refl.x, pwU, skU);
  const aL = ((refl.y + p.mrd2 * mm) - base(refl.x)) / bump(refl.x, pwL, skL);
  const yU = x => base(x) - aU * bump(x, pwU, skU);
  const yL = x => base(x) + aL * bump(x, pwL, skL);

  const pr = r * (0.3 + rnd() * 0.15);
  const irisC = IRIS[p.iris || 'brown'];
  const skinC = SKIN[p.skin ?? 2];
  const nS = noise2(p.seed + 1), nI = noise2(p.seed + 2), nL = noise2(p.seed + 3);
  const creaseY = x => yU(x) - (6 + rnd() * 0) * mm - 0.4 * mm * bump(x, 1, 0);

  function shade(ex, ey) {
    const inFis = ex > xm && ex < xl && ey > yU(ex) && ey < yL(ex);
    if (inFis) {
      const d = Math.hypot(ex, ey);
      if (d < r) {
        if (d < pr) return [12, 10, 10];
        const a = Math.atan2(ey, ex);
        const streak = 0.75 + 0.35 * nI(a * 14, d / r * 6) + 0.12 * Math.sin(a * 37 + nI(d / 4, a) * 3);
        const ring = d > 0.86 * r ? 1 - 0.38 * (d - 0.86 * r) / (0.14 * r) : 1;
        const coll = Math.abs(d / r - 0.55) < 0.06 ? 1.15 : 1;
        const f = streak * ring * coll;
        let c = irisC.map(v => v * f);
        // shadow of the upper lid on the cornea
        const sh = clamp((ey - yU(ex)) / (0.18 * r), 0, 1);
        c = c.map(v => v * (0.7 + 0.3 * sh));
        return c;
      }
      // sclera, with a soft limbus transition
      const t = clamp((d - r) / (0.05 * r), 0, 1);
      let c = [228, 218, 206].map((v, i) => v * (0.9 + 0.1 * nS(ex / 9, ey / 9)));
      const edge = Math.min((ex - xm) / (0.6 * r), (xl - ex) / (0.6 * r), 1);
      c = c.map(v => v * (0.78 + 0.22 * clamp(edge, 0, 1)));
      if (ex < xm + 0.35 * r) c = [c[0], c[1] * 0.8, c[2] * 0.8]; // caruncle
      const sh = clamp((ey - yU(ex)) / (0.16 * r), 0, 1);
      c = c.map(v => v * (0.62 + 0.38 * sh));
      // tear meniscus: a bright thin line just above the lower margin
      if (yL(ex) - ey < 0.035 * r) c = c.map(v => Math.min(255, v * 1.12 + 10));
      if (t < 1) c = c.map((v, i) => irisC[i] * 0.62 * (1 - t) + v * t);
      return c;
    }
    // skin
    let c = skinC.map(v => v * (0.88 + 0.18 * nS(ex / 14, ey / 14)) * (0.93 + 0.07 * nL(ex / 3, ey / 3)));
    if (ex > xm - 0.2 * r && ex < xl + 0.2 * r) {
      const dU = yU(clamp(ex, xm, xl)) - ey, dL = ey - yL(clamp(ex, xm, xl));
      if (dU > 0 && dU < 0.11 * r) c = c.map((v, i) => v * (i === 0 ? 0.82 : 0.7));   // upper lid margin
      if (dL > 0 && dL < 0.07 * r) c = c.map((v, i) => v * (i === 0 ? 0.92 : 0.8));   // lower lid margin
      const dc = Math.abs(ey - creaseY(clamp(ex, xm, xl)));
      if (ex > xm && ex < xl && dc < 0.12 * r) c = c.map(v => v * (0.82 + 0.18 * dc / (0.12 * r)));
    }
    return c;
  }

  const data = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const acc = [0, 0, 0];
    for (const [a, b] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
      const e = toEye(x + a, y + b), c = shade(e.x, e.y);
      acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2];
    }
    const i = (y * W + x) * 3;
    data[i] = acc[0] / 4; data[i + 1] = acc[1] / 4; data[i + 2] = acc[2] / 4;
  }

  // lashes: dark strokes rooted on the margin
  const stroke = (x0, y0, x1, y1, wpx, col, alpha) => {
    const minx = Math.floor(Math.min(x0, x1) - wpx - 1), maxx = Math.ceil(Math.max(x0, x1) + wpx + 1);
    const miny = Math.floor(Math.min(y0, y1) - wpx - 1), maxy = Math.ceil(Math.max(y0, y1) + wpx + 1);
    const dx = x1 - x0, dy = y1 - y0, L2 = dx * dx + dy * dy || 1;
    for (let y = Math.max(0, miny); y <= Math.min(H - 1, maxy); y++) for (let x = Math.max(0, minx); x <= Math.min(W - 1, maxx); x++) {
      const t = clamp(((x + 0.5 - x0) * dx + (y + 0.5 - y0) * dy) / L2, 0, 1);
      const d = Math.hypot(x + 0.5 - (x0 + t * dx), y + 0.5 - (y0 + t * dy));
      const cov = clamp(wpx / 2 + 0.5 - d, 0, 1) * alpha * (1 - 0.5 * t);
      if (cov <= 0) continue;
      const i = (y * W + x) * 3;
      for (let k = 0; k < 3; k++) data[i + k] = data[i + k] * (1 - cov) + col[k] * cov;
    }
  };
  const lashW = Math.max(1, r / 32);
  const nUp = Math.round(70 * (p.lashes ?? 1));
  for (let n = 0; n < nUp; n++) {
    const ex = xm + 0.25 * r + rnd() * (xl - xm - 0.45 * r);
    const ey = yU(ex) - 0.03 * r;
    const len = (1.4 + rnd() * 1.6) * mm;
    const lat = (ex - (xm + xl) / 2) / (xl - xm);
    const ang = -Math.PI / 2 + lat * 1.1 + (rnd() - 0.5) * 0.5;
    const droop = rnd() < 0.12 ? 0.35 * mm : 0;      // a few lashes hang over the globe
    const a = toImg(ex, ey + droop), b = toImg(ex + Math.cos(ang) * len, ey + Math.sin(ang) * len);
    stroke(a.x, a.y, b.x, b.y, lashW * (0.8 + rnd() * 0.7), [28, 22, 20], 0.9);
  }
  for (let n = 0; n < 25 * (p.lashes ?? 1); n++) {
    const ex = xm + 0.6 * r + rnd() * (xl - xm - 1.0 * r);
    const ey = yL(ex) + 0.04 * r, len = (0.5 + rnd() * 0.8) * mm;
    const ang = Math.PI / 2 + (ex / (xl - xm)) * 0.8 + (rnd() - 0.5) * 0.4;
    const a = toImg(ex, ey), b = toImg(ex + Math.cos(ang) * len, ey + Math.sin(ang) * len);
    stroke(a.x, a.y, b.x, b.y, lashW * 0.7, [40, 30, 26], 0.7);
  }
  // corneal light reflex (hidden when the lid covers it)
  const rc = toImg(refl.x, refl.y), rr = Math.max(1.5, 0.07 * r);
  if (refl.y > yU(refl.x) + rr) for (let y = Math.floor(rc.y - 3 * rr); y <= rc.y + 3 * rr; y++) for (let x = Math.floor(rc.x - 3 * rr); x <= rc.x + 3 * rr; x++) {
    const d = Math.hypot(x + 0.5 - rc.x, y + 0.5 - rc.y), a = Math.exp(-(d * d) / (2 * (rr * 0.6) ** 2));
    const i = (y * W + x) * 3;
    for (let k = 0; k < 3; k++) data[i + k] = data[i + k] * (1 - a) + 255 * a;
  }

  // optics and sensor: blur + noise + a light gradient
  const sig = p.blur ?? 0.8;
  const out = blurRGB(data, W, H, sig);
  const nrnd = mulberry32(p.seed + 9), ns = p.noise ?? 4;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const g = 0.92 + 0.12 * x / W;
    const i = y * W + x;
    for (let k = 0; k < 3; k++) rgba[i * 4 + k] = out[i * 3 + k] * g + gauss(nrnd) * ns;
    rgba[i * 4 + 3] = 255;
  }

  // what the face model would report: right shape, imprecise position
  const mrnd = mulberry32(p.seed + 77);
  const modelErr = p.modelErr ?? 1;
  const biasU = clamp(gauss(mrnd) * 0.9, -1.8, 1.8) * mm * modelErr, biasL = clamp(gauss(mrnd) * 0.8, -1.6, 1.6) * mm * modelErr;
  const poly = (fy, bias) => Array.from({ length: 9 }, (_, n) => {
    const ex = xm + (xl - xm) * n / 8;
    return toImg(ex, fy(ex) + bias + gauss(mrnd) * 0.25 * mm * modelErr);
  });
  const model = {
    center: toImg(gauss(mrnd) * 0.1 * r * modelErr, gauss(mrnd) * 0.1 * r * modelErr),
    r0: r * (1 + clamp(gauss(mrnd) * 0.08, -0.18, 0.18) * modelErr),
    upperPoly: poly(yU, biasU), lowerPoly: poly(yL, biasL),
    med: toImg(xm, ym), lat: toImg(xl, yl),
  };
  const F = { u: { x: ct, y: st }, v: { x: -st, y: ct } };
  return {
    img: { data: rgba, w: W, h: H, x0: 0, y0: 0 }, F, mm,
    truth: {
      reflexVisible: refl.y > yU(refl.x) + rr,
      reflex: toImg(refl.x, refl.y), irisC: toImg(0, 0), r,
      upper: toImg(refl.x, yU(refl.x)), lower: toImg(refl.x, yL(refl.x)),
    },
    model,
  };
}

function blurRGB(src, W, H, sigma) {
  if (sigma <= 0.05) return src;
  const rad = Math.ceil(sigma * 2.5), ker = [];
  let s = 0;
  for (let t = -rad; t <= rad; t++) { const g = Math.exp(-t * t / (2 * sigma * sigma)); ker.push(g); s += g; }
  ker.forEach((v, i) => { ker[i] = v / s; });
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let k = 0; k < 3; k++) {
    let a = 0;
    for (let t = -rad; t <= rad; t++) a += src[(y * W + clamp(x + t, 0, W - 1)) * 3 + k] * ker[t + rad];
    tmp[(y * W + x) * 3 + k] = a;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let k = 0; k < 3; k++) {
    let a = 0;
    for (let t = -rad; t <= rad; t++) a += tmp[(clamp(y + t, 0, H - 1) * W + x) * 3 + k] * ker[t + rad];
    out[(y * W + x) * 3 + k] = a;
  }
  return out;
}

// Encode RGBA as a binary PPM for quick visual checks.
export function toPPM(img) {
  const head = Buffer.from(`P6\n${img.w} ${img.h}\n255\n`);
  const body = Buffer.alloc(img.w * img.h * 3);
  for (let i = 0; i < img.w * img.h; i++) for (let k = 0; k < 3; k++) body[i * 3 + k] = img.data[i * 4 + k];
  return Buffer.concat([head, body]);
}
