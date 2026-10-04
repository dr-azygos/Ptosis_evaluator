import {
  EYES, dist, mid, makeFrame, toUV, fromUV, polyAt, extractGeometry, Sampler,
  autoPrimary, autoGaze, measurePrimary, measureGaze, ptosisGrade, lfGrade, suggestions,
} from './analysis.js?v=11';
import {
  PARAMS, MIN_SAMPLES, learnedBias, applyLearned, recordCorrections, recordValidation, loadSamples,
  loadValidation, validationStats, validationCSV, blandAltmanSVG, resetLearning, resetValidation, describeLearned,
} from './learn.js?v=11';

const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const STAGES = ['primary', 'down', 'up'];
const AUTO_MS = 1200; // aligned this long → automatic photo
const STAGE_TEXT = {
  primary: 'Patient looks straight at the light, brows relaxed.',
  down: 'Patient looks fully DOWN. Fix the brow with your thumb.',
  up: 'Patient looks fully UP. Keep the brow fixed — no frontalis.',
};
const COLORS = {
  reflex: '#ffe14d', upper: '#35d0ff', lower: '#7dff6b', crease: '#ff5ce1',
  brow: '#ff9f40', med: '#3ddc84', lat: '#3ddc84', ruler: '#ff6b6b', limbN: '#c9d6e3', limbT: '#c9d6e3',
};
const LETTER = { reflex: 'R', upper: 'U', lower: 'L', crease: 'C', brow: 'B', med: 'M', lat: 'T', a: '◆', b: '◆', limbN: 'I', limbT: 'I' };
const HANDLE_NAME = {
  reflex: 'light reflex', upper: 'upper lid margin', lower: 'lower lid margin', crease: 'lid crease',
  brow: 'brow', med: 'medial canthus', lat: 'lateral canthus', a: 'ruler end', b: 'ruler end', limbN: 'nasal limbus', limbT: 'temporal limbus',
};

const $ = id => document.getElementById(id);
const el = {
  video: $('video'), liveCanvas: $('liveCanvas'), editCanvas: $('editCanvas'),
  checks: $('checks'), loading: $('loading'), instruction: $('instruction'), liveTable: $('liveTable'),
  results: $('resultsTable'), interp: $('interp'), calibInfo: $('calibInfo'),
};

// ---------------- state ----------------
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
const S = {
  settings: Object.assign({ hvid: 11.7, facing: 'environment', calib: 'hvid', rulerMm: 10, normalMrd1: 4.5, learn: true }, store.get('ptosis.settings', {})),
  caps: { primary: null, down: null, up: null },
  stage: 'primary',
  lm: null, lmMode: null, lmLoading: null, lastTs: 0,
  stream: null, track: null, torch: false, ring: false, zoom: true, auto: true,
  live: { running: false, cap: null, geom: null, view: null, lastTime: -1, ok: false, cfg: { q: 0, pad: 1 }, miss: 0, src: null, rc: null, W: 0, H: 0 },
  rv: { which: 'primary', view: 'both', drag: null, sel: null, T: null, pending: false, zoom: 1, pan: { x: 0, y: 0 }, pointers: new Map(), pinch: null, panning: null },
  clinical: { phenylephrine: '', bells: '', jawwink: false, fatigue: false, notes: '' },
};
const sampler = new Sampler();
const saveSettings = () => store.set('ptosis.settings', S.settings);

function show(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === id));
  window.scrollTo(0, 0);
}

// ---------------- face landmarker ----------------
async function ensureLandmarker() {
  if (S.lm) return S.lm;
  if (!S.lmLoading) {
    S.lmLoading = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      const opts = d => ({
        baseOptions: { modelAssetPath: MODEL_URL, delegate: d },
        runningMode: 'VIDEO', numFaces: 1,
        minFaceDetectionConfidence: 0.3, minFacePresenceConfidence: 0.3, minTrackingConfidence: 0.3,
      });
      let lm;
      try { lm = await FaceLandmarker.createFromOptions(fileset, opts('GPU')); }
      catch { lm = await FaceLandmarker.createFromOptions(fileset, opts('CPU')); }
      S.lm = lm; S.lmMode = 'VIDEO';
      return lm;
    })();
    S.lmLoading.catch(() => { S.lmLoading = null; });
  }
  return S.lmLoading;
}
async function setMode(mode) {
  await ensureLandmarker();
  if (S.lmMode !== mode) { await S.lm.setOptions({ runningMode: mode }); S.lmMode = mode; }
}
const nextTs = () => (S.lastTs = Math.max(performance.now(), S.lastTs + 1));

// ---------------- camera ----------------
async function startCamera() {
  stopCamera();
  const facing = S.settings.facing;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  S.stream = stream;
  S.track = stream.getVideoTracks()[0];
  el.video.srcObject = stream;
  await el.video.play();
  let caps = {};
  try { caps = S.track.getCapabilities ? S.track.getCapabilities() : {}; } catch { /* unsupported */ }
  const torchOk = !!caps.torch;
  S.zoomCaps = caps.zoom && caps.zoom.max > caps.zoom.min ? caps.zoom : null;
  S.live.zf = 1;
  $('btnTorch').disabled = !torchOk;
  $('btnTorch').title = torchOk ? '' : 'Torch not available on this camera/browser — use Ring light or a pen-torch beside the lens';
  S.torch = false;
  if (torchOk && facing === 'environment') await setTorch(true);
  setRing(facing === 'user' && !torchOk);
  updateToolButtons();
}
function stopCamera() {
  if (S.stream) S.stream.getTracks().forEach(t => t.stop());
  S.stream = null; S.track = null;
}
async function setTorch(on) {
  if (!S.track) return;
  try { await S.track.applyConstraints({ advanced: [{ torch: on }] }); S.torch = on; }
  catch { S.torch = false; }
  updateToolButtons();
}
function setRing(on) { S.ring = on; document.body.classList.toggle('ringlight', on); updateToolButtons(); }
function updateToolButtons() {
  $('btnTorch').setAttribute('aria-pressed', S.torch);
  $('btnRing').setAttribute('aria-pressed', S.ring);
  $('btnAuto').setAttribute('aria-pressed', S.auto);
}

// ---------------- view transforms & drawing ----------------
function resizeCanvas(c) {
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return dpr;
}
function bbox(pts, padX, padY = padX) {
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = (x1 - x0) * padX, py = (y1 - y0) * padY;
  return { x: x0 - px, y: y0 - py, w: x1 - x0 + 2 * px, h: y1 - y0 + 2 * py };
}
function makeT(box, cw, ch, mirror) {
  let { x, y, w, h } = box;
  const a = cw / ch;
  if (w / h < a) { const nw = h * a; x -= (nw - w) / 2; w = nw; } else { const nh = w / a; y -= (nh - h) / 2; h = nh; }
  const k = cw / w;
  return {
    sx: x, sy: y, sw: w, sh: h, k, cw, ch, mirror,
    toS(p) { const X = (p.x - x) * k; return { x: mirror ? cw - X : X, y: (p.y - y) * k }; },
    toI(q) { const X = mirror ? cw - q.x : q.x; return { x: X / k + x, y: q.y / k + y }; },
  };
}
function drawImageView(ctx, src, W, H, T) {
  const ix0 = Math.max(0, T.sx), iy0 = Math.max(0, T.sy);
  const ix1 = Math.min(W, T.sx + T.sw), iy1 = Math.min(H, T.sy + T.sh);
  if (ix1 <= ix0 || iy1 <= iy0) return;
  ctx.save();
  if (T.mirror) { ctx.translate(T.cw, 0); ctx.scale(-1, 1); }
  ctx.drawImage(src, ix0, iy0, ix1 - ix0, iy1 - iy0, (ix0 - T.sx) * T.k, (iy0 - T.sy) * T.k, (ix1 - ix0) * T.k, (iy1 - iy0) * T.k);
  ctx.restore();
}
function capPoints(cap, eyes = EYES) {
  const pts = [];
  for (const s of eyes) {
    pts.push(...Object.values(cap.eyes[s].h));
    if (cap.eyes[s].irisC) {
      const c = cap.eyes[s].irisC, r = cap.eyes[s].irisRpx;
      pts.push({ x: c.x - r, y: c.y - r }, { x: c.x + r, y: c.y + r });
    }
  }
  return pts;
}

function line(ctx, a, b, color, w, dash) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = w; if (dash) ctx.setLineDash(dash);
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  ctx.restore();
}
function label(ctx, text, p, color, u, align) {
  ctx.save();
  ctx.font = `600 ${Math.round(12 * u)}px system-ui, sans-serif`;
  ctx.textAlign = align; ctx.textBaseline = 'middle';
  const w = ctx.measureText(text).width, h = 16 * u, pad = 4 * u;
  const x0 = align === 'left' ? p.x : align === 'right' ? p.x - w : p.x - w / 2;
  ctx.fillStyle = 'rgba(10,14,20,.72)';
  ctx.fillRect(x0 - pad, p.y - h / 2, w + 2 * pad, h);
  ctx.fillStyle = color; ctx.fillText(text, p.x, p.y);
  ctx.restore();
}
const f1 = v => (v == null || Number.isNaN(v) ? '—' : (Math.abs(v) < 0.05 ? 0 : v).toFixed(1));

// Overlay for a primary-gaze capture (live or review).
function drawPrimaryOverlay(ctx, T, cap, m, u) {
  const F = m.F;
  for (const s of EYES) {
    const E = cap.eyes[s], h = E.h, me = m.eyes[s];
    const limb = h.limbN ? { c: mid(h.limbN, h.limbT), r: dist(h.limbN, h.limbT) / 2 } : null;
    const R = toUV(F, h.reflex), irisR = limb ? limb.r : E.irisRpx;
    const at = v => T.toS(fromUV(F, R.u, v));
    const sR = T.toS(h.reflex);
    const latSide = Math.sign(toUV(F, h.lat).u - R.u) || 1;
    const side = v => T.toS(fromUV(F, R.u + latSide * irisR * 1.25, v));
    const align = side(R.v).x > sR.x ? 'left' : 'right';
    const tick = (v, color, half = irisR, w = 2) => line(ctx, T.toS(fromUV(F, R.u - half, v)), T.toS(fromUV(F, R.u + half, v)), color, w * u);

    if (limb) line(ctx, T.toS(h.limbN), T.toS(h.limbT), 'rgba(201,214,227,.8)', 1.5 * u);
    const irisCentre = limb ? limb.c : E.irisC;
    if (irisCentre) {
      ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = u;
      const c = T.toS(irisCentre); ctx.beginPath(); ctx.arc(c.x, c.y, irisR * T.k, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    }
    // canthal span (PFW)
    line(ctx, T.toS(h.med), T.toS(h.lat), 'rgba(61,220,132,.7)', 1.5 * u, [4 * u, 4 * u]);
    // reflex horizontal reference
    tick(R.v, 'rgba(255,225,77,.8)', irisR * 1.3, 1);
    const U = toUV(F, h.upper).v, L = toUV(F, h.lower).v, C = toUV(F, h.crease).v, B = toUV(F, h.brow).v;
    line(ctx, at(R.v), at(U), COLORS.upper, 2.5 * u);
    line(ctx, at(R.v), at(L), COLORS.lower, 2.5 * u);
    tick(U, COLORS.upper, irisR * 0.6);
    tick(L, COLORS.lower, irisR * 0.6);
    line(ctx, at(U), at(C), 'rgba(255,92,225,.7)', 1.5 * u, [3 * u, 3 * u]);
    tick(C, COLORS.crease, irisR * 0.8);
    tick(B, COLORS.brow, irisR * 0.8);
    // reflex dot
    ctx.save(); ctx.fillStyle = E.flags.reflex ? COLORS.reflex : 'rgba(255,225,77,.4)';
    ctx.beginPath(); ctx.arc(sR.x, sR.y, 3 * u, 0, Math.PI * 2); ctx.fill(); ctx.restore();

    const labels = [
      [`Brow ${f1(me.brow)}`, side(B), COLORS.brow],
      [`MCD ${f1(me.mcd)}`, side((U + C) / 2), COLORS.crease],
      [`MRD1 ${f1(me.mrd1)}`, side((R.v + U) / 2), COLORS.upper],
      [`MRD2 ${f1(me.mrd2)}`, side((R.v + L) / 2), COLORS.lower],
    ];
    // keep the stack from overlapping when the eye is small on screen
    for (let i = 1; i < labels.length; i++) {
      const prev = labels[i - 1][1], cur = labels[i][1];
      if (cur.y < prev.y + 18 * u) labels[i][1] = { x: cur.x, y: prev.y + 18 * u };
    }
    for (const [t, p, col] of labels) label(ctx, t, p, col, u, align);
    const sl = at(L);
    label(ctx, s, { x: sl.x, y: sl.y + 22 * u }, '#fff', u, 'center');
  }
}

function drawGazeOverlay(ctx, T, cap, m, u) {
  const F = m.F;
  for (const s of EYES) {
    const h = cap.eyes[s].h;
    const M = toUV(F, h.med), L = toUV(F, h.lat), Up = toUV(F, h.upper);
    const t = (Up.u - M.u) / ((L.u - M.u) || 1);
    const foot = fromUV(F, Up.u, M.v + t * (L.v - M.v));
    line(ctx, T.toS(h.med), T.toS(h.lat), COLORS.med, 2 * u, [5 * u, 4 * u]);
    line(ctx, T.toS(foot), T.toS(h.upper), COLORS.upper, 2.5 * u);
    const sU = T.toS(h.upper);
    label(ctx, `${s} lid ${f1(m.eyes[s].lidHeight)} mm`, { x: sU.x, y: sU.y - 18 * u }, COLORS.upper, u, 'center');
  }
}

function drawRuler(ctx, T, cap, u, mm) {
  if (!cap.ruler) return;
  const a = T.toS(cap.ruler.a), b = T.toS(cap.ruler.b);
  line(ctx, a, b, COLORS.ruler, 2.5 * u);
  label(ctx, `${mm} mm`, { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 16 * u }, COLORS.ruler, u, 'center');
}

function drawHandles(ctx, T, cap, u, selected) {
  for (const H of handleList(cap)) {
    const p = T.toS(H.get());
    const color = COLORS[H.key] || '#fff';
    const r = 10 * u;
    ctx.save();
    ctx.fillStyle = color + '40'; ctx.strokeStyle = color; ctx.lineWidth = 2 * u;
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (selected && selected.eye === H.eye && selected.key === H.key) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * u; ctx.beginPath(); ctx.arc(p.x, p.y, r + 4 * u, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = '#fff'; ctx.font = `700 ${Math.round(10 * u)}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(LETTER[H.key], p.x, p.y + 0.5);
    ctx.restore();
  }
}

function measureCap(cap) {
  if (!cap) return null;
  if (cap.kind === 'primary') return measurePrimary(cap, S.settings);
  const icd = S.caps.primary ? measurePrimary(S.caps.primary, S.settings).icd : null;
  return measureGaze(cap, S.settings, icd);
}
// Live view: alignment guide only. Measurements are made on the captured photo.
function drawGuide(ctx, T, cap, g, u, ok) {
  const col = ok ? '#3ddc84' : '#ffb547';
  const pts = EYES.map(s => T.toS(cap.eyes[s].h.reflex || cap.eyes[s].irisC));
  line(ctx, pts[0], pts[1], col, 1.5 * u, [6 * u, 5 * u]);
  for (const s of EYES) {
    const E = cap.eyes[s], h = E.h;
    if (g) for (const poly of [g[s].upper, g[s].lower]) {
      ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1.2 * u; ctx.beginPath();
      poly.forEach((p, i) => { const q = T.toS(p); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); });
      ctx.stroke(); ctx.restore();
    }
    const c = T.toS(h.limbN ? mid(h.limbN, h.limbT) : E.irisC);
    const r = (h.limbN ? dist(h.limbN, h.limbT) / 2 : E.irisRpx) * T.k;
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2 * u;
    ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, Math.PI * 2); ctx.stroke();
    // corner brackets around the eye
    const b = r * 2.6;
    for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      ctx.beginPath();
      ctx.moveTo(c.x + sx * b, c.y + sy * b * 0.55 - sy * 10 * u);
      ctx.lineTo(c.x + sx * b, c.y + sy * b * 0.55);
      ctx.lineTo(c.x + sx * b - sx * 10 * u, c.y + sy * b * 0.55);
      ctx.stroke();
    }
    ctx.restore();
    if (S.stage === 'primary') {
      const R = T.toS(h.reflex);
      ctx.save();
      if (E.flags.reflex) { ctx.fillStyle = COLORS.reflex; ctx.beginPath(); ctx.arc(R.x, R.y, 3.5 * u, 0, Math.PI * 2); ctx.fill(); }
      else { ctx.strokeStyle = '#ff6b6b'; ctx.lineWidth = 1.5 * u; ctx.beginPath(); ctx.arc(R.x, R.y, 6 * u, 0, Math.PI * 2); ctx.stroke(); }
      ctx.restore();
    }
  }
  if (S.live.okSince && S.auto) {
    const p = Math.min(1, (performance.now() - S.live.okSince) / AUTO_MS);
    const m = mid(pts[0], pts[1]);
    ctx.save(); ctx.strokeStyle = '#3ddc84'; ctx.lineWidth = 4 * u;
    ctx.beginPath(); ctx.arc(m.x, m.y, 22 * u, -Math.PI / 2, -Math.PI / 2 + p * Math.PI * 2); ctx.stroke(); ctx.restore();
  }
}

function drawOverlay(ctx, T, cap, m, u) {
  if (cap.kind === 'primary') { drawPrimaryOverlay(ctx, T, cap, m, u); if (S.settings.calib === 'ruler') drawRuler(ctx, T, cap, u, S.settings.rulerMm); }
  else drawGazeOverlay(ctx, T, cap, m, u);
}

// ---------------- live ----------------
function lerpPt(a, b, t) { return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }; }
function smoothCap(prev, cur, t) {
  if (!prev || prev.kind !== cur.kind) return cur;
  for (const s of EYES) {
    for (const k of Object.keys(cur.eyes[s].h)) cur.eyes[s].h[k] = lerpPt(prev.eyes[s].h[k], cur.eyes[s].h[k], t);
    cur.eyes[s].irisC = lerpPt(prev.eyes[s].irisC, cur.eyes[s].irisC, t);
    cur.eyes[s].irisRpx += (prev.eyes[s].irisRpx - cur.eyes[s].irisRpx) * (1 - t);
  }
  cur.irisDiamPx += (prev.irisDiamPx - cur.irisDiamPx) * (1 - t);
  return cur;
}
function buildCap(g, kind, smp, opts = {}) {
  const cap = kind === 'primary' ? autoPrimary(g, smp, { hvid: S.settings.hvid, ...opts }) : autoGaze(g, kind);
  for (const s of EYES) cap.eyes[s].irisC = { ...g[s].iris.c };
  return cap;
}

// Draw `src` rotated clockwise by cfg.q quarter-turns, optionally centred on a
// larger grey canvas (cfg.pad > 1). Padding lets the detector find a face that
// fills or overflows the frame, as in a close-up of the eyes.
function prepare(src, W, H, cfg, out) {
  const c = out || document.createElement('canvas');
  const sw = cfg.q % 2 === 1, pad = cfg.pad || 1;
  const rw = sw ? H : W, rh = sw ? W : H;
  const w = Math.round(rw * pad), h = Math.round(rh * pad);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const ctx = c.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (pad > 1) { ctx.fillStyle = '#808080'; ctx.fillRect(0, 0, w, h); }
  ctx.translate(w / 2, h / 2);
  ctx.rotate(cfg.q * Math.PI / 2);
  ctx.drawImage(src, -W / 2, -H / 2, W, H);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return c;
}
const isPlain = cfg => !cfg.q && (cfg.pad || 1) === 1;
// Orientations / framings tried, in order, when no face is found.
const SEARCH = [
  { q: 0, pad: 1 }, { q: 0, pad: 1.8 }, { q: 1, pad: 1 }, { q: 3, pad: 1 }, { q: 2, pad: 1 },
  { q: 1, pad: 1.8 }, { q: 3, pad: 1.8 }, { q: 2, pad: 1.8 },
];

// Quarter-turns (clockwise) needed to make the detected face upright.
function uprightTurns(g) {
  const m = mid(g.OD.iris.c, g.OS.iris.c);
  const deg = Math.atan2(g.nose.x - m.x, g.nose.y - m.y) * 180 / Math.PI; // 0 = face down is image down
  return ((Math.round(deg / 90) % 4) + 4) % 4;
}

function liveLoop() {
  if (!S.live.running) return;
  requestAnimationFrame(liveLoop);
  const v = el.video;
  if (v.readyState < 2 || !S.lm || S.lmMode !== 'VIDEO' || !v.videoWidth) return;
  if (v.currentTime === S.live.lastTime) { drawLive(); return; }
  S.live.lastTime = v.currentTime;
  const vw = v.videoWidth, vh = v.videoHeight;
  const run = cfg => {
    const src = isPlain(cfg) ? v : prepare(v, vw, vh, cfg, S.live.rc || (S.live.rc = document.createElement('canvas')));
    const W = isPlain(cfg) ? vw : src.width, H = isPlain(cfg) ? vh : src.height;
    const res = S.lm.detectForVideo(src, nextTs());
    const lms = res.faceLandmarks && res.faceLandmarks[0];
    return lms ? { src, W, H, g: extractGeometry(lms, W, H) } : null;
  };
  let hit;
  try {
    hit = run(S.live.cfg);
    // While the face is lost, keep trying the usual framing every frame and,
    // every few frames, one alternative (close-up padding or a rotation).
    if (!hit && ++S.live.miss % 4 === 0) {
      const alts = SEARCH.filter(c => c.q !== S.live.cfg.q || c.pad !== S.live.cfg.pad);
      const alt = alts[(S.live.miss / 4) % alts.length];
      hit = run(alt);
      if (hit) S.live.cfg = alt;
    }
  } catch (e) {
    el.instruction.textContent = `⚠ Face model error: ${e.message || e}`;
    return;
  }
  if (hit) {
    const turn = uprightTurns(hit.g);
    if (turn) { S.live.cfg = { q: (S.live.cfg.q + turn) % 4, pad: S.live.cfg.pad }; S.live.cap = null; S.live.view = null; drawLive(); return; }
    if (S.live.miss > 0) el.instruction.textContent = STAGE_TEXT[S.stage];
    S.live.miss = 0;
    S.live.src = hit.src; S.live.W = hit.W; S.live.H = hit.H;
    sampler.setSource(hit.src, hit.W, hit.H);
    S.live.cap = smoothCap(S.live.cap, buildCap(hit.g, S.stage, sampler, { skipCrease: true }), 0.45);
    S.live.geom = hit.g;
  } else {
    S.live.cap = null; S.live.geom = null;
    if (S.live.miss === 60) el.instruction.textContent = 'No face found. Hold the phone a little further back so the forehead, both eyes and nose are in view, or tap Capture to place the markers by hand.';
  }
  updateLivePanel();
  drawLive();
}

function drawLive() {
  const c = el.liveCanvas, u = resizeCanvas(c), ctx = c.getContext('2d');
  const cap = S.live.cap;
  // Without a face, show the plain feed so the preview doesn't jump around
  // while other framings are tried.
  const src = cap ? S.live.src : el.video;
  const W = cap ? S.live.W : el.video.videoWidth, H = cap ? S.live.H : el.video.videoHeight;
  let box = { x: 0, y: 0, w: W, h: H };
  if (cap && S.zoom) box = bbox(capPoints(cap), 0.3, 0.45);
  const prev = S.live.view;
  if (prev && S.zoom && cap) box = { x: prev.x + (box.x - prev.x) * 0.15, y: prev.y + (box.y - prev.y) * 0.15, w: prev.w + (box.w - prev.w) * 0.15, h: prev.h + (box.h - prev.h) * 0.15 };
  S.live.view = box;
  const mirror = S.settings.facing === 'user';
  let T = makeT(box, c.width, c.height, mirror);
  const dz = S.zoomCaps ? 1 : S.live.zf || 1; // digital zoom when the camera has none
  if (dz > 1) {
    // Preview-only zoom about the eyes (the photo is always the full frame).
    const eyes = cap ? bbox(capPoints(cap), 0, 0) : null;
    const cx = eyes ? eyes.x + eyes.w / 2 : T.sx + T.sw / 2, cy = eyes ? eyes.y + eyes.h / 2 : T.sy + T.sh / 2;
    const w = T.sw / dz, h = T.sh / dz;
    T = makeT({ x: cx - w / 2, y: cy - h / 2, w, h }, c.width, c.height, mirror);
  }
  ctx.fillStyle = S.ring ? '#fff' : '#000'; ctx.fillRect(0, 0, c.width, c.height);
  drawImageView(ctx, src, W, H, T);
  if (cap) drawGuide(ctx, T, cap, S.live.geom, u, S.live.ok);
  if ((S.live.zf || 1) > 1.01) label(ctx, `${S.live.zf.toFixed(1)}×`, { x: c.width - 12 * u, y: c.height - 16 * u }, '#fff', u, 'right');
}

async function setLiveZoom(z) {
  const zc = S.zoomCaps;
  const max = zc ? zc.max / zc.min : 4;
  S.live.zf = Math.max(1, Math.min(max, z));
  $('btnLiveZoom').textContent = `${S.live.zf < 1.95 ? S.live.zf.toFixed(1).replace('.0', '') : Math.round(S.live.zf)}×`;
  if (zc && S.track) {
    try { await S.track.applyConstraints({ advanced: [{ zoom: zc.min * S.live.zf }] }); } catch { /* keep digital */ }
  }
}

function chip(text, cls) { return `<span class="chip ${cls}">${text}</span>`; }
function updateLivePanel() {
  const cap = S.live.cap, g = S.live.geom;
  const chips = [];
  let ok = false;
  if (!cap) chips.push(chip('No face', 'bad'));
  else {
    const d = cap.irisDiamPx;
    chips.push(d >= 40 ? chip('Distance ✓', 'ok') : d >= 25 ? chip('Move closer', 'warn') : chip('Too far', 'bad'));
    const F = makeFrame(cap.eyes.OD.irisC, cap.eyes.OS.irisC, cap.nose);
    const roll = Math.abs(F.rollDeg);
    chips.push(roll < 3 ? chip('Level ✓', 'ok') : chip(`Head tilt ${roll.toFixed(0)}°`, roll < 6 ? 'warn' : 'bad'));
    const ipd = dist(cap.eyes.OD.irisC, cap.eyes.OS.irisC);
    const yaw = (toUV(F, g.nose).u - toUV(F, mid(cap.eyes.OD.irisC, cap.eyes.OS.irisC)).u) / ipd;
    chips.push(Math.abs(yaw) < 0.05 ? chip('Facing ✓', 'ok') : chip('Turn face to camera', Math.abs(yaw) < 0.1 ? 'warn' : 'bad'));
    ok = d >= 25 && roll < 6 && Math.abs(yaw) < 0.1;
    if (S.stage === 'primary') {
      const n = EYES.filter(s => cap.eyes[s].flags.reflex).length;
      chips.push(n === 2 ? chip('Reflex ✓', 'ok') : chip(n ? 'Reflex: 1 eye' : 'No reflex — light on?', n ? 'warn' : 'bad'));
    }
  }
  el.checks.innerHTML = chips.join('');
  $('btnCapture').disabled = S.live.busy;
  $('btnCapture').style.borderColor = ok ? 'var(--ok)' : 'var(--accent)';
  S.live.ok = ok;

  // Auto-capture once everything has stayed aligned for a moment.
  const now = performance.now();
  if (!ok || S.live.busy) S.live.okSince = null;
  else if (!S.live.okSince) S.live.okSince = now;
  let msg;
  if (S.live.busy) msg = 'Taking photo — hold still…';
  else if (!cap) msg = 'Find the face: forehead, both eyes and nose in view.';
  else if (!ok) msg = 'Line up: eyes level, face straight to the camera' + (S.stage === 'primary' ? ', patient looking at the light.' : '.');
  else if (S.auto) msg = 'Hold still — taking the photo…';
  else msg = 'Aligned. Tap the shutter to take the photo.';
  el.liveTable.innerHTML = `<tbody><tr><td class="guide">${msg}</td></tr></tbody>`;
  if (S.auto && S.live.okSince && now - S.live.okSince >= AUTO_MS) captureLive();
}

async function goLive(stage) {
  if (stage) setStage(stage);
  show('live');
  el.loading.hidden = false;
  el.loading.textContent = 'Starting camera…';
  try {
    if (!window.isSecureContext) throw new Error('Camera needs HTTPS (or localhost).');
    if (!S.stream) await startCamera();
    el.loading.textContent = 'Loading face model…';
    await setMode('VIDEO');
    el.loading.hidden = true;
    if (!S.live.running) { S.live.running = true; S.live.cap = null; S.live.view = null; S.live.src = null; S.live.miss = 0; S.live.cfg = { q: 0, pad: 1 }; requestAnimationFrame(liveLoop); }
  } catch (e) {
    el.loading.textContent = `⚠ ${e.message || e}`;
  }
}
function stopLive() { S.live.running = false; stopCamera(); setRing(false); }

function setStage(stage) {
  S.stage = stage;
  S.live.cap = null;
  document.querySelectorAll('#stageSeg button').forEach(b => {
    b.classList.toggle('on', b.dataset.stage === stage);
    b.classList.toggle('has', !!S.caps[b.dataset.stage]);
  });
  el.instruction.textContent = STAGE_TEXT[stage];
}

// ---------------- capture ----------------
async function captureFromSource(src, W, H, kind, isVideo) {
  await setMode(isVideo ? 'VIDEO' : 'IMAGE');
  const detect = c => {
    const res = isVideo ? S.lm.detectForVideo(c, nextTs()) : S.lm.detect(c);
    return res.faceLandmarks && res.faceLandmarks[0];
  };
  const tries = isVideo ? [S.live.cfg, ...SEARCH] : SEARCH;
  let c, lms, cfg;
  for (const t of tries) {
    c = prepare(src, W, H, t);
    lms = detect(c);
    if (lms) { cfg = t; break; }
  }
  if (!lms) {
    const cap = manualCap(prepare(src, W, H, { q: 0, pad: 1 }), kind);
    if (kind === 'primary') initRuler(cap);
    return cap;
  }
  let g = extractGeometry(lms, c.width, c.height);
  const turn = uprightTurns(g);
  if (turn) {
    const c2 = prepare(src, W, H, { q: (cfg.q + turn) % 4, pad: cfg.pad });
    const l2 = detect(c2);
    if (l2) { c = c2; g = extractGeometry(l2, c.width, c.height); }
  }
  sampler.setSource(c, c.width, c.height);
  const cap = buildCap(g, kind, sampler, { refineLids: true });
  cap.canvas = c; cap.W = c.width; cap.H = c.height; cap.time = new Date().toISOString();
  if (kind === 'primary') {
    applyLearned(cap, S.settings.learn ? learnedBias() : null, measureForLearning);
    initRuler(cap);
  }
  return cap;
}

// No face detected (e.g. a tight close-up of the eyes): start with markers in
// default positions for the examiner to drag into place. Calibration then comes
// from the limbus markers (I), so the corneal diameter is still used.
function manualCap(c, kind) {
  const W = c.width, H = c.height, r = Math.min(W * 0.06, H * 0.12);
  const ctr = { OD: { x: W * 0.3, y: H * 0.5 }, OS: { x: W * 0.7, y: H * 0.5 } };
  const at = (p, dx, dy) => ({ x: p.x + dx * r, y: p.y + dy * r });
  const eyes = {};
  for (const s of EYES) {
    const p = ctr[s], nasal = s === 'OD' ? 1 : -1;
    const h = kind === 'primary'
      ? { reflex: { ...p }, upper: at(p, 0, -0.6), lower: at(p, 0, 0.9), crease: at(p, 0, -1.8), brow: at(p, 0, -3), med: at(p, 2.3 * nasal, 0.2), lat: at(p, -2.3 * nasal, 0), limbN: at(p, nasal, 0), limbT: at(p, -nasal, 0) }
      : { upper: at(p, 0, -0.6), med: at(p, 2.3 * nasal, 0.2), lat: at(p, -2.3 * nasal, 0) };
    eyes[s] = { h, flags: { reflex: false, crease: false }, irisRpx: r, irisC: { ...p } };
  }
  return {
    kind, eyes, irisDiamPx: 2 * r, manual: true, nose: { x: W / 2, y: H / 2 + 6 * r },
    canvas: c, W, H, time: new Date().toISOString(),
  };
}

// Learning uses the iris scale (not the ruler) so corrections are comparable.
const measureForLearning = cap => measurePrimary(cap, { ...S.settings, calib: 'hvid' });

function initRuler(cap) {
  const m = measurePrimary(cap, { ...S.settings, calib: 'hvid' });
  const F = m.F, b1 = toUV(F, cap.eyes.OD.h.brow), b2 = toUV(F, cap.eyes.OS.h.brow);
  const cu = (b1.u + b2.u) / 2, cv = Math.min(b1.v, b2.v) - 12 / m.mmpp;
  const half = S.settings.rulerMm / 2 / m.mmpp;
  cap.ruler = { a: fromUV(F, cu - half, cv), b: fromUV(F, cu + half, cv) };
}

const nextFrame = v => new Promise(res => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(() => res()) : setTimeout(res, 50)));

// Openness of the narrower eye (fissure height / iris radius): low on a blink.
function openness(g) {
  return Math.min(...EYES.map(s => {
    const F = makeFrame(g.OD.iris.c, g.OS.iris.c, g.nose);
    const u = toUV(F, g[s].iris.c).u;
    const up = polyAt(F, g[s].upper, u), lo = polyAt(F, g[s].lower, u);
    return up && lo ? (toUV(F, lo).v - toUV(F, up).v) / g[s].iris.r : 0;
  }));
}

// Take a short burst of frames and keep the one with the eyes most open, so a
// blink or half-blink is never measured as ptosis. Marking runs on that photo.
async function captureLive() {
  const v = el.video;
  if (!v.videoWidth || S.live.busy) return;
  S.live.busy = true; S.live.okSince = null;
  const vw = v.videoWidth, vh = v.videoHeight;
  const flash = document.createElement('div');
  flash.style.cssText = 'position:fixed;inset:0;background:#fff;opacity:.6;pointer-events:none;transition:opacity .25s';
  document.body.appendChild(flash);
  requestAnimationFrame(() => { flash.style.opacity = '0'; setTimeout(() => flash.remove(), 300); });
  try {
    let best = null, bestScore = -1, cur = document.createElement('canvas');
    const scratch = document.createElement('canvas');
    for (let i = 0; i < 6; i++) {
      await nextFrame(v);
      cur.width = vw; cur.height = vh;
      cur.getContext('2d').drawImage(v, 0, 0, vw, vh);
      const src = isPlain(S.live.cfg) ? cur : prepare(cur, vw, vh, S.live.cfg, scratch);
      const res = S.lm.detectForVideo(src, nextTs());
      const lms = res.faceLandmarks && res.faceLandmarks[0];
      const score = lms ? (S.stage === 'primary' ? openness(extractGeometry(lms, src.width, src.height)) : 1) : 0;
      if (score > bestScore) { bestScore = score; const t = best; best = cur; cur = t || document.createElement('canvas'); }
      if (S.stage !== 'primary' && lms) break;
    }
    S.live.running = false;
    el.loading.hidden = false; el.loading.textContent = 'Marking the photo…';
    await new Promise(r => setTimeout(r, 30));
    const cap = await captureFromSource(best, vw, vh, S.stage, true);
    S.caps[S.stage] = cap;
    S.rv.which = S.stage;
    stopCamera(); setRing(false);
    el.loading.hidden = true;
    openReview();
  } catch (e) {
    el.loading.hidden = true;
    el.instruction.textContent = `⚠ ${e.message}`;
    if (!S.live.running) { S.live.running = true; requestAnimationFrame(liveLoop); }
  } finally {
    S.live.busy = false;
  }
}

async function loadPhoto(file, kind, msgEl) {
  if (!file) return;
  msgEl.textContent = 'Analysing photo…';
  try {
    await ensureLandmarker();
    const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }).catch(() => createImageBitmap(file));
    const scale = Math.min(1, 2400 / Math.max(bmp.width, bmp.height));
    const W = Math.round(bmp.width * scale), H = Math.round(bmp.height * scale);
    S.caps[kind] = await captureFromSource(bmp, W, H, kind, false);
    msgEl.textContent = '';
    S.rv.which = kind;
    openReview();
  } catch (e) {
    msgEl.textContent = `⚠ ${e.message || e}`;
  }
}

// ---------------- review / editor ----------------
function handleList(cap) {
  const out = [];
  for (const s of EYES) {
    for (const key of Object.keys(cap.eyes[s].h)) {
      out.push({ eye: s, key, get: () => cap.eyes[s].h[key], set: p => { cap.eyes[s].h[key] = p; } });
    }
  }
  if (cap.kind === 'primary' && S.settings.calib === 'ruler' && cap.ruler) {
    for (const key of ['a', 'b']) out.push({ eye: 'ruler', key, get: () => cap.ruler[key], set: p => { cap.ruler[key] = p; } });
  }
  return out;
}

function reviewT() {
  const cap = S.caps[S.rv.which], c = el.editCanvas;
  let box;
  if (S.rv.view === 'full') box = { x: 0, y: 0, w: cap.W, h: cap.H };
  else if (S.rv.view === 'both') {
    const pts = capPoints(cap);
    if (cap.kind === 'primary' && S.settings.calib === 'ruler' && cap.ruler) pts.push(cap.ruler.a, cap.ruler.b);
    box = bbox(pts, 0.22, 0.15);
  } else box = bbox(capPoints(cap, [S.rv.view]), 0.45, 0.2);
  // User zoom/pan on top of the chosen view.
  const z = S.rv.zoom, cx = box.x + box.w / 2 + S.rv.pan.x, cy = box.y + box.h / 2 + S.rv.pan.y;
  box = { x: cx - box.w / z / 2, y: cy - box.h / z / 2, w: box.w / z, h: box.h / z };
  return makeT(box, c.width, c.height, false);
}

// Zoom so the image point under screen point `m` stays under it.
function zoomAt(z, m) {
  const T0 = reviewT(), p = T0.toI(m);
  S.rv.zoom = Math.max(1, Math.min(10, z));
  if (S.rv.zoom === 1) { S.rv.pan = { x: 0, y: 0 }; return; }
  const T1 = reviewT(), s = T1.toS(p);
  S.rv.pan = { x: S.rv.pan.x + (s.x - m.x) / T1.k, y: S.rv.pan.y + (s.y - m.y) / T1.k };
}
function resetZoom() { S.rv.zoom = 1; S.rv.pan = { x: 0, y: 0 }; }

// Size the editor to the region being shown so a phone screen isn't mostly forehead.
function fitEditorHeight(cap) {
  const wrap = $('editWrap'), w = wrap.clientWidth;
  let aspect;
  if (S.rv.view === 'full') aspect = cap.H / cap.W;
  else {
    const pts = capPoints(cap, S.rv.view === 'both' ? EYES : [S.rv.view]);
    const b = S.rv.view === 'both' ? bbox(pts, 0.22, 0.15) : bbox(pts, 0.45, 0.2);
    aspect = b.h / b.w;
  }
  const h = Math.max(220, Math.min(window.innerHeight * 0.5, w * aspect));
  wrap.style.height = `${Math.round(h)}px`;
}

function drawEditor() {
  S.rv.pending = false;
  const cap = S.caps[S.rv.which], c = el.editCanvas;
  $('editEmpty').hidden = !!cap;
  if (!cap) { $('nudge').hidden = true; return; }
  if (!S.rv.drag) fitEditorHeight(cap);
  const u = resizeCanvas(c), ctx = c.getContext('2d');
  if (!S.rv.drag) S.rv.T = reviewT();
  const T = S.rv.T, m = measureCap(cap);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, c.width, c.height);
  drawImageView(ctx, cap.canvas, cap.W, cap.H, T);
  drawOverlay(ctx, T, cap, m, u);
  drawHandles(ctx, T, cap, u, S.rv.sel);
  if (S.rv.drag) drawLoupe(ctx, cap, m, u);
  const sel = S.rv.sel;
  $('nudge').hidden = !sel;
  if (sel) $('nudgeLabel').textContent = `${sel.eye === 'ruler' ? 'Ruler' : sel.eye} ${HANDLE_NAME[sel.key]}`;
}

function drawLoupe(ctx, cap, m, u) {
  const H = S.rv.drag.h, p = H.get(), T = S.rv.T;
  const R = 70 * u, zoom = 3;
  const sp = T.toS(p);
  const cx = sp.x < T.cw / 2 ? T.cw - R - 10 * u : R + 10 * u, cy = R + 10 * u;
  const L = makeT({ x: p.x - R / (T.k * zoom), y: p.y - R / (T.k * zoom), w: 2 * R / (T.k * zoom), h: 2 * R / (T.k * zoom) }, 2 * R, 2 * R, false);
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.clip();
  ctx.translate(cx - R, cy - R);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 2 * R, 2 * R);
  drawImageView(ctx, cap.canvas, cap.W, cap.H, L);
  drawOverlay(ctx, L, cap, m, u * 0.8);
  line(ctx, { x: R - 12 * u, y: R }, { x: R + 12 * u, y: R }, '#fff', u);
  line(ctx, { x: R, y: R - 12 * u }, { x: R, y: R + 12 * u }, '#fff', u);
  ctx.restore();
  ctx.save(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * u; ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
}

function scheduleDraw() {
  if (S.rv.pending) return;
  S.rv.pending = true;
  requestAnimationFrame(() => { drawEditor(); renderResults(); });
}

function canvasPt(e) {
  const r = el.editCanvas.getBoundingClientRect(), dpr = el.editCanvas.width / r.width;
  return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
}
// iOS Safari turns two-finger pinches into page zoom and cancels the pointer
// stream; it exposes the pinch through its own gesture events instead. Use
// those where they exist, and stop the page from zooming over the canvases.
const HAS_GESTURE = typeof window.GestureEvent !== 'undefined';
function pinchable(canvas, onScale) {
  canvas.addEventListener('touchmove', e => { if (e.touches.length > 1) e.preventDefault(); }, { passive: false });
  if (!HAS_GESTURE) return;
  let start = null;
  canvas.addEventListener('gesturestart', e => { e.preventDefault(); start = onScale.begin(e); }, { passive: false });
  canvas.addEventListener('gesturechange', e => { e.preventDefault(); if (start) onScale.change(start, e); }, { passive: false });
  canvas.addEventListener('gestureend', e => { e.preventDefault(); start = null; }, { passive: false });
}

function setupEditor() {
  const c = el.editCanvas;
  pinchable(c, {
    begin: () => { S.rv.drag = null; S.rv.panning = null; return { z0: S.rv.zoom }; },
    change: (st, e) => { zoomAt(st.z0 * e.scale, canvasPt(e)); scheduleDraw(); },
  });
  const P = S.rv.pointers;
  const pinchState = () => {
    const [a, b] = [...P.values()];
    return { d: dist(a, b), m: mid(a, b) };
  };
  c.addEventListener('pointerdown', e => {
    const cap = S.caps[S.rv.which];
    if (!cap || !S.rv.T) return;
    const q = canvasPt(e);
    P.set(e.pointerId, q);
    try { c.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
    e.preventDefault();
    if (P.size === 2) {
      // Second finger: switch from dragging/panning to pinch-zoom.
      S.rv.drag = null; S.rv.panning = null;
      const st = pinchState();
      S.rv.pinch = { d0: st.d, z0: S.rv.zoom, m: st.m };
      scheduleDraw();
      return;
    }
    if (P.size > 2) return;
    const u = c.width / c.getBoundingClientRect().width;
    let best = null, bd = 28 * u;
    for (const H of handleList(cap)) {
      const d = dist(S.rv.T.toS(H.get()), q);
      if (d < bd) { bd = d; best = H; }
    }
    if (best) {
      const s = S.rv.T.toS(best.get());
      S.rv.drag = { h: best, off: { x: s.x - q.x, y: s.y - q.y } };
      S.rv.sel = { eye: best.eye, key: best.key };
    } else {
      S.rv.sel = null;
      S.rv.panning = { last: q };
    }
    scheduleDraw();
  });
  c.addEventListener('pointermove', e => {
    if (!P.has(e.pointerId)) return;
    const q = canvasPt(e);
    P.set(e.pointerId, q);
    if (S.rv.pinch && P.size === 2 && HAS_GESTURE) {
      // Safari: zoom comes from gesture events; just pan with the midpoint.
      const st = pinchState(), pz = S.rv.pinch, T = reviewT();
      S.rv.pan = { x: S.rv.pan.x - (st.m.x - pz.m.x) / T.k, y: S.rv.pan.y - (st.m.y - pz.m.y) / T.k };
      pz.m = st.m;
      scheduleDraw();
    } else if (S.rv.pinch && P.size === 2) {
      const st = pinchState(), pz = S.rv.pinch;
      // pan with the midpoint, then zoom about it
      const T = reviewT();
      S.rv.pan = { x: S.rv.pan.x - (st.m.x - pz.m.x) / T.k, y: S.rv.pan.y - (st.m.y - pz.m.y) / T.k };
      pz.m = st.m;
      zoomAt(pz.z0 * st.d / pz.d0, st.m);
      scheduleDraw();
    } else if (S.rv.drag) {
      const o = S.rv.drag.off;
      S.rv.drag.h.set(S.rv.T.toI({ x: q.x + o.x, y: q.y + o.y }));
      scheduleDraw();
    } else if (S.rv.panning && S.rv.zoom > 1) {
      const T = reviewT(), l = S.rv.panning.last;
      S.rv.pan = { x: S.rv.pan.x - (q.x - l.x) / T.k, y: S.rv.pan.y - (q.y - l.y) / T.k };
      S.rv.panning.last = q;
      scheduleDraw();
    }
  });
  const end = e => {
    P.delete(e.pointerId);
    if (P.size < 2) S.rv.pinch = null;
    if (P.size === 0) { S.rv.panning = null; if (S.rv.drag) S.rv.drag = null; }
    scheduleDraw();
  };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
  c.addEventListener('wheel', e => {
    if (!S.caps[S.rv.which]) return;
    e.preventDefault();
    zoomAt(S.rv.zoom * Math.exp(-e.deltaY * 0.002), canvasPt(e));
    scheduleDraw();
  }, { passive: false });
  $('zoombar').addEventListener('click', e => {
    const z = e.target.dataset.z;
    if (!z || !S.caps[S.rv.which]) return;
    const m = { x: c.width / 2, y: c.height / 2 };
    if (z === 'in') zoomAt(S.rv.zoom * 1.5, m);
    else if (z === 'out') zoomAt(S.rv.zoom / 1.5, m);
    else resetZoom();
    scheduleDraw();
  });
  $('nudge').addEventListener('click', e => {
    const d = e.target.dataset.d;
    const cap = S.caps[S.rv.which];
    if (!d || !S.rv.sel || !cap) return;
    const [dx, dy] = d.split(',').map(Number);
    const H = handleList(cap).find(h => h.eye === S.rv.sel.eye && h.key === S.rv.sel.key);
    if (!H) return;
    // one step ≈ 0.1 mm, or 1 px at most
    const mmpp = cap.kind === 'primary' ? measurePrimary(cap, S.settings).mmpp : measureCap(cap).mmpp;
    const step = Math.min(1, 0.1 / mmpp);
    const p = H.get();
    H.set({ x: p.x + dx * step, y: p.y + dy * step });
    scheduleDraw();
  });
  new ResizeObserver(() => { if ($('review').classList.contains('active')) scheduleDraw(); }).observe($('editWrap'));
}

function openReview() {
  resetZoom();
  show('review');
  syncReviewInputs();
  updateSegs();
  scheduleDraw();
}
function updateSegs() {
  document.querySelectorAll('#capSeg button').forEach(b => {
    b.classList.toggle('on', b.dataset.cap === S.rv.which);
    b.classList.toggle('has', !!S.caps[b.dataset.cap]);
  });
  document.querySelectorAll('#viewSeg button').forEach(b => b.classList.toggle('on', b.dataset.v === S.rv.view));
}

// ---------------- results & report ----------------
function computeAll() {
  const mp = S.caps.primary ? measurePrimary(S.caps.primary, S.settings) : null;
  const md = S.caps.down ? measureCap(S.caps.down) : null;
  const mu = S.caps.up ? measureCap(S.caps.up) : null;
  const lf = md && mu ? { OD: mu.eyes.OD.lidHeight - md.eyes.OD.lidHeight, OS: mu.eyes.OS.lidHeight - md.eyes.OS.lidHeight } : null;
  return { mp, md, mu, lf };
}

function resultRows(R) {
  const { mp, md, mu, lf } = R, n = S.settings.normalMrd1;
  const e = (s, k) => (mp ? mp.eyes[s][k] : null);
  const rows = [
    ['MRD1', 'mrd1', 'Reflex → upper lid margin'],
    ['MRD2', 'mrd2', 'Reflex → lower lid margin'],
    ['Palpebral fissure height', 'pfh', 'MRD1 + MRD2'],
    ['Palpebral fissure width', 'pfw', 'Medial → lateral canthus'],
    ['Lid crease height (MCD)', 'mcd', 'Lid margin → crease'],
    ['Reflex → brow', 'brow', 'To lower brow margin'],
    ['Corneal coverage (upper lid)', 'coverage', 'Iris radius − MRD1'],
    ['Inferior scleral show', 'scleralShowInf', 'MRD2 − iris radius'],
  ].map(([name, k, sub]) => ({ name, sub, OD: e('OD', k), OS: e('OS', k), key: k }));
  if (mp) {
    const ruler = S.settings.calib === 'ruler' && S.caps.primary.ruler;
    rows.push({ name: 'Corneal diameter (HVID)', sub: ruler ? 'Limbus to limbus, measured with ruler' : `Limbus to limbus; mean set to ${S.settings.hvid} mm`, OD: e('OD', 'hvid'), OS: e('OS', 'hvid'), key: 'hvid' });
  }
  rows.push({ name: 'Lid height, down-gaze', sub: 'Above intercanthal line', OD: md && md.eyes.OD.lidHeight, OS: md && md.eyes.OS.lidHeight });
  rows.push({ name: 'Lid height, up-gaze', sub: 'Above intercanthal line', OD: mu && mu.eyes.OD.lidHeight, OS: mu && mu.eyes.OS.lidHeight });
  rows.push({ name: 'Levator function', sub: 'Up − down excursion', OD: lf && lf.OD, OS: lf && lf.OS, key: 'lf' });
  const grade = s => (mp ? ptosisGrade(mp.eyes[s].mrd1, n, mp.eyes[s === 'OD' ? 'OS' : 'OD'].mrd1) : null);
  return { rows, grade };
}

function renderResults() {
  const R = computeAll(), { mp, lf } = R;
  const { rows, grade } = resultRows(R);
  const cell = (r, s) => {
    let v = f1(r[s]);
    if (r.key === 'mrd1' && mp && !mp.eyes[s].reflexFound) v += '<small class="flag">reflex est.</small>';
    if (r.key === 'mcd' && mp && !mp.eyes[s].creaseAuto) v += '<small class="flag">check crease</small>';
    if (r.key === 'lf' && lf) v += `<small>${lfGrade(lf[s])}</small>`;
    if (r.key === 'hvid' && mp && !mp.eyes[s].limbusAuto) v += '<small class="flag">check I markers</small>';
    return `<td>${v}</td>`;
  };
  let html = '<thead><tr><th>mm</th><th>OD (R)</th><th>OS (L)</th></tr></thead><tbody>';
  for (const r of rows) html += `<tr><td>${r.name}<small>${r.sub}</small></td>${cell(r, 'OD')}${cell(r, 'OS')}</tr>`;
  const gc = s => { const g = grade(s); return g ? `<td class="sev${g.level}">${g.label}<small>${g.level ? `≈ ${g.amount.toFixed(1)} mm` : ''}</small></td>` : '<td>—</td>'; };
  html += `<tr><td>Grade<small>vs normal MRD1 ${S.settings.normalMrd1} mm</small></td>${gc('OD')}${gc('OS')}</tr>`;
  if (mp) html += `<tr><td>MRD1 asymmetry</td><td colspan="2">${Math.abs(mp.eyes.OD.mrd1 - mp.eyes.OS.mrd1).toFixed(1)} mm</td></tr>`;
  el.results.innerHTML = html + '</tbody>';

  const L = S.caps.primary && S.caps.primary.learned;
  const learnedTxt = L ? ` Learned correction applied: ${describeLearned(L)}.` : '';
  el.calibInfo.textContent = (mp
    ? `Scale: ${mp.method} → ${(mp.mmpp * 1000).toFixed(1)} µm/px. IPD ${f1(mp.ipd)} mm, intercanthal ${f1(mp.icd)} mm.${lf ? ' Gaze captures scaled by intercanthal distance.' : ''}`
    : 'Capture primary gaze for MRD measurements.') + learnedTxt;

  const tips = suggestions(mp, lf, S.clinical, S.settings.normalMrd1);
  if (!mp) tips.unshift('Capture primary gaze to grade ptosis.');
  if (mp && !lf) tips.push('Levator function not yet measured — capture Down-gaze and Up-gaze.');
  if (Object.values(S.caps).some(c => c && c.manual)) tips.unshift('Face not detected automatically — markers were placed at default positions. Drag every marker onto the eye, including the two I markers onto the nasal and temporal limbus (they set the mm scale).');
  if (mp && !S.caps.primary.manual && EYES.some(s => !mp.eyes[s].reflexFound)) tips.push('Corneal reflex not detected in one eye — MRD1 estimated from iris centre. Verify, or re-capture with the light on.');
  el.interp.innerHTML = tips.map(t => `<li>${t}</li>`).join('');
}

// ---------- clinical reference values (validation) ----------
function buildRefGrid() {
  const head = '<thead><tr><th>mm</th><th>OD (R)</th><th>OS (L)</th></tr></thead>';
  const rows = PARAMS.map(([k, n]) => `<tr><td>${n}</td>${EYES.map(s => `<td><input type="number" inputmode="decimal" step="0.5" id="ref_${s}_${k}" aria-label="Clinical ${n} ${s}"></td>`).join('')}</tr>`).join('');
  $('refGrid').innerHTML = head + `<tbody>${rows}</tbody>`;
}
function readRef() {
  const ref = { OD: {}, OS: {} };
  let any = false;
  for (const s of EYES) for (const [k] of PARAMS) {
    const v = parseFloat($(`ref_${s}_${k}`).value);
    if (Number.isFinite(v)) { ref[s][k] = v; any = true; }
  }
  return any ? ref : null;
}
function appValues() {
  const { mp, lf } = computeAll();
  const out = { OD: {}, OS: {} };
  for (const s of EYES) {
    if (mp) for (const k of ['mrd1', 'mrd2', 'pfh', 'mcd']) out[s][k] = mp.eyes[s][k];
    if (lf) out[s].lf = lf[s];
  }
  return out;
}

function renderLearning() {
  const b = learnedBias();
  const fmt = (o, k) => (o.n >= MIN_SAMPLES ? `${o.mm > 0 ? '+' : ''}${o.mm.toFixed(2)} mm` : `${o.n}/${MIN_SAMPLES} eyes`);
  $('learnSummary').innerHTML = `<p class="hint" style="margin-top:0">Learned from <b>${b.eyes}</b> eye${b.eyes === 1 ? '' : 's'}.
    Upper lid ${fmt(b.upper)} · Lower lid ${fmt(b.lower)} · Crease ${fmt(b.crease)} ·
    Limbus ${b.limbus.n >= MIN_SAMPLES ? `×${b.limbus.scale.toFixed(3)}` : `${b.limbus.n}/${MIN_SAMPLES} eyes`}
    <br>(+ = examiners move the marker down; applied ${S.settings.learn ? 'to new captures' : '— switched off'})</p>`;
  $('learnOn').checked = S.settings.learn;
  const st = validationStats();
  const rows = PARAMS.filter(([k]) => st[k].n).map(([k, n]) => {
    const x = st[k];
    return `<tr><td>${n}</td><td>${x.n}</td><td>${x.bias.toFixed(2)}</td><td>${x.n > 1 ? `${x.lo.toFixed(1)} to ${x.hi.toFixed(1)}` : '—'}</td><td>${x.mae.toFixed(2)}</td><td>${Math.round(x.within1 * 100)}%</td></tr>`;
  }).join('');
  $('validSummary').innerHTML = rows
    ? `<table class="stats"><thead><tr><th>mm</th><th>n</th><th>Bias</th><th>95% LoA</th><th>MAE</th><th>±1 mm</th></tr></thead><tbody>${rows}</tbody></table>
       ${blandAltmanSVG(st.mrd1, 'MRD1')}<p class="hint">Bias = mean (app − clinical); LoA = bias ± 1.96 SD; MAE = mean absolute error. Plot: MRD1, both eyes pooled.</p>`
    : '<p class="hint">No clinical comparisons yet.</p>';
}

function reportText() {
  const R = computeAll(), { mp, lf } = R;
  const { rows, grade } = resultRows(R);
  const p = { id: $('pId').value.trim(), age: $('pAge').value.trim(), sex: $('pSex').value };
  const pad = (s, n) => String(s).padEnd(n);
  const L = [];
  L.push('PTOSIS EVALUATION');
  L.push(`Patient: ${p.id || '—'}${p.age ? `, ${p.age} y` : ''}${p.sex ? `, ${p.sex}` : ''}`);
  L.push(`Date: ${new Date().toLocaleString()}`);
  if (mp) L.push(`Calibration: ${mp.method}`);
  L.push('');
  L.push(`${pad('(mm)', 30)}${pad('OD', 16)}OS`);
  for (const r of rows) {
    if (r.OD == null && r.OS == null) continue;
    let a = f1(r.OD), b = f1(r.OS);
    if (r.key === 'lf' && lf) { a += ` ${lfGrade(lf.OD)}`; b += ` ${lfGrade(lf.OS)}`; }
    L.push(`${pad(r.name, 30)}${pad(a, 16)}${b}`);
  }
  if (mp) L.push(`${pad('Grade', 30)}${grade('OD').label} | ${grade('OS').label}`);
  const c = S.clinical;
  const tests = [];
  if (c.phenylephrine) tests.push(`Phenylephrine test ${c.phenylephrine}`);
  if (c.bells) tests.push(`Bell's ${c.bells}`);
  if (c.jawwink) tests.push('Jaw-winking present');
  if (c.fatigue) tests.push('Fatigability present');
  if (tests.length) { L.push(''); L.push(tests.join('; ')); }
  if (c.notes) L.push(`Notes: ${c.notes}`);
  const ref = readRef();
  if (ref) {
    L.push(''); L.push('Clinical measurements (ruler / slit-lamp):');
    for (const [k, n] of PARAMS) if (ref.OD[k] != null || ref.OS[k] != null) L.push(`${pad(n, 30)}${pad(f1(ref.OD[k]), 16)}${f1(ref.OS[k])}`);
  }
  if (S.caps.primary && S.caps.primary.learned) L.push(`Learned marker correction applied: ${describeLearned(S.caps.primary.learned)}`);
  const tips = suggestions(mp, lf, c, S.settings.normalMrd1);
  if (tips.length) { L.push(''); L.push('Interpretation:'); tips.forEach(t => L.push(`- ${t}`)); }
  L.push('');
  L.push('Photogrammetric estimate (Ptosis Evaluator); verify clinically.');
  return L.join('\n');
}

async function annotatedImage() {
  const cap = S.caps.primary || S.caps[S.rv.which];
  if (!cap) return null;
  const m = measureCap(cap);
  const box = bbox(capPoints(cap), 0.22, 0.15);
  const cw = 1600, ch = Math.round(cw * Math.max(0.45, box.h / box.w)), head = 120;
  const c = document.createElement('canvas');
  c.width = cw; c.height = ch + head;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#0f1620'; ctx.fillRect(0, 0, c.width, c.height);
  ctx.save(); ctx.translate(0, head);
  const T = makeT(box, cw, ch, false);
  drawImageView(ctx, cap.canvas, cap.W, cap.H, T);
  drawOverlay(ctx, T, cap, m, 2);
  ctx.restore();
  ctx.fillStyle = '#e8eef5'; ctx.font = '600 34px system-ui, sans-serif';
  ctx.fillText(`Ptosis evaluation — ${$('pId').value.trim() || 'patient'} — ${new Date().toLocaleDateString()}`, 24, 48);
  ctx.font = '26px system-ui, sans-serif'; ctx.fillStyle = '#93a4b8';
  if (cap.kind === 'primary') {
    const e = m.eyes;
    ctx.fillText(`OD: MRD1 ${f1(e.OD.mrd1)}  MRD2 ${f1(e.OD.mrd2)}  PFH ${f1(e.OD.pfh)}  MCD ${f1(e.OD.mcd)}     OS: MRD1 ${f1(e.OS.mrd1)}  MRD2 ${f1(e.OS.mrd2)}  PFH ${f1(e.OS.pfh)}  MCD ${f1(e.OS.mcd)}`, 24, 92);
  }
  return new Promise(res => c.toBlob(res, 'image/jpeg', 0.92));
}

// ---------------- history ----------------
function renderHistory() {
  const list = store.get('ptosis.history', []);
  $('histCount').textContent = list.length;
  $('historyList').innerHTML = list.length ? list.map((h, i) => `
    <div class="hist">
      <div class="hist-head"><b>${escapeHtml(h.id || 'Unnamed')}</b><span class="hint">${new Date(h.t).toLocaleString()}</span>
      <span><button data-copy="${i}">Copy</button> <button data-del="${i}">Delete</button></span></div>
      <pre>${escapeHtml(h.text)}</pre>
    </div>`).join('') : '<p class="hint">Nothing saved yet. Saved reports stay on this device only.</p>';
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// ---------------- wiring ----------------
function syncReviewInputs() {
  $('calib').value = S.settings.calib;
  $('hvid2').value = S.settings.hvid;
  $('rulerMm').value = S.settings.rulerMm;
  $('normalMrd1').value = S.settings.normalMrd1;
}

function msg(text) { $('reviewMsg').textContent = text; if (text) setTimeout(() => { if ($('reviewMsg').textContent === text) $('reviewMsg').textContent = ''; }, 3000); }

function init() {
  $('hvid').value = S.settings.hvid;
  $('facing').value = S.settings.facing;
  $('hvid').addEventListener('change', e => { S.settings.hvid = +e.target.value || 11.7; saveSettings(); });
  $('facing').addEventListener('change', e => { S.settings.facing = e.target.value; saveSettings(); });
  $('btnStart').addEventListener('click', () => goLive('primary'));
  $('fileHome').addEventListener('change', e => { loadPhoto(e.target.files[0], 'primary', $('homeMsg')); e.target.value = ''; });
  // Start loading the model early so the camera view is ready faster.
  ensureLandmarker().catch(e => { $('homeMsg').textContent = `Could not load face model: ${e.message}`; });

  $('btnBack').addEventListener('click', () => { stopLive(); show('home'); });
  $('btnTorch').addEventListener('click', () => setTorch(!S.torch));
  $('btnRing').addEventListener('click', () => setRing(!S.ring));
  $('btnAuto').addEventListener('click', () => { S.auto = !S.auto; S.live.okSince = null; updateToolButtons(); });
  // Live preview: pinch to zoom the camera, tap to toggle eye framing.
  const LP = new Map();
  let livePinch = null, pinched = 0;
  pinchable(el.liveCanvas, {
    begin: () => ({ z0: S.live.zf || 1 }),
    change: (st, e) => { setLiveZoom(st.z0 * e.scale); pinched = performance.now(); },
  });
  el.liveCanvas.addEventListener('pointerdown', e => {
    LP.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try { el.liveCanvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
    if (LP.size === 2) { const [a, b] = [...LP.values()]; livePinch = { d0: dist(a, b), z0: S.live.zf || 1 }; }
  });
  el.liveCanvas.addEventListener('pointermove', e => {
    if (!LP.has(e.pointerId)) return;
    LP.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (livePinch && LP.size === 2 && !HAS_GESTURE) {
      const [a, b] = [...LP.values()];
      setLiveZoom(livePinch.z0 * dist(a, b) / livePinch.d0);
      pinched = performance.now();
    }
  });
  const liveEnd = e => { LP.delete(e.pointerId); if (LP.size < 2) livePinch = null; };
  el.liveCanvas.addEventListener('pointerup', liveEnd);
  el.liveCanvas.addEventListener('pointercancel', liveEnd);
  el.liveCanvas.addEventListener('wheel', e => { e.preventDefault(); setLiveZoom((S.live.zf || 1) * Math.exp(-e.deltaY * 0.002)); }, { passive: false });
  // Tap to step through 1× → 2× → 3× → 1×.
  $('btnLiveZoom').addEventListener('click', () => {
    const z = S.live.zf || 1;
    setLiveZoom(z < 1.5 ? 2 : z < 2.5 ? 3 : 1);
  });
  el.liveCanvas.addEventListener('click', () => {
    if (performance.now() - pinched < 400) return; // end of a pinch, not a tap
    S.zoom = !S.zoom; S.live.view = null;
  });
  $('btnFlip').addEventListener('click', async () => {
    S.settings.facing = S.settings.facing === 'user' ? 'environment' : 'user';
    $('facing').value = S.settings.facing; saveSettings();
    S.live.cap = null;
    try { await startCamera(); } catch (e) { el.instruction.textContent = `⚠ ${e.message}`; }
  });
  $('btnCapture').addEventListener('click', captureLive);
  $('btnToReview').addEventListener('click', () => { stopLive(); S.rv.which = S.stage; openReview(); });
  $('stageSeg').addEventListener('click', e => { if (e.target.dataset.stage) setStage(e.target.dataset.stage); });

  $('btnReviewBack').addEventListener('click', () => goLive(S.rv.which));
  $('capSeg').addEventListener('click', e => { if (e.target.dataset.cap) { S.rv.which = e.target.dataset.cap; S.rv.sel = null; resetZoom(); updateSegs(); scheduleDraw(); } });
  $('viewSeg').addEventListener('click', e => { if (e.target.dataset.v) { S.rv.view = e.target.dataset.v; resetZoom(); updateSegs(); scheduleDraw(); } });
  $('btnCaptureThis').addEventListener('click', () => goLive(S.rv.which));
  $('btnRetake').addEventListener('click', () => goLive(S.rv.which));
  $('fileReview').addEventListener('change', e => { loadPhoto(e.target.files[0], S.rv.which, $('reviewMsg')); e.target.value = ''; });

  const num = (id, key, fallback) => $(id).addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    S.settings[key] = Number.isFinite(v) && v > 0 ? v : fallback;
    if (key === 'hvid') $('hvid').value = S.settings.hvid;
    saveSettings(); scheduleDraw();
  });
  num('hvid2', 'hvid', 11.7); num('rulerMm', 'rulerMm', 10); num('normalMrd1', 'normalMrd1', 4.5);
  $('calib').addEventListener('change', e => {
    S.settings.calib = e.target.value; saveSettings();
    if (S.caps.primary && !S.caps.primary.ruler) initRuler(S.caps.primary);
    scheduleDraw();
  });
  const clin = (id, key, prop = 'value') => $(id).addEventListener('input', e => { S.clinical[key] = e.target[prop]; renderResults(); });
  clin('phenyl', 'phenylephrine'); clin('bells', 'bells'); clin('jawwink', 'jawwink', 'checked'); clin('fatigue', 'fatigue', 'checked'); clin('notes', 'notes');

  $('btnCopy').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(reportText()); msg('Report copied.'); } catch { msg('Clipboard unavailable.'); }
  });
  $('btnShare').addEventListener('click', async () => {
    const text = reportText();
    try {
      const blob = await annotatedImage();
      const file = blob && new File([blob], 'ptosis.jpg', { type: 'image/jpeg' });
      if (file && navigator.canShare && navigator.canShare({ files: [file] })) await navigator.share({ title: 'Ptosis evaluation', text, files: [file] });
      else if (navigator.share) await navigator.share({ title: 'Ptosis evaluation', text });
      else { await navigator.clipboard.writeText(text); msg('Sharing not supported — report copied instead.'); }
    } catch (e) { if (e.name !== 'AbortError') msg(`Share failed: ${e.message}`); }
  });
  $('btnImage').addEventListener('click', async () => {
    const blob = await annotatedImage();
    if (!blob) { msg('Nothing captured yet.'); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `ptosis_${($('pId').value.trim() || 'patient').replace(/\W+/g, '_')}_${new Date().toISOString().slice(0, 10)}.jpg`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  });
  $('btnSave').addEventListener('click', () => {
    if (!S.caps.primary) { msg('Capture primary gaze first.'); return; }
    const cap = S.caps.primary, id = $('pId').value.trim();
    const list = store.get('ptosis.history', []).filter(h => h.cap !== cap.time);
    list.unshift({ t: Date.now(), id, cap: cap.time, text: reportText() });
    store.set('ptosis.history', list.slice(0, 100));
    const learned = recordCorrections(cap, measureForLearning);
    const ref = readRef();
    if (ref) recordValidation({ cap: cap.time, t: Date.now(), id, app: appValues(), ref, learned: cap.learned });
    renderHistory(); renderLearning();
    const n = loadSamples().length;
    msg(`Saved.${learned ? ` Learned from ${learned} eyes (${n} total${n < MIN_SAMPLES ? `; corrections start at ${MIN_SAMPLES}` : ''}).` : ''}${ref ? ' Clinical values recorded.' : ''}`);
  });
  $('learnOn').addEventListener('change', e => { S.settings.learn = e.target.checked; saveSettings(); renderLearning(); });
  $('btnResetLearn').addEventListener('click', () => { if (confirm('Delete all learned marker corrections?')) { resetLearning(); renderLearning(); } });
  $('btnResetValid').addEventListener('click', () => { if (confirm('Delete all clinical comparison data?')) { resetValidation(); renderLearning(); } });
  $('btnExportCsv').addEventListener('click', () => {
    if (!loadValidation().length) { alert('No clinical comparisons saved yet.'); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([validationCSV()], { type: 'text/csv' }));
    a.download = `ptosis_validation_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  });
  $('btnNew').addEventListener('click', () => {
    S.caps = { primary: null, down: null, up: null };
    S.clinical = { phenylephrine: '', bells: '', jawwink: false, fatigue: false, notes: '' };
    ['pId', 'pAge', 'notes'].forEach(id => { $(id).value = ''; });
    ['pSex', 'phenyl', 'bells'].forEach(id => { $(id).value = ''; });
    ['jawwink', 'fatigue'].forEach(id => { $(id).checked = false; });
    document.querySelectorAll('#refGrid input').forEach(i => { i.value = ''; });
    S.rv.which = 'primary'; S.rv.sel = null;
    show('home');
  });
  $('historyList').addEventListener('click', async e => {
    const list = store.get('ptosis.history', []);
    if (e.target.dataset.del != null) { list.splice(+e.target.dataset.del, 1); store.set('ptosis.history', list); renderHistory(); }
    if (e.target.dataset.copy != null) { try { await navigator.clipboard.writeText(list[+e.target.dataset.copy].text); e.target.textContent = 'Copied'; } catch { /* ignore */ } }
  });

  buildRefGrid();
  renderLearning();
  setupEditor();
  setStage('primary');
  updateToolButtons();
  renderHistory();
  document.addEventListener('visibilitychange', () => { if (document.hidden && S.live.running) stopLive(); else if (!document.hidden && $('live').classList.contains('active')) goLive(); });
}

init();

// Exposed for automated testing.
window.__ptosis = { S, loadPhoto, computeAll, reportText, learnedBias, validationStats };
