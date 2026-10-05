// Liquid glass for controls that float over the camera image or the photo.
// For each [data-liquid] element, build a displacement map from the signed
// distance field of its rounded rectangle: pixels near the edge are pushed
// along the edge normal, so the image behind bends like a lens rim. Applied
// through an SVG <feDisplacementMap> in backdrop-filter where the browser
// supports it; otherwise the plain blur from .glass in CSS stays in place.

const NS = 'http://www.w3.org/2000/svg';
let svg = null, supported = null, uid = 0;
const done = new WeakMap(); // element -> "w×h×r" the map was built for

function canUseSvgBackdrop() {
  if (supported != null) return supported;
  // WebKit parses url() in backdrop-filter but does not render it.
  const webkit = /AppleWebKit/.test(navigator.userAgent) && !/Chrome|Chromium|Edg\//.test(navigator.userAgent);
  supported = !webkit && typeof CSS !== 'undefined' && CSS.supports('backdrop-filter', 'url(#x)');
  return supported;
}

// Signed distance from (px, py) to a rounded rectangle centred at the origin.
function sdRoundRect(px, py, hw, hh, r) {
  const qx = Math.abs(px) - (hw - r), qy = Math.abs(py) - (hh - r);
  const ox = Math.max(qx, 0), oy = Math.max(qy, 0);
  return Math.hypot(ox, oy) + Math.min(Math.max(qx, qy), 0) - r;
}

function displacementMap(w, h, r, bezel) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(w, h), d = img.data;
  const hw = w / 2, hh = h / 2, e = 0.5;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const px = x + 0.5 - hw, py = y + 0.5 - hh;
      const dist = sdRoundRect(px, py, hw, hh, r);
      let dx = 0, dy = 0;
      if (dist < 0 && -dist < bezel) {
        // gradient of the SDF = outward normal
        const nx = sdRoundRect(px + e, py, hw, hh, r) - sdRoundRect(px - e, py, hw, hh, r);
        const ny = sdRoundRect(px, py + e, hw, hh, r) - sdRoundRect(px, py - e, hw, hh, r);
        const len = Math.hypot(nx, ny) || 1;
        const k = (1 - -dist / bezel) ** 2; // strongest at the rim
        dx = -nx / len * k; dy = -ny / len * k; // sample inward: refraction
      }
      const i = (y * w + x) * 4;
      d[i] = 128 + dx * 127; d[i + 1] = 128 + dy * 127; d[i + 2] = 128; d[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL();
}

function ensureSvg() {
  if (svg) return svg;
  svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', '0'); svg.setAttribute('height', '0');
  svg.setAttribute('aria-hidden', 'true');
  svg.style.position = 'absolute';
  document.body.appendChild(svg);
  return svg;
}

function apply(el) {
  const w = Math.round(el.offsetWidth), h = Math.round(el.offsetHeight);
  if (!w || !h) return; // hidden; built when it becomes visible
  const r = Math.min(parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0, w / 2, h / 2);
  const key = `${w}x${h}x${r}`;
  if (done.get(el) === key) return;
  done.set(el, key);
  const id = el.dataset.liquidId || (el.dataset.liquidId = `lg${++uid}`);
  const root = ensureSvg();
  root.querySelector(`#${id}`)?.remove();
  const f = document.createElementNS(NS, 'filter');
  f.setAttribute('id', id);
  f.setAttribute('x', '0'); f.setAttribute('y', '0'); f.setAttribute('width', '100%'); f.setAttribute('height', '100%');
  f.setAttribute('color-interpolation-filters', 'sRGB');
  const fi = document.createElementNS(NS, 'feImage');
  fi.setAttribute('href', displacementMap(w, h, r, Math.min(12, Math.min(w, h) / 3)));
  fi.setAttribute('x', '0'); fi.setAttribute('y', '0'); fi.setAttribute('width', w); fi.setAttribute('height', h);
  fi.setAttribute('result', 'map');
  const dm = document.createElementNS(NS, 'feDisplacementMap');
  dm.setAttribute('in', 'SourceGraphic'); dm.setAttribute('in2', 'map');
  dm.setAttribute('scale', '8'); dm.setAttribute('xChannelSelector', 'R'); dm.setAttribute('yChannelSelector', 'G');
  f.append(fi, dm);
  root.appendChild(f);
  el.style.backdropFilter = `url(#${id}) blur(0.3px) saturate(1.3)`;
}

export function refreshLiquidGlass() {
  if (!canUseSvgBackdrop()) return;
  document.querySelectorAll('[data-liquid]').forEach(apply);
}

export function initLiquidGlass() {
  if (!canUseSvgBackdrop()) return;
  const ro = new ResizeObserver(entries => entries.forEach(e => apply(e.target)));
  document.querySelectorAll('[data-liquid]').forEach(el => ro.observe(el));
  refreshLiquidGlass();
}
