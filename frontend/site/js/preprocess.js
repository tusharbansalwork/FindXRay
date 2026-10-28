// X-ray -> model input, reproducing notebook 01's Python preprocessing exactly:
//   PIL .convert("L") -> resize short side to 518 (bicubic) -> resize short side to 224 (bicubic)
//   -> centre crop 224 -> /255 -> normalise with RAD-DINO's mean/std -> 3 identical channels.
// The resize is a port of Pillow's ImagingResample (8-bit fixed point, horizontal pass then vertical),
// so the browser feeds the model the same pixels the library embeddings were built from.

export const STORE_SHORT = 518;   // notebook 01 kept every image at 518 px on the short side first
export const CROP = 224;
export const MEAN = 0.5307;
export const STD = 0.2583;

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;

function bicubic(x) {
  const a = -0.5;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc for one axis.
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const pre = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) { pre[x] = bicubic((x + xmin - center + 0.5) * ss); ww += pre[x]; }
    for (let x = 0; x < xmax; x++) {
      const k = ww !== 0 ? pre[x] / ww : pre[x];
      kk[xx * ksize + x] = k < 0 ? Math.trunc(-0.5 + k * ONE) : Math.trunc(0.5 + k * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { kk, bounds, ksize };
}

function clip8(v) {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

// Bicubic resize of an 8-bit grayscale image, bit-exact with Pillow's Image.resize(size, BICUBIC).
export function resizeGray(src, w, h, outW, outH) {
  let cur = src, cw = w;
  if (outW !== w) {
    const { kk, bounds, ksize } = coefficients(w, outW);
    const tmp = new Uint8Array(outW * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], k = xx * ksize;
        let ss = ONE / 2;
        for (let x = 0; x < xmax; x++) ss += src[row + xmin + x] * kk[k + x];
        tmp[y * outW + xx] = clip8(ss);
      }
    }
    cur = tmp; cw = outW;
  }
  if (outH !== h) {
    const { kk, bounds, ksize } = coefficients(h, outH);
    const out = new Uint8Array(cw * outH);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], k = yy * ksize;
      for (let x = 0; x < cw; x++) {
        let ss = ONE / 2;
        for (let y = 0; y < ymax; y++) ss += cur[(ymin + y) * cw + x] * kk[k + y];
        out[yy * cw + x] = clip8(ss);
      }
    }
    cur = out;
  }
  return cur;
}

// Python's round() (half to even), used by the notebook's resize_short.
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export function shortSideSize(w, h, s) {
  const r = s / Math.min(w, h);
  return [Math.max(s, pyRound(w * r)), Math.max(s, pyRound(h * r))];
}

// RGBA pixels (from a canvas) -> Pillow "L" (ITU-R 601-2 luma, Pillow's integer formula).
export function rgbaToGray(rgba, w, h) {
  const g = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) {
    g[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >>> 16;
  }
  return g;
}

// Grayscale image -> Float32Array [3, 224, 224] ready for the encoder.
export function toTensor(gray, w, h) {
  let [w1, h1] = shortSideSize(w, h, STORE_SHORT);
  let img = (w1 === w && h1 === h) ? gray : resizeGray(gray, w, h, w1, h1);
  let [w2, h2] = Math.min(w1, h1) === CROP ? [w1, h1] : shortSideSize(w1, h1, CROP);
  if (w2 !== w1 || h2 !== h1) img = resizeGray(img, w1, h1, w2, h2);
  const left = Math.floor((w2 - CROP) / 2), top = Math.floor((h2 - CROP) / 2);
  const n = CROP * CROP, out = new Float32Array(3 * n);
  const f = Math.fround, mean = f(MEAN), std = f(STD);
  for (let y = 0; y < CROP; y++) {
    for (let x = 0; x < CROP; x++) {
      const v = f(f(f(img[(top + y) * w2 + left + x]) / 255) - mean) / std;
      const i = y * CROP + x;
      out[i] = out[n + i] = out[2 * n + i] = f(v);
    }
  }
  return out;
}
