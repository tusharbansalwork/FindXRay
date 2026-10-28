// In-browser RAD-DINO encoder: image file -> 768-number embedding (ONNX Runtime Web, int8, WASM).
// Needs the global `ort` from vendor/ort.wasm.min.js. Multi-threading needs a cross-origin isolated page.

import { rgbaToGray, toTensor, CROP } from "./preprocess.js";

const CACHE = "findxray-model-v1";

// Decode to grayscale pixels at full resolution (no colour management, so pixel values match Pillow).
export async function decodeGray(blob) {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const { width: w, height: h } = bmp;
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement("canvas"), { width: w, height: h });
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  bmp.close?.();
  return { gray: rgbaToGray(ctx.getImageData(0, 0, w, h).data, w, h), width: w, height: h };
}

// Download with progress; keep a copy in Cache Storage so later visits skip the 86 MB download.
async function fetchModel(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) { onProgress?.(1, true); return new Uint8Array(await hit.arrayBuffer()); }
  } catch { cache = null; }

  const res = await fetch(url, { mode: "cors" });
  if (!res.ok) throw new Error(`model download failed (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress?.(got / total, false);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { await cache?.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota or private mode: fine */ }
  onProgress?.(1, false);
  return buf;
}

export class Encoder {
  constructor(modelUrl, { wasmPaths = new URL("../vendor/", import.meta.url).href } = {}) {
    this.modelUrl = modelUrl;
    this.wasmPaths = wasmPaths;
    this.session = null;
    this._loading = null;
  }

  load(onProgress) {
    this._loading ??= (async () => {
      ort.env.wasm.wasmPaths = this.wasmPaths;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? 0 : 1;   // 0 = let ORT choose
      const bytes = await fetchModel(this.modelUrl, onProgress);
      this.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return this;
    })().catch((e) => { this._loading = null; throw e; });
    return this._loading;
  }

  async embedPixels(gray, width, height) {
    await this.load();
    const input = new ort.Tensor("float32", toTensor(gray, width, height), [1, 3, CROP, CROP]);
    const out = await this.session.run({ [this.session.inputNames[0]]: input });
    return Float32Array.from(out[this.session.outputNames[0]].data);
  }

  async embed(blob) {
    const t0 = performance.now();
    const { gray, width, height } = await decodeGray(blob);
    const embedding = await this.embedPixels(gray, width, height);
    return { embedding, ms: performance.now() - t0, width, height };
  }
}
