// Copies third-party runtime files into the published site (run after `npm ci`):
// ONNX Runtime Web (WASM build only) -> site/vendor/; Instrument Serif + Instrument Sans (latin) -> site/fonts/.
import { copyFileSync, mkdirSync } from "node:fs";

const nm = "node_modules/";
const copies = [
  [nm + "onnxruntime-web/dist/ort.wasm.min.js", "site/vendor/ort.wasm.min.js"],
  [nm + "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", "site/vendor/ort-wasm-simd-threaded.mjs"],
  [nm + "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", "site/vendor/ort-wasm-simd-threaded.wasm"],
  [nm + "@fontsource/instrument-serif/files/instrument-serif-latin-400-normal.woff2", "site/fonts/instrument-serif.woff2"],
  [nm + "@fontsource-variable/instrument-sans/files/instrument-sans-latin-standard-normal.woff2", "site/fonts/instrument-sans.woff2"],
  [nm + "@fontsource/instrument-serif/LICENSE", "site/fonts/INSTRUMENT-SERIF-LICENSE.txt"],
  [nm + "@fontsource-variable/instrument-sans/LICENSE", "site/fonts/INSTRUMENT-SANS-LICENSE.txt"],
];
for (const dir of ["site/vendor", "site/fonts"]) mkdirSync(dir, { recursive: true });
for (const [from, to] of copies) copyFileSync(from, to);
console.log("copied:", copies.map(([, to]) => to).join(", "));
