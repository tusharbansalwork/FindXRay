// Copies third-party runtime files into the published site (run after `npm ci`):
// ONNX Runtime Web (WASM build only) -> site/vendor/; Source Serif 4 + Instrument Sans (latin) -> site/fonts/.
import { copyFileSync, mkdirSync } from "node:fs";

const nm = "node_modules/";
const copies = [
  [nm + "onnxruntime-web/dist/ort.wasm.min.js", "site/vendor/ort.wasm.min.js"],
  [nm + "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", "site/vendor/ort-wasm-simd-threaded.mjs"],
  [nm + "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", "site/vendor/ort-wasm-simd-threaded.wasm"],
  [nm + "@fontsource-variable/source-serif-4/files/source-serif-4-latin-opsz-normal.woff2", "site/fonts/source-serif-4.woff2"],
  [nm + "@fontsource-variable/instrument-sans/files/instrument-sans-latin-standard-normal.woff2", "site/fonts/instrument-sans.woff2"],
  [nm + "@fontsource-variable/source-serif-4/LICENSE", "site/fonts/SOURCE-SERIF-4-LICENSE.txt"],
  [nm + "@fontsource-variable/instrument-sans/LICENSE", "site/fonts/INSTRUMENT-SANS-LICENSE.txt"],
];
for (const dir of ["site/vendor", "site/fonts"]) mkdirSync(dir, { recursive: true });
for (const [from, to] of copies) copyFileSync(from, to);
console.log("copied:", copies.map(([, to]) => to).join(", "));
