// Copies third-party runtime files into the published site (run after `npm install`):
// ONNX Runtime Web (WASM build only) -> site/vendor/, Archivo variable font (latin) -> site/fonts/.
import { copyFileSync, mkdirSync } from "node:fs";

const copies = [
  ["node_modules/onnxruntime-web/dist/ort.wasm.min.js", "site/vendor/ort.wasm.min.js"],
  ["node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", "site/vendor/ort-wasm-simd-threaded.mjs"],
  ["node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", "site/vendor/ort-wasm-simd-threaded.wasm"],
  ["node_modules/@fontsource-variable/archivo/files/archivo-latin-standard-normal.woff2", "site/fonts/archivo.woff2"],
  ["node_modules/@fontsource-variable/archivo/LICENSE", "site/fonts/ARCHIVO-LICENSE.txt"],
];
for (const dir of ["site/vendor", "site/fonts"]) mkdirSync(dir, { recursive: true });
for (const [from, to] of copies) copyFileSync(from, to);
console.log("copied:", copies.map(([, to]) => to).join(", "));
