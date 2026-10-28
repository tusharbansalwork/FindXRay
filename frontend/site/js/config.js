// Where the page finds the API and the encoder model. For local testing, override with
// ?api=http://127.0.0.1:8000&model=/path/to/model.onnx in the page URL.
const params = new URLSearchParams(location.search);
const local = ["localhost", "127.0.0.1"].includes(location.hostname);

export const API_BASE = params.get("api") || (local ? "http://127.0.0.1:8000" : "https://findxray-api.onrender.com");
export const MODEL_URL = params.get("model")
  || "https://huggingface.co/tush-2ar/findxray-encoder/resolve/main/rad-dino-224.int8.onnx";
export const MODEL_MB = 86;
export const LIBRARY_SIZE = 2517;
export const REPO_URL = "https://github.com/tusharbansalwork/FindXRay";
