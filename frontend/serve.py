"""Preview the site locally with the headers the in-browser encoder needs (COOP/COEP).

  cd frontend && npm ci && node build.mjs     # once: copies ONNX Runtime Web and the font into site/
  python frontend/serve.py                    # http://localhost:8080
  uvicorn casefinder.api:app --app-dir src --env-file .env    # the API, in another terminal

Before the model is on Hugging Face, add ?model=/model.onnx and copy artifacts/onnx/rad-dino-224.int8.onnx
to frontend/site/model.onnx (git ignores it).
"""

import functools
import http.server
from pathlib import Path

SITE = Path(__file__).resolve().parent / "site"
PORT = 8080


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=str(SITE))) as httpd:
        print(f"Open http://localhost:{PORT}/   (Ctrl+C to stop)")
        httpd.serve_forever()
