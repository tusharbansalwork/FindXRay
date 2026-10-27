"""Serve the repo locally with the headers browsers need for multi-threaded WASM.

Usage (from anywhere):  python tools/browser_bench/serve.py
Then open:              http://localhost:8000/tools/browser_bench/

Multi-threaded WebAssembly needs SharedArrayBuffer, which browsers only enable on
"cross-origin isolated" pages (COOP + COEP headers). Opening index.html as a file
cannot set those headers, so use this server.
"""

import functools
import http.server
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORT = 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"Serving {ROOT}")
        print(f"Open http://localhost:{PORT}/tools/browser_bench/   (Ctrl+C to stop)")
        httpd.serve_forever()
