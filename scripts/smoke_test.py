"""Post-deploy smoke test for the FindXRay API (run by CI after every deploy, or by hand).

Checks, without spending any Gemini quota:
  1. /health: library loaded (2,517 cases), report drafting configured
  2. /search: 10 library X-rays each find themselves as the top match (similarity ~1.0)
  3. CORS: the website's origin is allowed, an unknown origin is not
  4. /report/stream rejects a bad image with 422 before any model call
  5. warm /search latency over 30 requests, on new connections and on one reused connection (median, p95)
Exits with status 1 if any check fails.

Usage:
  python scripts/smoke_test.py --url https://<id>.lambda-url.ap-southeast-2.on.aws
  python scripts/smoke_test.py --url http://127.0.0.1:8000 --origin http://localhost:8080 --no-reports
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from keepalive_http import KeepAlive  # scripts/ is on sys.path when run as a script

ROOT = Path(__file__).resolve().parents[1]


def request(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def library_files() -> tuple[np.ndarray, list[dict]]:
    for d in (ROOT / "build" / "lambda" / "library", ROOT / "artifacts" / "library", ROOT / "build" / "hf_library" / "library"):
        if (d / "embeddings.npy").exists():
            return np.load(d / "embeddings.npy"), json.loads((d / "cases.json").read_text())
    sys.exit("no case library found (build/lambda/library or artifacts/library)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--origin", default="https://findxray.onrender.com")
    ap.add_argument("--no-reports", action="store_true", help="don't require report drafting to be configured")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            failures.append(what)

    t = time.perf_counter()
    status, _, raw = request("GET", f"{base}/health", timeout=120)
    first_ms = 1000 * (time.perf_counter() - t)
    health = json.loads(raw) if status == 200 else {}
    check(status == 200 and health.get("library_size") == 2517, f"/health ok, 2,517 cases ({first_ms:.0f} ms, may include a cold start)")
    if not args.no_reports:
        check(health.get("reports_enabled") is True, "report drafting configured (Gemini key loaded)")

    emb, cases = library_files()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(cases), 10, replace=False):
        status, _, raw = request("POST", f"{base}/search", {"embedding": emb[i].tolist(), "k": 5})
        top = json.loads(raw)["results"][0] if status == 200 else {}
        check(top.get("uid") == cases[i]["uid"] and top.get("similarity", 0) > 0.999,
              f"/search: case {cases[i]['uid']} finds itself first")

    pre = {"Origin": args.origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    _, h, _ = request("OPTIONS", f"{base}/search", headers=pre)
    allowed = {k.lower(): v for k, v in h.items()}.get("access-control-allow-origin")
    check(allowed == args.origin, f"CORS allows {args.origin}")
    _, h, _ = request("OPTIONS", f"{base}/search", headers={**pre, "Origin": "https://evil.example"})
    check("access-control-allow-origin" not in {k.lower() for k in h}, "CORS rejects an unknown origin")

    status, _, _ = request("POST", f"{base}/report/stream",
                           {"embedding": emb[0].tolist(), "image_base64": "bm90IGFuIGltYWdl", "mime_type": "image/jpeg"})
    check(status == 422, "/report/stream rejects a non-image with 422 (no model call)")

    def summary(lat: list[float]) -> str:
        lat = sorted(lat)
        return f"median {statistics.median(lat):.0f} ms, p95 {lat[int(0.95 * len(lat)) - 1]:.0f} ms"

    fresh, reused = [], []
    for i in rng.choice(len(cases), 30):   # a new TCP + TLS connection each time (worst case: first request)
        t = time.perf_counter()
        request("POST", f"{base}/search", {"embedding": emb[i].tolist(), "k": 5})
        fresh.append(1000 * (time.perf_counter() - t))
    client = KeepAlive(base)
    client.request("GET", "/health")       # open the connection once, as a browser does
    for i in rng.choice(len(cases), 30):
        t = time.perf_counter()
        client.post_json("/search", {"embedding": emb[i].tolist(), "k": 5})
        reused.append(1000 * (time.perf_counter() - t))
    client.close()
    print(f"warm /search from this machine, new connection each time: {summary(fresh)} (30 requests)")
    print(f"warm /search from this machine, connection reused (what a browser sees): {summary(reused)} (30 requests)")

    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
