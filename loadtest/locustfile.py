"""Load test for the FindXRay search API (Locust).

Each simulated visitor sends what the website sends after encoding an X-ray: a 768-number embedding
to /search, then pauses 1-3 s, like someone reading results. Embeddings are real library X-rays with a
little noise added, so no two requests are identical. /report is not load-tested: it calls Gemini, whose
free quota is the real limit (60 reports a day by design).

Run (headless, CSV results):
  pip install -e ".[load]"
  locust -f loadtest/locustfile.py --headless --host https://<api-url> -u 50 -r 5 -t 3m --csv results/03_cloud/aws
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]


def _library() -> np.ndarray:
    for d in (ROOT / "artifacts" / "library", ROOT / "build" / "lambda" / "library", ROOT / "build" / "hf_library" / "library"):
        if (d / "embeddings.npy").exists():
            return np.load(d / "embeddings.npy")
    raise SystemExit("no case library found: run scripts/build_library.py or scripts/build_lambda.py first")


EMB = _library()
ORIGIN = os.environ.get("LOAD_ORIGIN", "https://findxray.onrender.com")


class Visitor(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.rng = np.random.default_rng()

    @task
    def search(self):
        q = EMB[self.rng.integers(len(EMB))] + self.rng.normal(0, 0.01, EMB.shape[1]).astype(np.float32)
        q /= np.linalg.norm(q)
        with self.client.post("/search", data=json.dumps({"embedding": q.tolist(), "k": 5}),
                              headers={"Content-Type": "application/json", "Origin": ORIGIN},
                              name="/search", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif len(r.json().get("results", [])) != 5:
                r.failure("expected 5 results")
