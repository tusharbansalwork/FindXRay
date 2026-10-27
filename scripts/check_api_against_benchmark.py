"""End-to-end check: query the running API with every test-split embedding and
recompute retrieval metrics from the HTTP responses. They must match notebook 01
for the same model (results/01_encoder_benchmark/results_retrieval.csv).

Usage:
  uvicorn casefinder.api:app --app-dir src      (in one terminal)
  python scripts/check_api_against_benchmark.py [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casefinder.labels import mesh_to_groups  # noqa: E402

MODEL = "rad-dino-224"
K = 10


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    uids = np.load(ROOT / "artifacts" / "embeddings" / "uids.npy")
    emb = np.load(ROOT / "artifacts" / "embeddings" / f"{MODEL}.npy")
    row_of = {int(u): i for i, u in enumerate(uids)}

    fs = HfFileSystem()
    meta = pd.concat([pq.read_table(s, columns=["uid", "MeSH"], filesystem=fs).to_pandas()
                      for s in sorted(fs.glob("datasets/ykumards/open-i/data/*.parquet"))])
    groups_of = {int(u): mesh_to_groups(m) for u, m in zip(meta["uid"], meta["MeSH"])}

    test_uids = splits.loc[splits["split"] == "test", "uid"].astype(int).tolist()
    train_uids = splits.loc[splits["split"] == "train", "uid"].astype(int).tolist()
    lib_groups = [groups_of[u] for u in train_uids]

    health = json.loads(urllib.request.urlopen(f"{args.url}/health").read())
    print("health:", health)

    p5, ap10, latencies, abnormal = [], [], [], []
    for u in test_uids:
        t = time.perf_counter()
        res = post(f"{args.url}/search", {"embedding": emb[row_of[u]].tolist(), "k": K})["results"]
        latencies.append(1000 * (time.perf_counter() - t))
        q = groups_of[u]
        rel = [bool(q & set(r["groups"])) for r in res]
        n_rel = sum(bool(q & g) for g in lib_groups)
        p5.append(sum(rel[:5]) / 5)
        hits, ap = 0, 0.0
        for i, r in enumerate(rel, 1):
            if r:
                hits += 1
                ap += hits / i
        ap10.append(ap / max(min(K, n_rel), 1))
        abnormal.append(q != {"No Finding"})

    bench = pd.read_csv(ROOT / "results" / "01_encoder_benchmark" / "results_retrieval.csv")
    def ref(metric, sl):
        return bench[(bench.model == MODEL) & (bench.slice == sl) & (bench.metric == metric)]["mean"].item()

    ab = np.array(abnormal)
    rows = [
        ("P@5, all", np.mean(p5), ref("P@5", "all")),
        ("mAP@10, all", np.mean(ap10), ref("mAP@10", "all")),
        ("P@5, abnormal", np.mean(np.array(p5)[ab]), ref("P@5", "abnormal")),
        ("mAP@10, abnormal", np.mean(np.array(ap10)[ab]), ref("mAP@10", "abnormal")),
    ]
    print(f"\n{len(test_uids)} test queries through {args.url}/search")
    print(f"{'metric':18s} {'via API':>9s} {'notebook':>9s} {'diff':>9s}")
    ok = True
    for name, got, want in rows:
        diff = got - want
        ok &= abs(diff) < 1e-6
        print(f"{name:18s} {got:9.4f} {want:9.4f} {diff:+9.1e}")
    lat = np.array(latencies)
    print(f"\nlatency per request (local, includes JSON of 768 floats): median {np.median(lat):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms")
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
