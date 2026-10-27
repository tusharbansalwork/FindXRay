"""Try the report agent on one held-out test X-ray, with real Gemini calls and Langfuse tracing.

Prints the verified report, removed sentences, the cited cases, and the radiologist's actual
report for comparison. Reads keys from .env (see .env.example).

Usage:
  python scripts/draft_report.py                 # a random test X-ray
  python scripts/draft_report.py --uid 2042      # a specific test study
  python scripts/draft_report.py --list-models   # show Gemini models your key can use
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casefinder.agent import GeminiReportModel, XRayImage, run_report  # noqa: E402
from casefinder.agent.tracing import callback_handlers, flushing, trace_attributes, trace_url  # noqa: E402
from casefinder.labels import mesh_to_groups  # noqa: E402
from casefinder.search import CaseLibrary  # noqa: E402

SOURCE = "datasets/ykumards/open-i"


def load_study(uid: int) -> dict:
    """Fetch one study's frontal image and report from the source dataset (reads one row group only)."""
    fs = HfFileSystem()
    for shard in sorted(fs.glob(f"{SOURCE}/data/*.parquet")):
        pf = pq.ParquetFile(shard, filesystem=fs)
        for rg in range(pf.num_row_groups):
            uids = pf.read_row_group(rg, columns=["uid"]).column("uid").to_pylist()
            if uid in uids:
                row = pf.read_row_group(rg, columns=["uid", "MeSH", "indication", "findings", "impression",
                                                     "img_frontal"]).to_pylist()[uids.index(uid)]
                return row
    raise SystemExit(f"uid {uid} not found in {SOURCE}")


def list_models() -> None:
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    for m in client.models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []):
            print(m.name.removeprefix("models/"))


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", type=int)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    if args.list_models:
        return list_models()

    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    test = splits.loc[splits["split"] == "test", "uid"].astype(int).tolist()
    uid = args.uid
    if uid is None:
        uid = random.choice(test)
    elif uid not in test:
        raise SystemExit(f"uid {uid} is not in the test split; pick one that is (library cases would find themselves)")

    study = load_study(uid)
    data = study["img_frontal"]
    mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"

    uids = np.load(ROOT / "artifacts" / "embeddings" / "uids.npy")
    emb = np.load(ROOT / "artifacts" / "embeddings" / "rad-dino-224.npy")[list(uids).index(uid)]
    library = CaseLibrary.load(ROOT / "artifacts" / "library")
    model = GeminiReportModel()
    handlers = callback_handlers()

    print(f"Test study {uid} | true labels: {', '.join(sorted(mesh_to_groups(study['MeSH']) or []))}")
    print(f"Model {model.name} | Langfuse tracing {'on' if handlers else 'off'}\n")
    print(f"Thinking {model.thinking_level} | image resolution {model.media_resolution}\n")
    t0 = time.perf_counter()
    with flushing(), trace_attributes(model=model.name, k=args.k, uid=uid, source="draft_report.py"):
        result = run_report(library, model, XRayImage(data, mime), emb, k=args.k, callbacks=handlers)
        elapsed = time.perf_counter() - t0
        url = trace_url(handlers)

    wrap = lambda s: textwrap.fill(s, 100, subsequent_indent="      ")  # noqa: E731
    print("DRAFT REPORT (verified sentences)")
    for section in ("findings", "impression"):
        for s in (x for x in result.report if x.section == section):
            print(wrap(f"  [{section[0].upper()}] {s.text}  <- {', '.join(s.sources)}"))
    if result.removed:
        print("\nREMOVED (failed verification)")
        for s in result.removed:
            print(wrap(f"  - {s.text}  <- {', '.join(s.sources)} | {s.reason}"))
    print(f"\ntime: {elapsed:.1f} s | revisions: {result.revisions} | first-draft sentences supported: {result.first_pass_supported:.0%}"
          f" | prompt versions: {result.prompt_versions}")
    print("\nSIMILAR CASES")
    for c in result.cases:
        print(wrap(f"  case:{c['uid']} sim {c['similarity']:.2f} [{', '.join(c['groups'])}] {c.get('impression') or ''}"))
    print("\nRADIOLOGIST'S ACTUAL REPORT")
    print(wrap(f"  findings: {study['findings'] or '-'}"))
    print(wrap(f"  impression: {study['impression'] or '-'}"))
    if url:
        print(f"\nLangfuse trace: {url}")


if __name__ == "__main__":
    main()
