"""Generate reports for the evaluation sample. Resumable: rerun to continue or to retry failures.

Systems (scored later by scripts/eval_score.py; the nearest-case baseline needs no API calls):
  agent       full LangGraph agent (retrieve, draft, verify, revise); also yields its first draft
  image_only  one Gemini draft call with the same prompt but no similar cases

Each result is appended to results/02_report_eval/generations.jsonl as soon as it finishes, with
its Langfuse trace id. Traces are tagged "eval-v1" and carry system/uid metadata.

Usage:
  python scripts/eval_generate.py                    # 50-study sample, both systems
  python scripts/eval_generate.py --n 10             # smaller first pass
  python scripts/eval_generate.py --pause 8          # seconds between calls (free-tier rate limits)
  python scripts/eval_generate.py --start 11 --n 60  # studies 11-60 of the fixed sample (held-out final set)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casefinder.agent import GeminiReportModel, XRayImage, run_report  # noqa: E402
from casefinder.agent.tracing import callback_handlers, flushing, trace_attributes  # noqa: E402
from casefinder.search import CaseLibrary  # noqa: E402
from casefinder.studies import fetch_studies, fetch_texts, reference_text  # noqa: E402

OUT_DIR = ROOT / "results" / "02_report_eval"
EVAL_TAG = "eval-v1"


def choose_sample(n: int, seed: int) -> list[int]:
    """A fixed random sample of test-split studies that have a radiologist report. Saved for reuse."""
    path = OUT_DIR / "sample.csv"
    if path.exists():
        saved = pd.read_csv(path)["uid"].astype(int).tolist()
        if len(saved) >= n:
            return saved[:n]
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    test = sorted(splits.loc[splits["split"] == "test", "uid"].astype(int))
    texts = fetch_texts()
    with_report = [u for u in test if reference_text(texts[u])]
    # Shuffle everything once with a fixed seed and take a prefix, so a larger --n extends the same sample.
    chosen = random.Random(seed).sample(with_report, len(with_report))[:n]
    pd.DataFrame({"uid": chosen}).to_csv(path, index=False)
    return chosen


def done_keys(path: Path, model: str, config: str) -> set[tuple[str, int]]:
    """(system, uid) pairs already generated successfully with THIS model and config label.
    Results from other models/configs are ignored (records without a label count as "v1")."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("ok") and rec.get("model") == model and rec.get("config", "v1") == config:
            keys.add((rec["system"], rec["uid"]))
    return keys


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="use studies up to this position in the fixed sample")
    ap.add_argument("--start", type=int, default=1, help="first position to use (1-based), e.g. 11 for the final set")
    ap.add_argument("--config", default="v2", help="label stored with each result; scoring filters on it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--systems", default="agent,image_only")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pause", type=float, default=4.0, help="seconds to wait after each study/system")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "generations.jsonl"
    sample = choose_sample(args.n, args.seed)[args.start - 1:]
    studies = fetch_studies(sample, OUT_DIR / "cache")
    library = CaseLibrary.load(ROOT / "artifacts" / "library")
    uids_all = list(np.load(ROOT / "artifacts" / "embeddings" / "uids.npy"))
    emb_all = np.load(ROOT / "artifacts" / "embeddings" / "rad-dino-224.npy")
    model = GeminiReportModel()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    done = done_keys(out_path, model.name, args.config)
    todo = [(s, u) for u in sample for s in systems if (s, u) not in done]
    print(f"studies {args.start}-{args.start + len(sample) - 1} of the sample | config {args.config} | "
          f"systems {systems} | model {model.name} | "
          f"thinking {model.thinking_level} | pacing {model.rpm:g} calls/min | done {len(done)} | to do {len(todo)}")

    for i, (system, uid) in enumerate(todo, 1):
        study = studies[uid]
        image = XRayImage(study["image"], study["mime_type"])
        emb = emb_all[uids_all.index(uid)]
        handlers = callback_handlers()
        rec = {"system": system, "uid": uid, "model": model.name, "config": args.config,
               "thinking": model.thinking_level,
               "media_resolution": model.media_resolution, "k": args.k,
               "timestamp": datetime.now(timezone.utc).isoformat()}
        t0 = time.perf_counter()
        try:
            with flushing(), trace_attributes(name=f"findxray-eval-{system}", tags=[EVAL_TAG, system],
                                              system=system, uid=uid, model=model.name):
                if system == "agent":
                    r = run_report(library, model, image, emb, k=args.k, callbacks=handlers)
                    rec |= {"report": [s.model_dump() for s in r.report],
                            "removed": [s.model_dump() for s in r.removed],
                            "first_draft": [s.model_dump() for s in r.first_draft],
                            "missed_findings": [m.model_dump() for m in r.missed_findings],
                            "cases": [{"uid": c["uid"], "similarity": c["similarity"]} for c in r.cases],
                            "revisions": r.revisions, "first_pass_supported": r.first_pass_supported,
                            "prompt_versions": r.prompt_versions}
                elif system == "image_only":
                    rep = model.draft(image, [], {"callbacks": handlers, "run_name": "findxray-image-only"})
                    rec |= {"report": [s.model_dump() for s in rep.sentences],
                            "prompt_versions": dict(model.prompt_versions)}
                else:
                    raise ValueError(f"unknown system {system}")
            rec["ok"] = True
        except Exception as e:  # recorded and retried on the next run
            rec |= {"ok": False, "error": f"{type(e).__name__}: {str(e)[:3000]}"}
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        rec["trace_id"] = next((h.last_trace_id for h in handlers if getattr(h, "last_trace_id", None)), None)
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        status = "ok" if rec["ok"] else f"FAILED ({rec['error'][:80]})"
        print(f"[{i}/{len(todo)}] {system:10s} study {uid}: {status} in {rec['seconds']:.1f} s", flush=True)
        if not rec["ok"] and "PerDay" in rec["error"]:
            print("\nDaily free-tier quota reached for this model; stopping. Rerun after the quota resets "
                  "(midnight Pacific time) and the script continues where it stopped.")
            break
        time.sleep(args.pause)

    remaining = [(s, u) for u in sample for s in systems if (s, u) not in done_keys(out_path, model.name, args.config)]
    print(f"\nfinished; {len(remaining)} still missing" + (" (rerun to retry)" if remaining else ""))


if __name__ == "__main__":
    main()
