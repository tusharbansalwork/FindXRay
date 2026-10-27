"""Find which part of the draft call makes Gemini slow or fail.

Runs four calls, adding one ingredient at a time, each with a 90 s deadline and no retries:
  A  text only                       "reply OK"
  B  text + X-ray image              one-sentence description, free text
  C  structured output, no image     the real draft prompt and cases, Report schema
  D  structured output + image       exactly the agent's draft call
The first step that times out or slows down sharply points at the cause.

Usage:
  python scripts/diagnose_gemini.py                         # model/settings from .env
  python scripts/diagnose_gemini.py --model <name>          # try another model
  python scripts/diagnose_gemini.py --thinking minimal      # try another thinking level
  python scripts/diagnose_gemini.py --quick --model a,b,c   # step A only (30 s limit) for several models
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from casefinder.agent.prompts import DRAFT, format_cases  # noqa: E402
from casefinder.agent.schemas import Report  # noqa: E402
from casefinder.search import CaseLibrary  # noqa: E402
from draft_report import load_study  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"))
    ap.add_argument("--thinking", default=os.environ.get("GEMINI_THINKING_LEVEL", "low"))
    ap.add_argument("--resolution", default=os.environ.get("GEMINI_MEDIA_RESOLUTION", "MEDIA_RESOLUTION_MEDIUM"))
    ap.add_argument("--uid", type=int, default=1212)
    ap.add_argument("--quick", action="store_true", help="only step A, 30 s limit; --model may be a comma list")
    args = ap.parse_args()

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    if args.quick:
        for model in [m.strip() for m in args.model.split(",") if m.strip()]:
            quick = ChatGoogleGenerativeAI(model=model, google_api_key=os.environ.get("GEMINI_API_KEY"),
                                           max_retries=1, timeout=30)
            t0 = time.perf_counter()
            try:
                out = quick.invoke([HumanMessage(content="Reply with the single word OK.")])
                status = f"OK    {str(out.content)[:40]!r}"
            except Exception as e:
                status = f"FAIL  {type(e).__name__}: {str(e)[:90]}"
            print(f"{model:28s} {time.perf_counter() - t0:5.1f} s  {status}", flush=True)
        return

    llm = ChatGoogleGenerativeAI(model=args.model, google_api_key=os.environ.get("GEMINI_API_KEY"),
                                 thinking_level=args.thinking, max_retries=1, timeout=90)
    study = load_study(args.uid)
    data = study["img_frontal"]
    image = {"type": "image", "base64": base64.b64encode(data).decode(),
             "mime_type": "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png",
             "media_resolution": args.resolution}
    lib = CaseLibrary.load(ROOT / "artifacts" / "library")
    uids = list(np.load(ROOT / "artifacts" / "embeddings" / "uids.npy"))
    emb = np.load(ROOT / "artifacts" / "embeddings" / "rad-dino-224.npy")[uids.index(args.uid)]
    cases = [{**h.case, "similarity": h.similarity} for h in lib.search(emb, k=5)]
    case_text = f"Similar past cases:\n{format_cases(cases)}\n\nDraft the report for the attached X-ray."

    steps = [
        ("A text only", llm, [HumanMessage(content="Reply with the single word OK.")]),
        ("B text + image", llm, [HumanMessage(content=[
            {"type": "text", "text": "Describe this chest X-ray in one sentence."}, image])]),
        ("C schema, no image", llm.with_structured_output(Report), [
            SystemMessage(content=DRAFT), HumanMessage(content=case_text + "\n(No image attached for this test.)")]),
        ("D schema + image", llm.with_structured_output(Report), [
            SystemMessage(content=DRAFT), HumanMessage(content=[{"type": "text", "text": case_text}, image])]),
    ]
    print(f"model {args.model} | thinking {args.thinking} | image {args.resolution} | "
          f"study {args.uid} ({len(data) / 1024:.0f} KB)\n")
    for name, runnable, messages in steps:
        t0 = time.perf_counter()
        try:
            out = runnable.invoke(messages)
            preview = (out.content if hasattr(out, "content") else
                       f"{len(out.sentences)} sentences: {out.sentences[0].text if out.sentences else ''}")
            status = f"OK    {str(preview)[:90]!r}"
        except Exception as e:  # report and continue to the next step
            status = f"FAIL  {type(e).__name__}: {str(e)[:110]}"
        print(f"{name:20s} {time.perf_counter() - t0:6.1f} s  {status}", flush=True)


if __name__ == "__main__":
    main()
