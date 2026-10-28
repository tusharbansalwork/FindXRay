"""HTTP API for similar-case search.

The browser embeds the X-ray locally (ONNX Runtime Web) and sends only the
768-number embedding here; no image leaves the visitor's device for search.

Run locally:  uvicorn casefinder.api:app --reload --app-dir src
Environment:
  CASEFINDER_LIBRARY   path to artifacts/library (default: <repo>/artifacts/library)
  CASEFINDER_ORIGINS   comma-separated allowed browser origins (default: the local preview, http://localhost:8080)
  CASEFINDER_IMAGE_URL image URL template with {uid} (default: the public Hugging Face case library)
  CASEFINDER_LIBRARY_REPO  if the library folder is missing, download it from this Hugging Face dataset
                       (folder "library/"), e.g. tush-2ar/findxray-case-library
  CASEFINDER_DAILY_REPORTS live reports allowed per day, counted per server process (default 60; 0 = no cap).
                       The day resets at midnight Pacific time, when Gemini's free-tier daily quota resets.
  GEMINI_API_KEY       enables POST /report and /report/stream (draft report agent); GEMINI_MODEL picks the model
  LANGFUSE_*           enables tracing of report runs (see casefinder.agent.tracing)

Case images must be loaded with <img crossorigin="anonymous">: the app page is cross-origin
isolated (needed for multi-threaded WASM), and Hugging Face sends CORS headers but no
Cross-Origin-Resource-Policy, so a plain <img> is blocked by the browser.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from casefinder.agent import ReportResult, XRayImage, run_report, stream_report
from casefinder.agent.tracing import callback_handlers, trace_attributes, trace_url
from casefinder.search import CaseLibrary

log = logging.getLogger("findxray")

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "artifacts" / "library"
MAX_K = 20
DEFAULT_IMAGE_URL = "https://huggingface.co/datasets/tush-2ar/findxray-case-library/resolve/main/images/{uid}.jpg"


QUOTA_TZ = ZoneInfo("America/Los_Angeles")
CAP_MESSAGE = "Live report drafting is paused for today (daily limit reached). Try one of the sample X-rays."
QUOTA_MESSAGE = "The AI model's free daily quota is used up. Try one of the sample X-rays."

MAX_IMAGE_BYTES = 8 * 2**20
IMAGE_SIGNATURES = {"image/jpeg": b"\xff\xd8\xff", "image/png": b"\x89PNG\r\n\x1a\n"}


class SearchRequest(BaseModel):
    embedding: list[float] = Field(..., min_length=1, max_length=4096)
    k: int = Field(5, ge=1, le=MAX_K)

    @field_validator("embedding")
    @classmethod
    def finite(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("embedding contains NaN or infinite values")
        return v


class ReportRequest(SearchRequest):
    image_base64: str = Field(..., description="The X-ray file, base64-encoded (JPEG or PNG, max 8 MB).")
    mime_type: str = Field(..., pattern="^image/(jpeg|png)$")

    def decoded_image(self) -> XRayImage:
        if len(self.image_base64) > MAX_IMAGE_BYTES * 4 // 3 + 4:
            raise ValueError("image is larger than 8 MB")
        try:
            data = base64.b64decode(self.image_base64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError("image_base64 is not valid base64") from e
        if not data.startswith(IMAGE_SIGNATURES[self.mime_type]):
            raise ValueError(f"file content is not {self.mime_type}")
        return XRayImage(data=data, mime_type=self.mime_type)


class DailyCap:
    """Counts live reports per Pacific-time day (in this process). limit 0 = unlimited."""

    def __init__(self, limit: int, clock=lambda: datetime.now(QUOTA_TZ).date()):
        self.limit, self.clock = limit, clock
        self.day, self.used = clock(), 0
        self.lock = threading.Lock()

    def _roll(self):
        today = self.clock()
        if today != self.day:
            self.day, self.used = today, 0

    def take(self) -> bool:
        with self.lock:
            self._roll()
            if self.limit and self.used >= self.limit:
                return False
            self.used += 1
            return True

    def remaining(self) -> int | None:
        with self.lock:
            self._roll()
            return None if not self.limit else max(0, self.limit - self.used)


def is_quota_error(e: BaseException) -> bool:
    text = f"{type(e).__name__} {e}"
    return "RESOURCE_EXHAUSTED" in text or "PerDay" in text or ("429" in text and "quota" in text.lower())


def ensure_library(path: Path) -> Path:
    """Download the case library from Hugging Face if it is not on disk (used on the deployed server)."""
    if (path / "embeddings.npy").exists():
        return path
    repo = os.environ.get("CASEFINDER_LIBRARY_REPO")
    if not repo:
        raise FileNotFoundError(f"no case library at {path}; build it (scripts/build_library.py) "
                                "or set CASEFINDER_LIBRARY_REPO")
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=["library/*"],
                             local_dir=path.parent / "_hf_library")
    return Path(root) / "library"


class CaseOut(BaseModel):
    rank: int
    similarity: float
    uid: int
    groups: list[str]
    indication: str | None
    findings: str | None
    impression: str | None
    image_url: str


class SearchResponse(BaseModel):
    model: str
    library_size: int
    results: list[CaseOut]


def create_app(library: CaseLibrary | None = None, report_model=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.library = library or CaseLibrary.load(
            ensure_library(Path(os.environ.get("CASEFINDER_LIBRARY", DEFAULT_LIBRARY))))
        app.state.report_model = report_model
        app.state.cap = DailyCap(int(os.environ.get("CASEFINDER_DAILY_REPORTS", "60")))
        yield

    app = FastAPI(title="FindXRay", version="0.1.0", lifespan=lifespan)
    image_url = os.environ.get("CASEFINDER_IMAGE_URL", DEFAULT_IMAGE_URL)
    if "{uid}" not in image_url:
        raise ValueError("CASEFINDER_IMAGE_URL must contain {uid}")
    origins = [o.strip() for o in os.environ.get("CASEFINDER_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 echoes the offending input; that crashes on NaN (not valid JSON)
        # and would send a 768-number list back. Return location + message only.
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "invalid")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health")
    def health():
        lib: CaseLibrary = app.state.library
        return {"status": "ok", "model": lib.model, "library_size": len(lib), "dim": lib.dim,
                "reports_enabled": app.state.report_model is not None
                or bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
                "reports_left_today": app.state.cap.remaining()}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        lib: CaseLibrary = app.state.library
        if len(req.embedding) != lib.dim:
            raise HTTPException(422, f"embedding must have {lib.dim} values, got {len(req.embedding)}")
        try:
            hits = lib.search(np.asarray(req.embedding, dtype=np.float32), k=req.k)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return SearchResponse(
            model=lib.model,
            library_size=len(lib),
            results=[
                CaseOut(rank=h.rank, similarity=h.similarity, uid=h.case["uid"], groups=h.case["groups"],
                        indication=h.case.get("indication"), findings=h.case.get("findings"),
                        impression=h.case.get("impression"),
                        image_url=image_url.format(uid=h.case["uid"]))
                for h in hits
            ],
        )

    def get_report_model():
        if app.state.report_model is None:
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                raise HTTPException(503, "report drafting is not configured on this server")
            from casefinder.agent import GeminiReportModel
            app.state.report_model = GeminiReportModel()
        return app.state.report_model

    def prepare_report(req: ReportRequest):
        """Validation, model and daily cap shared by both report endpoints."""
        lib: CaseLibrary = app.state.library
        if len(req.embedding) != lib.dim:
            raise HTTPException(422, f"embedding must have {lib.dim} values, got {len(req.embedding)}")
        try:
            image = req.decoded_image()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        model = get_report_model()
        if not app.state.cap.take():
            raise HTTPException(429, CAP_MESSAGE)
        return lib, model, image

    @app.post("/report", response_model=ReportResult)
    def report(req: ReportRequest):
        lib, model, image = prepare_report(req)
        handlers = callback_handlers()
        try:
            with trace_attributes(model=model.name, k=req.k, library_size=len(lib)):
                result = run_report(lib, model, image, np.asarray(req.embedding, dtype=np.float32), k=req.k,
                                    callbacks=handlers)
        except Exception as e:  # model/network failures: don't leak internals to the browser
            log.exception("report generation failed")
            if is_quota_error(e):
                raise HTTPException(429, QUOTA_MESSAGE) from e
            raise HTTPException(502, f"the report model failed: {type(e).__name__}") from e
        url = trace_url(handlers)
        if url:
            log.info("report trace: %s", url)
        return result

    @app.post("/report/stream")
    def report_stream(req: ReportRequest):
        """Same as /report, but streams progress as newline-delimited JSON:
        {"event": "step", "step": "draft"} ... then {"event": "result", "result": {...}}
        or {"event": "error", "status": 429|502, "message": "..."}."""
        lib, model, image = prepare_report(req)
        events: queue.Queue = queue.Queue()
        emb = np.asarray(req.embedding, dtype=np.float32)

        def work():
            # One thread runs the whole graph, so the tracing context stays in place for every step.
            handlers = callback_handlers()
            try:
                with trace_attributes(model=model.name, k=req.k, library_size=len(lib)):
                    for item in stream_report(lib, model, image, emb, k=req.k, callbacks=handlers):
                        if isinstance(item, str):
                            events.put({"event": "step", "step": item})
                        else:
                            events.put({"event": "result", "result": item.model_dump(mode="json")})
            except Exception as e:
                log.exception("report generation failed")
                status, msg = (429, QUOTA_MESSAGE) if is_quota_error(e) else (502, f"the report model failed: {type(e).__name__}")
                events.put({"event": "error", "status": status, "message": msg})
            finally:
                url = trace_url(handlers)
                if url:
                    log.info("report trace: %s", url)
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def lines():
            yield json.dumps({"event": "started", "model": model.name}) + "\n"
            while (ev := events.get()) is not None:
                yield json.dumps(ev) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return app


app = create_app()
