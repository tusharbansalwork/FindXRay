"""HTTP API for similar-case search.

The browser embeds the X-ray locally (ONNX Runtime Web) and sends only the
768-number embedding here; no image leaves the visitor's device for search.

Run locally:  uvicorn casefinder.api:app --reload --app-dir src
Environment:
  CASEFINDER_LIBRARY   path to artifacts/library (default: <repo>/artifacts/library)
  CASEFINDER_ORIGINS   comma-separated allowed browser origins (default: http://localhost:8000)
  CASEFINDER_IMAGE_URL image URL template with {uid} (default: the public Hugging Face case library)

Case images must be loaded with <img crossorigin="anonymous">: the app page is cross-origin
isolated (needed for multi-threaded WASM), and Hugging Face sends CORS headers but no
Cross-Origin-Resource-Policy, so a plain <img> is blocked by the browser.
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from casefinder.search import CaseLibrary

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "artifacts" / "library"
MAX_K = 20
DEFAULT_IMAGE_URL = "https://huggingface.co/datasets/tush-2ar/findxray-case-library/resolve/main/images/{uid}.jpg"


class SearchRequest(BaseModel):
    embedding: list[float] = Field(..., min_length=1, max_length=4096)
    k: int = Field(5, ge=1, le=MAX_K)

    @field_validator("embedding")
    @classmethod
    def finite(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("embedding contains NaN or infinite values")
        return v


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


def create_app(library: CaseLibrary | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.library = library or CaseLibrary.load(os.environ.get("CASEFINDER_LIBRARY", DEFAULT_LIBRARY))
        yield

    app = FastAPI(title="FindXRay", version="0.1.0", lifespan=lifespan)
    image_url = os.environ.get("CASEFINDER_IMAGE_URL", DEFAULT_IMAGE_URL)
    if "{uid}" not in image_url:
        raise ValueError("CASEFINDER_IMAGE_URL must contain {uid}")
    origins = [o.strip() for o in os.environ.get("CASEFINDER_ORIGINS", "http://localhost:8000").split(",") if o.strip()]
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
        return {"status": "ok", "model": lib.model, "library_size": len(lib), "dim": lib.dim}

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

    return app


app = create_app()
