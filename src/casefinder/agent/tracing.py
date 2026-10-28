"""Langfuse tracing. Disabled (no-op) unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL from the environment.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager, nullcontext
from typing import Any

from casefinder.agent.model import langfuse_configured

_B64_RUN = re.compile(r"[A-Za-z0-9+/]{2000,}={0,2}")
_client = None


def _placeholder(b64: str) -> str:
    return f"<image omitted from trace: ~{len(b64) * 3 // 4} bytes, sha256:{hashlib.sha256(b64.encode()).hexdigest()[:12]}>"


def mask_images(*, data: Any, **_: Any) -> Any:
    """Langfuse mask: X-ray bytes never leave the machine in traces (privacy, and ~300 KB per call).

    Replaces the base64 payload of LangChain image blocks, and any other long base64 run inside
    strings, with a short placeholder carrying the size and a hash prefix.
    """
    if isinstance(data, dict):
        if data.get("type") == "image" and isinstance(data.get("base64"), str):
            return {**{k: mask_images(data=v) for k, v in data.items() if k != "base64"},
                    "base64": _placeholder(data["base64"])}
        return {k: mask_images(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_images(data=v) for v in data]
    if isinstance(data, str) and len(data) >= 2000:
        return _B64_RUN.sub(lambda m: _placeholder(m.group()), data)
    return data


def langfuse_client():
    """The process-wide Langfuse client, created once with the image mask."""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(mask=mask_images)
    return _client


def callback_handlers() -> list:
    if not langfuse_configured():
        return []
    langfuse_client()  # make sure the masked client exists before the handler looks it up
    from langfuse.langchain import CallbackHandler
    return [CallbackHandler()]


def trace_attributes(name: str = "findxray-report", tags: list[str] | None = None, **metadata):
    """Names the trace and attaches tags/metadata (e.g. model, k) when Langfuse is on."""
    if not langfuse_configured():
        return nullcontext()
    from langfuse import propagate_attributes
    return propagate_attributes(trace_name=name, tags=["findxray", *(tags or [])],
                                metadata={k: str(v) for k, v in metadata.items()})


@contextmanager
def flushing():
    """Make sure buffered traces are sent before a short-lived script exits."""
    try:
        yield
    finally:
        if langfuse_configured():
            langfuse_client().flush()


def trace_url(handlers: list) -> str | None:
    """Link to the trace the handler just recorded (CallbackHandler keeps its last trace id)."""
    trace_id = next((h.last_trace_id for h in handlers if getattr(h, "last_trace_id", None)), None)
    if not trace_id:
        return None
    try:
        # Needs a call to Langfuse (project lookup); never let an unreachable Langfuse break a finished report.
        return langfuse_client().get_trace_url(trace_id=trace_id)
    except Exception:
        return f"(trace id {trace_id}; Langfuse unreachable, link unavailable)"
