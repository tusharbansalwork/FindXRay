"""The language-model side of the agent: prompt lookup (Langfuse or local) and Gemini calls.

Everything the graph needs from a model is the three methods of `ReportModel`, so tests can
substitute a deterministic stand-in and never touch the network.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from casefinder.agent.prompts import LOCAL_PROMPTS, format_cases, format_report
from casefinder.agent.schemas import Report, Verdict, Verification

# 2026-10-28: gemini-2.5-flash is closed to new API projects (404); gemini-3.8-flash, 3.7-flash and
# gemini-flash-latest timed out even on "reply OK"; gemini-3.6-flash answered in ~3 s (scripts/diagnose_gemini.py).
DEFAULT_MODEL = "gemini-3.6-flash"


@dataclass(frozen=True)
class XRayImage:
    data: bytes
    mime_type: str  # image/jpeg or image/png

    def data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode()}"


class ReportModel(Protocol):
    name: str
    prompt_versions: dict[str, str]

    def draft(self, image: XRayImage, cases: list[dict], config: RunnableConfig | None = None) -> Report: ...
    def verify(self, image: XRayImage, cases: list[dict], report: Report,
               config: RunnableConfig | None = None) -> Verification: ...
    def revise(self, image: XRayImage, cases: list[dict], report: Report, verdicts: list[Verdict],
               config: RunnableConfig | None = None) -> Report: ...


@dataclass
class _LocalPrompt:
    prompt: str
    version: str = "local"


def langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


class PromptSource:
    """Fetches the 'production' version of each prompt from Langfuse, falling back to the local text."""

    def __init__(self, use_langfuse: bool | None = None):
        self.use_langfuse = langfuse_configured() if use_langfuse is None else use_langfuse
        self._client = None
        if self.use_langfuse:
            from casefinder.agent.tracing import langfuse_client
            self._client = langfuse_client()

    def get(self, name: str):
        if not self._client:
            return _LocalPrompt(LOCAL_PROMPTS[name])
        return self._client.get_prompt(name, label="production", type="text", fallback=LOCAL_PROMPTS[name],
                                       cache_ttl_seconds=300)

    @staticmethod
    def version_of(prompt) -> str:
        if isinstance(prompt, _LocalPrompt):
            return "local"
        return "local-fallback" if getattr(prompt, "is_fallback", False) else str(prompt.version)


class GeminiReportModel:
    def __init__(self, model: str | None = None, api_key: str | None = None, prompts: PromptSource | None = None,
                 temperature: float | None = None):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.name = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini report model")
        # Latency controls. With defaults (dynamic thinking, full-resolution image) a single call to
        # gemini-3.8-flash exceeded a 90 s deadline on a 2048x2048 X-ray. All overridable from .env.
        self.thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", "low")            # minimal|low|medium|high
        self.media_resolution = os.environ.get("GEMINI_MEDIA_RESOLUTION", "MEDIA_RESOLUTION_MEDIUM")
        # Gemini 3.x Flash models use fixed sampling defaults and ignore temperature (with a warning), so it
        # is only sent when explicitly requested.
        extra = {} if temperature is None else {"temperature": temperature}
        self.llm = ChatGoogleGenerativeAI(model=self.name, google_api_key=key, **extra,
                                          thinking_level=self.thinking_level,
                                          # LangChain's max_retries is the TOTAL number of attempts (1 = no retry).
                                          # google-genai backs off exponentially on 429/5xx, e.g. 503 "high demand".
                                          max_retries=int(os.environ.get("GEMINI_MAX_ATTEMPTS", "4")),
                                          timeout=float(os.environ.get("GEMINI_TIMEOUT", "180")))
        self.prompts = prompts or PromptSource()
        self.prompt_versions: dict[str, str] = {}

    def _call(self, prompt_name: str, schema: type, user_text: str, image: XRayImage,
              config: RunnableConfig | None) -> Any:
        prompt = self.prompts.get(prompt_name)
        self.prompt_versions[prompt_name] = PromptSource.version_of(prompt)

        def build_messages(_: Any) -> list:
            return [
                SystemMessage(content=prompt.prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_text},
                    # LangChain's v1 image block: the only form that passes media_resolution to Gemini.
                    {"type": "image", "base64": base64.b64encode(image.data).decode(), "mime_type": image.mime_type,
                     "media_resolution": self.media_resolution},
                ]),
            ]

        # Metadata on the message-building step links the generation to the Langfuse prompt version.
        meta = {} if isinstance(prompt, _LocalPrompt) else {"langfuse_prompt": prompt}
        chain = (RunnableLambda(build_messages).with_config(run_name=f"{prompt_name}-messages", metadata=meta)
                 | self.llm.with_structured_output(schema))
        result = chain.with_config(run_name=prompt_name).invoke({}, config=config)
        if result is None:
            raise RuntimeError(f"{self.name} returned no parseable {schema.__name__} for {prompt_name}")
        return result

    def draft(self, image, cases, config=None) -> Report:
        text = f"Similar past cases:\n{format_cases(cases)}\n\nDraft the report for the attached X-ray."
        return self._call("findxray-draft", Report, text, image, config)

    def verify(self, image, cases, report, config=None) -> Verification:
        text = (f"Similar past cases:\n{format_cases(cases)}\n\nDraft sentences:\n{format_report(report.sentences)}"
                "\n\nJudge every sentence.")
        return self._call("findxray-verify", Verification, text, image, config)

    def revise(self, image, cases, report, verdicts, config=None) -> Report:
        verdict_text = "\n".join(f"[{v.index}] {'supported' if v.supported else 'UNSUPPORTED'}: {v.reason}"
                                 for v in verdicts)
        text = (f"Similar past cases:\n{format_cases(cases)}\n\nDraft sentences:\n{format_report(report.sentences)}"
                f"\n\nVerdicts:\n{verdict_text}\n\nReturn the revised report.")
        return self._call("findxray-revise", Report, text, image, config)
