import base64

import numpy as np
import pytest
from langchain_core.runnables import RunnableLambda

from casefinder.agent import MAX_REVISIONS, Report, Sentence, Verdict, Verification, XRayImage, run_report
from casefinder.agent.graph import citation_problems
from casefinder.agent.model import GeminiReportModel, PromptSource
from casefinder.agent.prompts import LOCAL_PROMPTS
from casefinder.search import CaseLibrary

IMG = XRayImage(b"\xff\xd8\xff fake jpeg", "image/jpeg")


def library():
    emb = np.eye(6, dtype=np.float32)
    cases = [{"uid": 10 + i, "groups": ["Cardiomegaly"], "findings": f"f{i}", "impression": f"i{i}"} for i in range(6)]
    return CaseLibrary(emb, cases, model="test")


def S(text, *sources, section="findings"):
    return Sentence(section=section, text=text, sources=list(sources))


class ScriptedModel:
    """Returns scripted drafts/revisions; judges a sentence supported unless its text contains 'BAD'."""

    name = "scripted"

    def __init__(self, draft, revisions=()):
        self._draft, self._revisions = draft, list(revisions)
        self.calls = []
        self.prompt_versions = {"findxray-draft": "test"}

    def draft(self, image, cases, config=None):
        self.calls.append(("draft", image, [c["uid"] for c in cases]))
        return self._draft

    def verify(self, image, cases, report, config=None):
        self.calls.append(("verify", len(report.sentences)))
        return Verification(verdicts=[Verdict(index=i, supported="BAD" not in s.text, reason="scripted")
                                      for i, s in enumerate(report.sentences)])

    def revise(self, image, cases, report, verdicts, config=None):
        self.calls.append(("revise",))
        return self._revisions.pop(0) if self._revisions else report


def run(model, k=3):
    return run_report(library(), model, IMG, embedding=np.eye(6)[0], k=k)


def test_all_supported_needs_no_revision():
    m = ScriptedModel(Report(sentences=[S("Heart is enlarged.", "image", "case:10"),
                                        S("Cardiomegaly.", "image", section="impression")]))
    r = run(m)
    assert r.revisions == 0 and len(r.report) == 2 and r.removed == []
    assert r.first_pass_supported == 1.0
    assert [c[0] for c in m.calls] == ["draft", "verify"]
    assert m.calls[0][1] is IMG and m.calls[0][2][0] == 10    # image passed through; nearest case first
    assert [c["uid"] for c in r.cases] == [10, 11, 12] and r.cases[0]["similarity"] == pytest.approx(1.0)


def test_unsupported_sentence_is_revised_away():
    draft = Report(sentences=[S("Heart is enlarged.", "image"), S("BAD nodule in left lung.", "case:11")])
    fixed = Report(sentences=[S("Heart is enlarged.", "image")])
    m = ScriptedModel(draft, revisions=[fixed])
    r = run(m)
    assert r.revisions == 1
    assert [s.text for s in r.report] == ["Heart is enlarged."] and r.removed == []
    assert r.first_pass_supported == 0.5


def test_stops_after_max_revisions_and_reports_what_was_removed():
    stubborn = Report(sentences=[S("Heart is enlarged.", "image"), S("BAD fracture.", "image")])
    m = ScriptedModel(stubborn, revisions=[stubborn, stubborn, stubborn])
    r = run(m)
    assert r.revisions == MAX_REVISIONS
    assert [s.text for s in r.report] == ["Heart is enlarged."]
    assert [s.text for s in r.removed] == ["BAD fracture."] and not r.removed[0].supported
    assert sum(c[0] == "revise" for c in m.calls) == MAX_REVISIONS


def test_citation_outside_retrieved_cases_fails_even_if_model_approves():
    m = ScriptedModel(Report(sentences=[S("Effusion on the right.", "case:99")]),
                      revisions=[Report(sentences=[S("Effusion on the right.", "image")])])
    r = run(m)
    assert r.first_pass_supported == 0.0 and r.revisions == 1
    assert [s.sources for s in r.report] == [["image"]]


def test_citation_problems():
    rep = Report(sentences=[S("a", "image"), S("b"), S("c", "case:10", "case:12"), S("d", "Case:10 "),
                            S("e", "case:abc"), S("f", "radiopaedia")])
    assert citation_problems(rep, {10, 11, 12}).keys() == {1, 4, 5}


def test_missing_verdict_counts_as_unsupported():
    class Lazy(ScriptedModel):
        def verify(self, image, cases, report, config=None):
            return Verification(verdicts=[Verdict(index=0, supported=True, reason="ok")])  # skips index 1
    m = Lazy(Report(sentences=[S("a", "image"), S("b", "image")]))
    r = run(m)
    assert [s.text for s in r.report] == ["a"]
    assert r.removed[0].reason == "not judged by the verifier"


def test_empty_draft_is_handled():
    r = run(ScriptedModel(Report(sentences=[])))
    assert r.report == [] and r.removed == [] and r.revisions == 0


class FakeStructuredLLM:
    """Stands in for ChatGoogleGenerativeAI: records the messages, returns a fixed object."""

    def __init__(self, result):
        self.result, self.seen = result, []

    def with_structured_output(self, schema):
        def respond(messages):
            self.seen.append((schema, messages))
            return self.result
        return RunnableLambda(respond)


def test_gemini_model_builds_multimodal_messages_with_local_prompts():
    m = GeminiReportModel(api_key="test-key", prompts=PromptSource(use_langfuse=False))
    m.llm = FakeStructuredLLM(Report(sentences=[S("Clear lungs.", "image")]))
    out = m.draft(IMG, [{"uid": 10, "similarity": 0.81, "groups": ["No Finding"], "impression": "Normal."}])
    assert out.sentences[0].text == "Clear lungs."
    schema, messages = m.llm.seen[0]
    assert schema is Report
    assert messages[0].content == LOCAL_PROMPTS["findxray-draft"]
    text_block, image_block = messages[1].content
    assert "case:10 (similarity 0.81)" in text_block["text"] and "impression: Normal." in text_block["text"]
    assert image_block["type"] == "image" and image_block["mime_type"] == "image/jpeg"
    assert base64.b64decode(image_block["base64"]) == IMG.data
    assert image_block["media_resolution"] == "MEDIA_RESOLUTION_MEDIUM"
    assert m.prompt_versions == {"findxray-draft": "local"}


def test_gemini_request_carries_latency_settings():
    """Checks the request LangChain would send to Gemini (no network): thinking level, image resolution, timeout."""
    from langchain_core.messages import HumanMessage
    m = GeminiReportModel(api_key="test-key", prompts=PromptSource(use_langfuse=False))
    block = {"type": "image", "base64": base64.b64encode(IMG.data).decode(), "mime_type": "image/jpeg",
             "media_resolution": m.media_resolution}
    req = m.llm._prepare_request([HumanMessage(content=[{"type": "text", "text": "x"}, block])])
    image_part = req["contents"][-1].parts[-1]
    assert image_part.inline_data.data == IMG.data
    assert image_part.media_resolution.level.value == "MEDIA_RESOLUTION_MEDIUM"
    assert req["config"].thinking_config.thinking_level.value == "LOW"
    assert req["config"].http_options.timeout == 180_000
    assert req["config"].http_options.retry_options.attempts == 4   # total attempts, i.e. 3 retries


def test_mask_removes_image_bytes_from_traces():
    from casefinder.agent.tracing import mask_images
    b64 = base64.b64encode(bytes(range(256)) * 40).decode()
    data = {"messages": [[{"content": [{"type": "text", "text": "hello"},
                                       {"type": "image", "base64": b64, "mime_type": "image/jpeg"}]}]],
            "serialized": f"prefix {b64} suffix", "short": "abc"}
    out = mask_images(data=data)
    flat = str(out)
    assert b64 not in flat and flat.count("image omitted from trace") == 2
    assert out["messages"][0][0]["content"][0] == {"type": "text", "text": "hello"}
    assert out["messages"][0][0]["content"][1]["mime_type"] == "image/jpeg"
    assert out["serialized"].startswith("prefix <image omitted") and out["short"] == "abc"


def test_gemini_model_requires_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiReportModel(prompts=PromptSource(use_langfuse=False))
