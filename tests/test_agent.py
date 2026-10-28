import base64

import numpy as np
import pytest
from langchain_core.runnables import RunnableLambda

from casefinder.agent import MAX_REVISIONS, Report, Sentence, Verdict, Verification, XRayImage, run_report
from casefinder.agent.schemas import CheckItem, LookAgain
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

    missed: list = []          # findings the scripted second reader reports as missing
    amended: Report | None = None

    def look_again(self, image, cases, report, config=None):
        self.calls.append(("look_again",))
        return LookAgain(items=list(self.missed))

    def amend(self, image, cases, report, missed, config=None):
        self.calls.append(("amend", [m.finding for m in missed]))
        return self.amended or report


def run(model, k=3):
    return run_report(library(), model, IMG, embedding=np.eye(6)[0], k=k)


def test_all_supported_needs_no_revision():
    m = ScriptedModel(Report(sentences=[S("Heart is enlarged.", "image", "case:10"),
                                        S("Cardiomegaly.", "image", section="impression")]))
    r = run(m)
    assert r.revisions == 0 and len(r.report) == 2 and r.removed == []
    assert r.first_pass_supported == 1.0
    assert [c[0] for c in m.calls] == ["draft", "look_again", "verify"]
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
    assert image_block["media_resolution"] == "MEDIA_RESOLUTION_HIGH"
    assert "Checklist:" in text_block["text"] and "Pneumothorax" in text_block["text"]
    assert m.prompt_versions == {"findxray-draft": "local"}


def test_gemini_request_carries_latency_settings():
    """Checks the request LangChain would send to Gemini (no network): thinking level, image resolution, timeout."""
    from langchain_core.messages import HumanMessage
    m = GeminiReportModel(api_key="test-key", prompts=PromptSource(use_langfuse=False))
    block = {"type": "image", "base64": base64.b64encode(IMG.data).decode(), "mime_type": "image/jpeg",
             "media_resolution": m.media_resolution}
    assert m.name == "gemini-3.5-flash-lite"
    req = m.llm._prepare_request([HumanMessage(content=[{"type": "text", "text": "x"}, block])])
    image_part = req["contents"][-1].parts[-1]
    assert image_part.inline_data.data == IMG.data
    assert image_part.media_resolution.level.value == "MEDIA_RESOLUTION_HIGH"
    assert req["config"].thinking_config.thinking_level.value == "MEDIUM"
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


def test_first_draft_is_kept_for_evaluation():
    draft = Report(sentences=[S("Heart is enlarged.", "image"), S("BAD nodule.", "image")])
    m = ScriptedModel(draft, revisions=[Report(sentences=[S("Heart is enlarged.", "image")])])
    r = run(m)
    assert [s.text for s in r.first_draft] == ["Heart is enlarged.", "BAD nodule."]
    assert [s.text for s in r.report] == ["Heart is enlarged."]


def test_image_only_draft_prompt_has_no_cases():
    m = GeminiReportModel(api_key="test-key", prompts=PromptSource(use_langfuse=False))
    m.llm = FakeStructuredLLM(Report(sentences=[S("Clear lungs.", "image")]))
    m.draft(IMG, [])
    text = m.llm.seen[0][1][1].content[0]["text"]
    assert "No similar cases are provided" in text and "case:" not in text


def test_rate_limiter_paces_calls(monkeypatch):
    monkeypatch.setenv("GEMINI_RPM", "6")
    m = GeminiReportModel(api_key="test-key", prompts=PromptSource(use_langfuse=False))
    assert m.llm.rate_limiter is m.rate_limiter
    assert m.rate_limiter.requests_per_second == pytest.approx(0.1)
    monkeypatch.setenv("GEMINI_RPM", "0")
    assert GeminiReportModel(api_key="k", prompts=PromptSource(use_langfuse=False)).llm.rate_limiter is None


def item(finding, present, in_draft):
    return CheckItem(finding=finding, present=present, in_draft=in_draft, evidence="seen")


def test_look_again_adds_missed_findings_via_amend():
    m = ScriptedModel(Report(sentences=[S("Lungs are clear.", "image")]))
    m.missed = [item("Pleural Effusion", "yes", False), item("Cardiomegaly", "no", True),
                item("Pneumothorax", "uncertain", False), item("Atelectasis", "yes", True)]
    m.amended = Report(sentences=[S("Small right pleural effusion.", "image"), S("Possible apical pneumothorax.", "image")])
    r = run(m)
    assert ("amend", ["Pleural Effusion", "Pneumothorax"]) in m.calls          # only present/uncertain AND not in draft
    assert [c[0] for c in m.calls][:4] == ["draft", "look_again", "amend", "verify"]
    assert [s.text for s in r.report] == ["Small right pleural effusion.", "Possible apical pneumothorax."]
    assert [s.text for s in r.first_draft] == ["Lungs are clear."]
    assert [f.finding for f in r.missed_findings] == ["Pleural Effusion", "Pneumothorax"]


def test_look_again_without_misses_skips_amend():
    m = ScriptedModel(Report(sentences=[S("Lungs are clear.", "image")]))
    m.missed = [item("Pleural Effusion", "no", True)]
    r = run(m)
    assert [c[0] for c in m.calls] == ["draft", "look_again", "verify"] and r.missed_findings == []


def test_look_again_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FINDXRAY_LOOK_AGAIN", "0")
    m = ScriptedModel(Report(sentences=[S("Lungs are clear.", "image")]))
    run(m)
    assert [c[0] for c in m.calls] == ["draft", "verify"]


def test_checklist_lists_neighbour_findings_with_their_cases():
    from casefinder.agent.prompts import ALWAYS_CHECK, checklist
    items = dict(checklist([{"uid": 1, "groups": ["Cardiomegaly", "Aorta Abnormal"]},
                            {"uid": 2, "groups": ["No Finding"]}, {"uid": 3, "groups": ["Cardiomegaly"]}]))
    assert items["Cardiomegaly"] == [1, 3] and items["Aorta Abnormal"] == [1] and items["Pneumothorax"] == []
    assert "No Finding" not in items and set(ALWAYS_CHECK) <= set(items)
