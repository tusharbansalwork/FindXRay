import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from casefinder.agent import Report, Sentence, Verdict, Verification
from casefinder.api import create_app
from casefinder.search import CaseLibrary

D = 6
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class OneShotModel:
    name = "one-shot"
    prompt_versions = {"findxray-draft": "test"}

    def __init__(self, fail=False):
        self.fail, self.images = fail, []

    def draft(self, image, cases, config=None):
        if self.fail:
            raise ConnectionError("quota exceeded, secret details here")
        self.images.append(image)
        return Report(sentences=[Sentence(section="findings", text="Heart size normal.",
                                          sources=["image", f"case:{cases[0]['uid']}"])])

    def verify(self, image, cases, report, config=None):
        return Verification(verdicts=[Verdict(index=0, supported=True, reason="visible")])

    def revise(self, image, cases, report, verdicts, config=None):
        return report

    def look_again(self, image, cases, report, config=None):
        from casefinder.agent.schemas import LookAgain
        return LookAgain(items=[])

    def amend(self, image, cases, report, missed, config=None):
        return report


def client_with(model):
    emb = np.eye(D, dtype=np.float32)
    lib = CaseLibrary(emb, [{"uid": 50 + i, "groups": ["No Finding"], "impression": "Normal."} for i in range(D)])
    return TestClient(create_app(lib, report_model=model)), emb


def body(emb, data=JPEG, mime="image/jpeg", **kw):
    return {"embedding": emb[1].tolist(), "image_base64": base64.b64encode(data).decode(), "mime_type": mime, **kw}


def test_report_happy_path():
    model = OneShotModel()
    c, emb = client_with(model)
    with c:
        r = c.post("/report", json=body(emb, k=3))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["report"][0]["text"] == "Heart size normal." and out["report"][0]["sources"] == ["image", "case:51"]
    assert out["cases"][0]["uid"] == 51
    assert len(out["cases"]) == 3 and out["revisions"] == 0 and out["model"] == "one-shot"
    assert model.images[0].data == JPEG and model.images[0].mime_type == "image/jpeg"


def test_png_accepted():
    c, emb = client_with(OneShotModel())
    with c:
        assert c.post("/report", json=body(emb, data=PNG, mime="image/png")).status_code == 200


@pytest.mark.parametrize("kwargs, fragment", [
    ({"data": PNG, "mime": "image/jpeg"}, "not image/jpeg"),       # content doesn't match declared type
    ({"mime": "image/gif"}, "mime_type"),                          # unsupported type
])
def test_bad_images_rejected(kwargs, fragment):
    c, emb = client_with(OneShotModel())
    with c:
        r = c.post("/report", json=body(emb, **kwargs))
    assert r.status_code == 422 and fragment in r.text


def test_invalid_base64_rejected():
    c, emb = client_with(OneShotModel())
    with c:
        r = c.post("/report", json={**body(emb), "image_base64": "not base64!!"})
    assert r.status_code == 422 and "base64" in r.text


def test_oversized_image_rejected():
    c, emb = client_with(OneShotModel())
    big = JPEG + b"\x00" * (8 * 2**20)
    with c:
        r = c.post("/report", json=body(emb, data=big))
    assert r.status_code == 422 and "8 MB" in r.text


def test_model_failure_is_502_without_internal_details():
    c, emb = client_with(OneShotModel(fail=True))
    with c:
        r = c.post("/report", json=body(emb))
    assert r.status_code == 502
    assert "ConnectionError" in r.text and "secret details" not in r.text


def test_report_not_configured_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    c, emb = client_with(None)
    with c:
        r = c.post("/report", json=body(emb))
    assert r.status_code == 503


# --- daily cap, quota errors, streaming -------------------------------------------------------

import json
from datetime import date

from casefinder.api import DailyCap, ensure_library, is_quota_error


class QuotaModel(OneShotModel):
    def draft(self, image, cases, config=None):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier")


def test_daily_cap_blocks_after_limit(monkeypatch):
    monkeypatch.setenv("CASEFINDER_DAILY_REPORTS", "2")
    c, emb = client_with(OneShotModel())
    with c:
        codes = [c.post("/report", json=body(emb)).status_code for _ in range(3)]
        health = c.get("/health").json()
    assert codes == [200, 200, 429]
    assert health["reports_left_today"] == 0


def test_daily_cap_message_points_to_samples(monkeypatch):
    monkeypatch.setenv("CASEFINDER_DAILY_REPORTS", "1")
    c, emb = client_with(OneShotModel())
    with c:
        c.post("/report", json=body(emb))
        r = c.post("/report/stream", json=body(emb))
    assert r.status_code == 429 and "sample" in r.json()["detail"]


def test_invalid_requests_do_not_use_the_cap(monkeypatch):
    monkeypatch.setenv("CASEFINDER_DAILY_REPORTS", "1")
    c, emb = client_with(OneShotModel())
    with c:
        assert c.post("/report", json=body(emb, data=b"not an image")).status_code == 422
        assert c.post("/report", json=body(emb)).status_code == 200


def test_daily_cap_resets_on_a_new_day():
    today = [date(2026, 10, 28)]
    cap = DailyCap(1, clock=lambda: today[0])
    assert cap.take() and not cap.take()
    today[0] = date(2026, 10, 29)
    assert cap.remaining() == 1 and cap.take()


def test_zero_means_no_cap():
    cap = DailyCap(0)
    assert all(cap.take() for _ in range(500)) and cap.remaining() is None


def test_quota_error_is_429_with_friendly_message():
    c, emb = client_with(QuotaModel())
    with c:
        r = c.post("/report", json=body(emb))
    assert r.status_code == 429 and "quota" in r.json()["detail"] and "PerDay" not in r.json()["detail"]
    assert is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED")) and not is_quota_error(ConnectionError("reset"))


def read_stream(r):
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def test_stream_reports_steps_then_result():
    c, emb = client_with(OneShotModel())
    with c:
        r = c.post("/report/stream", json=body(emb, k=3))
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = read_stream(r)
    assert events[0] == {"event": "started", "model": "one-shot"}
    steps = [e["step"] for e in events if e["event"] == "step"]
    assert steps == ["retrieve", "draft", "look_again", "check"]
    result = events[-1]
    assert result["event"] == "result" and result["result"]["report"][0]["text"] == "Heart size normal."
    assert len(result["result"]["cases"]) == 3


def test_stream_error_event_hides_details():
    c, emb = client_with(OneShotModel(fail=True))
    with c:
        events = read_stream(c.post("/report/stream", json=body(emb)))
    assert events[-1]["event"] == "error" and events[-1]["status"] == 502
    assert "secret" not in json.dumps(events)


def test_stream_quota_error_event():
    c, emb = client_with(QuotaModel())
    with c:
        events = read_stream(c.post("/report/stream", json=body(emb)))
    assert events[-1] == {"event": "error", "status": 429,
                          "message": "The AI model's free daily quota is used up. Try one of the sample X-rays."}


def test_missing_library_without_repo_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CASEFINDER_LIBRARY_REPO", raising=False)
    with pytest.raises(FileNotFoundError, match="CASEFINDER_LIBRARY_REPO"):
        ensure_library(tmp_path / "library")
