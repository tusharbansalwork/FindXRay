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
